#!/usr/bin/env python3
"""
cycle_geometries_extraction_pipeline.py

Extract per-frame STL meshes from 4D cine NIfTI volumes using a trained
multi-class segmentation model.

Key behaviors:
- Loads a trained segmentation model (multi-class) via twin_core.utils.segmentation_model.load_model.
- Runs inference per 2D frame (or per-slice stack if segment_volume supports 3D).
- Converts labeled segmentation volumes to STL meshes per requested class using mask_to_mesh.
- Saves one STL per requested class per time frame with clear filenames.
- Optional: extract a single template mesh (e.g., ED) for later correspondence/registration.

Assumptions:
- `segment_volume(model, frame, device)` returns an integer label array (Z, Y, X) or (Y, X) for 2D.
- `mask_to_mesh(label_or_binary_mask, spacing, out_path, class_id=None, **kwargs)` accepts either:
    - a binary mask (0/1) and no class_id, or
    - a labeled volume and a class_id to extract that class.
"""

from pathlib import Path
import argparse
from typing import Sequence, List, Optional, Tuple

import torch
from tqdm import tqdm

from twin_core.data_ingestion.dataloaders import load_4d_image
from .segmentation_model import load_model
from .segmentation_inference import segment_volume
from .mesh_extraction import mask_to_mesh
from .paths import dataset_paths, ensure_dir


def _parse_class_list(s: str) -> List[int]:
    """
    Parse a comma-separated list of integers like "1,2,3" into [1,2,3].
    """
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _ensure_out_dirs(base_out: Path, patient_id: str, class_ids: Sequence[int]) -> dict:
    """
    Create and return a mapping class_id -> directory for that class under patient folder.
    """
    patient_dir = ensure_dir(base_out / patient_id)
    class_dirs = {}
    for cid in class_ids:
        d = ensure_dir(patient_dir / f"class_{cid}")
        class_dirs[cid] = d
    return class_dirs


def _save_template_mesh(patient_out: Path, class_id: int, mesh_path: Path) -> None:
    """
    Save or copy the template mesh path to a canonical filename for later use.
    """
    template_path = patient_out / f"template_class{class_id}.stl"
    try:
        # If mesh_path exists, copy it to template_path (best-effort)
        if mesh_path.exists():
            import shutil
            shutil.copy(mesh_path, template_path)
    except Exception:
        # ignore copy errors; template is optional
        pass


def process_patient(
    nifti_path: Path,
    model,
    device: str,
    out_dir: Path,
    class_ids: Sequence[int],
    template_frame: Optional[int] = None,
    postprocess: bool = True,
    smoothing_iterations: int = 5,
):
    """
    Process a single patient NIfTI:
    - load 4D image (T, Z, Y, X)
    - for each time frame, run segmentation inference -> labeled volume
    - for each requested class_id, extract binary mask and call mask_to_mesh
    - save per-class STL files under out_dir/<patient_id>/class_<cid>/
    - optionally save a template mesh for the template_frame
    """
    arr, spacing = load_4d_image(nifti_path)  # arr shape: (T, Z, Y, X) or (T, Y, X)
    # spacing returned as (sx, sy, sz) or similar; mask_to_mesh expects (Z, Y, X) ordering
    # The original code used spacing = spacing[2::-1] to reorder; keep that behavior
    spacing = tuple(spacing[2::-1])

    patient_id = nifti_path.stem.split("_")[0]
    class_dirs = _ensure_out_dirs(out_dir, patient_id, class_ids)

    # Iterate frames
    for t in tqdm(range(arr.shape[0]), desc=f"Patient {patient_id}"):
        frame = arr[t]  # shape: (Z, Y, X) or (Y, X)
        # Run segmentation inference: should return integer labels (Z, Y, X) or (Y, X)
        labels = segment_volume(model, frame, device=device)

        # Ensure labels is a numpy array
        # (segment_volume is expected to return np.ndarray)
        # For each requested class, extract binary mask and save mesh
        for cid in class_ids:
            # If labels is 3D (Z, Y, X), extract per-slice or whole-volume mesh depending on mask_to_mesh
            # We pass the labeled volume and class_id to mask_to_mesh; mask_to_mesh will handle binary extraction.
            out_path = class_dirs[cid] / f"{patient_id}_t{t:02d}_class{cid}.stl"
            try:
                # mask_to_mesh should accept (labels, spacing, out_path, class_id=cid, smoothing_iterations=...)
                mask_to_mesh(labels, spacing, out_path, class_id=cid, postprocess=postprocess, smoothing_iterations=smoothing_iterations)
            except TypeError:
                # Fallback if mask_to_mesh expects a binary mask only
                import numpy as _np
                binary = (_np.asarray(labels) == cid).astype(_np.uint8)
                mask_to_mesh(binary, spacing, out_path, postprocess=postprocess, smoothing_iterations=smoothing_iterations)

        # If this frame is the template_frame, copy one of the class meshes as template
        if template_frame is not None and t == template_frame:
            # Save template for each class (best-effort)
            for cid in class_ids:
                mesh_path = class_dirs[cid] / f"{patient_id}_t{t:02d}_class{cid}.stl"
                _save_template_mesh(out_dir / patient_id, cid, mesh_path)


def main(
    dataset_root: Path,
    checkpoint_name: str,
    class_list: Sequence[int],
    template_frame: Optional[int],
    device: Optional[str] = None,
    postprocess: bool = True,
    smoothing_iterations: int = 5,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = dataset_paths(dataset_root)
    raw_dir = paths["raw"]
    meshes_dir = ensure_dir(paths["meshes"])
    weights_dir = paths["weights"]

    checkpoint_path = weights_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model (expects a model that outputs multi-class logits and whose checkpoint was saved as state_dict)
    model = load_model(checkpoint_path=checkpoint_path, device=device)

    # Find patient NIfTIs
    patients = sorted(raw_dir.glob("patient*_4d.nii.gz"))
    if not patients:
        raise RuntimeError(f"No patient NIfTI files found in {raw_dir}")

    # Process each patient
    for p in patients:
        process_patient(
            nifti_path=p,
            model=model,
            device=device,
            out_dir=meshes_dir,
            class_ids=class_list,
            template_frame=template_frame,
            postprocess=postprocess,
            smoothing_iterations=smoothing_iterations,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract STL meshes for all frames in 4D cine volumes (multi-class)")

    parser.add_argument("--dataset_root", required=True, help="Dataset root containing raw/, weights/, meshes/")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint filename inside weights/")
    parser.add_argument("--classes", default="3,2", help="Comma-separated class ids to export (e.g. '3,2' for LV cavity and myocardium)")
    parser.add_argument("--template_frame", type=int, default=None, help="Optional frame index to save as template (e.g., ED frame index)")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--no_postprocess", action="store_true", help="Disable postprocessing (hole fill, largest component, smoothing)")
    parser.add_argument("--smoothing_iterations", type=int, default=5, help="Smoothing iterations applied in mesh postprocessing")

    args = parser.parse_args()

    class_ids = _parse_class_list(args.classes)
    main(
        dataset_root=Path(args.dataset_root),
        checkpoint_name=args.checkpoint,
        class_list=class_ids,
        template_frame=args.template_frame,
        device=args.device,
        postprocess=not args.no_postprocess,
        smoothing_iterations=args.smoothing_iterations,
    )
