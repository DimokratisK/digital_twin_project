#!/usr/bin/env python3
"""
cine_to_stl_pipeline.py

Export time-resolved STL meshes from a 4D cine NIfTI using a trained UNet.

Configuration is loaded from a JSON file (use --config like the training script).
CLI args override config fields.

Outputs:
  <meshes_root>/<run_name>_<timestamp>/<patient_id>/<label_name>/<label_name>_tXX.stl
  optionally: <...>/<patient_id>/combined/<patient_id>_tXX_combined.stl
"""
from pathlib import Path
import json
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm
import torch

from .time import run_timestamp
from .paths import ensure_dir, make_run_dir, dataset_paths
from .segmentation_model import load_model
from .segmentation_inference import segment_volume
from .mesh_extraction import mask_to_mesh
from twin_core.data_ingestion.dataloaders import load_4d_image


def _parse_labels_arg(labels_cfg: Optional[Any]) -> Dict[str, int]:
    """
    Accept either:
      - a dict already parsed from JSON: {"LV":3,"MYO":2,"RV":1}
      - a JSON-like string: '{"LV":3,"MYO":2,"RV":1}'
      - a comma-separated mapping string: "LV:3,MYO:2,RV:1"
      - None -> returns default mapping
    """
    if labels_cfg is None:
        return {"RV": 1, "MYO": 2, "LV": 3}

    # If already a dict (from config JSON), coerce values to int
    if isinstance(labels_cfg, dict):
        return {str(k): int(v) for k, v in labels_cfg.items()}

    # If it's a string, try JSON parse then fallback to comma list
    if isinstance(labels_cfg, str):
        try:
            parsed = json.loads(labels_cfg)
            if isinstance(parsed, dict):
                return {str(k): int(v) for k, v in parsed.items()}
        except Exception:
            pass
        out = {}
        for part in labels_cfg.split(","):
            if not part.strip():
                continue
            if ":" not in part:
                raise ValueError(f"Invalid label mapping fragment: {part!r}; expected NAME:ID")
            name, sid = part.split(":", 1)
            out[name.strip()] = int(sid.strip())
        return out

    raise ValueError("labels_to_export must be dict, JSON string, or comma list")


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def cine_to_stl(
    dataset_root: Path,
    nifti_path: Path,
    weights_name: str,
    run_name: str = "cine_to_stl_run",
    device: Optional[str] = None,
    labels_to_export: Optional[Dict[str, int]] = None,
    export_combined: bool = False,
    batch_slices: Optional[int] = None,
):
    """
    Main function to export meshes.

    dataset_root: project root (used to resolve weights and meshes folders via dataset_paths)
    nifti_path: path to the 4D NIfTI file to segment
    weights_name: filename of checkpoint inside weights/ (or absolute path)
    run_name: human-readable run name; timestamp appended automatically
    device: "cuda" or "cpu"; if None choose automatically
    labels_to_export: dict name->label_id mapping; default {'RV':1,'MYO':2,'LV':3}
    export_combined: whether to also export combined non-zero mesh per frame
    batch_slices: forwarded to segment_volume for batched inference
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve standardized paths and weight file
    paths = dataset_paths(dataset_root)
    weights_path = Path(weights_name)
    if not weights_path.exists():
        weights_path = paths["weights"] / weights_name
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    ts = run_timestamp()
    run_dir = make_run_dir(paths["meshes"], run_name, ts)

    print(f"[cine_to_stl] loading 4D image: {nifti_path}")
    arr, spacing = load_4d_image(nifti_path)  # expected arr shape (T, Z, Y, X)
    if arr.ndim != 4:
        raise RuntimeError(f"Expected 4D image (T,Z,Y,X); got shape {arr.shape}")

    # Convert spacing to (dz, dy, dx) to match mask axes (Z,Y,X)
    try:
        spacing = tuple(spacing[2::-1])
    except Exception:
        spacing = tuple(spacing)[:3][::-1]

    model = load_model(weights_path, device=device)
    model_device = next(model.parameters()).device
    print(f"[cine_to_stl] model loaded on device {model_device}; running inference on {device}")

    patient_id = nifti_path.stem.split("_")[0]
    patient_dir = ensure_dir(run_dir / patient_id)

    if labels_to_export is None:
        labels_to_export = {"RV": 1, "MYO": 2, "LV": 3}

    # create per-label directories
    label_dirs = {}
    for name in labels_to_export.keys():
        label_dirs[name] = ensure_dir(patient_dir / name)
    combined_dir = ensure_dir(patient_dir / "combined") if export_combined else None

    # Loop frames
    T = arr.shape[0]
    print(f"[cine_to_stl] frames={T}, spacing(Z,Y,X)={spacing}")
    for t in tqdm(range(T), desc=f"{patient_id} frames"):
        frame = arr[t]  # shape (Z, Y, X)

        # Run segmentation; segment_volume accepts (Z,Y,X) and returns integer labels (Z,Y,X)
        mask = segment_volume(model, frame, device=device, batch_slices=batch_slices)

        # Basic validation
        if not isinstance(mask, np.ndarray):
            raise RuntimeError(f"segment_volume returned {type(mask)} instead of numpy array")
        if mask.ndim != 3:
            # If model returned (H,W) for single-slice, expand to (1,H,W)
            if mask.ndim == 2:
                mask = mask[np.newaxis, :, :]
            else:
                raise RuntimeError(f"segment_volume must return 3D volume (Z,Y,X). Got shape: {mask.shape}")

        # combined (non-zero) export
        if export_combined:
            out_comb = combined_dir / f"{patient_id}_t{t:02d}_combined.stl"
            try:
                mask_to_mesh(mask, spacing, out_comb, class_id=None)
            except Exception as e:
                print(f"[warn] combined mesh failed for t{t}: {e}")

        # per-label exports
        for label_name, label_id in labels_to_export.items():
            out_path = label_dirs[label_name] / f"{label_name}_t{t:02d}.stl"
            try:
                mesh = mask_to_mesh(mask, spacing, out_path, class_id=int(label_id))
                if mesh is None:
                    pass
            except Exception as e:
                print(f"[warn] mesh export failed for label={label_id} ({label_name}) t={t}: {e}")

    print(f"[cine_to_stl] meshes saved to: {patient_dir}")


def _merge_config_and_args(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    CLI args override config file. Convert JSON null -> None where appropriate.
    """
    out = dict(cfg)  # base
    # map CLI args (only override if provided)
    if args.dataset_root:
        out["dataset_root"] = args.dataset_root
    if args.nifti:
        out["nifti"] = args.nifti
    if args.weights:
        out["weights"] = args.weights
    if args.run_name:
        out["run_name"] = args.run_name
    if args.device is not None:
        out["device"] = args.device
    if args.labels_to_export is not None:
        out["labels_to_export"] = args.labels_to_export
    if args.export_combined is not None:
        out["export_combined"] = args.export_combined
    if args.batch_slices is not None:
        out["batch_slices"] = args.batch_slices
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export STL meshes from a 4D cine NIfTI using a trained UNet")
    parser.add_argument("--config", help="Path to JSON config file", required=False)
    parser.add_argument("--dataset_root", help="Root directory of the dataset (contains preprocessed/, weights/, meshes/)")
    parser.add_argument("--nifti", help="Path to 4D cine NIfTI file")
    parser.add_argument("--weights", help="Checkpoint filename inside weights/ (e.g. epoch12-val0.1345.ckpt)")
    parser.add_argument("--run_name", help="Run name for output folder")
    parser.add_argument("--device", help="cuda or cpu; overrides config")
    parser.add_argument("--labels_to_export", help="JSON string or comma list mapping NAME:ID (overrides config)")
    parser.add_argument("--export_combined", action="store_true", help="Also export combined non-zero mesh per frame")
    parser.add_argument("--batch_slices", type=int, help="If provided, forward to segment_volume for batched inference")

    args = parser.parse_args()

    cfg = _load_config(args.config) if args.config else {}
    merged = _merge_config_and_args(cfg, args)

    # Required fields check
    if "dataset_root" not in merged or "nifti" not in merged or "weights" not in merged:
        raise RuntimeError("Config must provide dataset_root, nifti, and weights (or pass via CLI)")

    # Prepare typed values
    dataset_root = Path(merged["dataset_root"])
    nifti_path = Path(merged["nifti"])
    weights_name = merged["weights"]
    run_name = merged.get("run_name", "cine_to_stl_run")
    device = merged.get("device", None)
    labels_to_export = _parse_labels_arg(merged.get("labels_to_export", None))
    export_combined = bool(merged.get("export_combined", False))
    batch_slices = merged.get("batch_slices", None)
    if batch_slices is not None:
        batch_slices = int(batch_slices)

    cine_to_stl(
        dataset_root=dataset_root,
        nifti_path=nifti_path,
        weights_name=weights_name,
        run_name=run_name,
        device=device,
        labels_to_export=labels_to_export,
        export_combined=export_combined,
        batch_slices=batch_slices,
    )
