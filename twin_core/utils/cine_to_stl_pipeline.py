from pathlib import Path
import argparse
import torch
from tqdm import tqdm

from .time import run_timestamp
from .paths import ensure_dir, make_run_dir, dataset_paths
from .segmentation_model import load_model
from .segmentation_inference import segment_volume
from .mesh_extraction import mask_to_mesh
from twin_core.data_ingestion.dataloaders import load_4d_image


def cine_to_stl(
    dataset_root: Path,
    nifti_path: Path,
    weights_name: str,
    run_name: str = "cine_to_stl_run",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve standardized paths
    paths = dataset_paths(dataset_root)
    weights_path = paths["weights"] / weights_name

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    ts = run_timestamp()
    run_dir = make_run_dir(paths["meshes"], run_name, ts)

    arr, spacing = load_4d_image(nifti_path)
    spacing = spacing[2::-1]  # (Z, Y, X)

    model = load_model(weights_path, device=device)

    patient_id = nifti_path.stem.split("_")[0]
    patient_dir = ensure_dir(run_dir / patient_id)

    for t in tqdm(range(arr.shape[0]), desc=patient_id):
        frame = arr[t]
        mask = segment_volume(model, frame, device=device)

        out_path = patient_dir / f"{patient_id}_t{t:02d}.stl"
        mask_to_mesh(mask, spacing, out_path)

    print(f"Meshes saved to: {patient_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export STL meshes from 4D cine NIfTI using a trained UNet"
    )

    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Root directory of the dataset (contains preprocessed/, weights/, meshes/)",
    )
    parser.add_argument(
        "--nifti",
        required=True,
        help="Path to 4D cine NIfTI file",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Checkpoint filename inside weights/ (e.g. epoch12-val0.1345.ckpt)",
    )
    parser.add_argument(
        "--run_name",
        default="cine_to_stl_run",
        help="Run name for output folder",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda or cpu",
    )

    args = parser.parse_args()

    cine_to_stl(
        dataset_root=Path(args.dataset_root),
        nifti_path=Path(args.nifti),
        weights_name=args.weights,
        run_name=args.run_name,
        device=args.device,
    )
