"""
Convert nnU-Net prediction masks (.nii.gz) to STL meshes.

Takes the output folder from run_inference.py and generates per-structure STL
meshes using marching cubes + smoothing + repair.

Usage:
    # Convert all predictions in a folder (ACDC labels):
    python -m twin_core.nnunet_pipeline.predictions_to_stl \
        -i nnunet_data/results/predictions -o meshes/acdc --dataset acdc

    # Convert with MM-WHS labels:
    python -m twin_core.nnunet_pipeline.predictions_to_stl \
        -i nnunet_data/results/predictions -o meshes/mmwhs --dataset mmwhs

    # Convert a single file:
    python -m twin_core.nnunet_pipeline.predictions_to_stl \
        -i predictions/patient001_frame01.nii.gz -o meshes/patient001 --dataset acdc
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np

from twin_core.utils.mesh_extraction import mask_to_mesh


# Label definitions per dataset
DATASET_LABELS = {
    "acdc": {
        "RV": 1,
        "MYO": 2,
        "LV": 3,
    },
    "mmwhs": {
        "LV_Myo": 1,
        "LA": 2,
        "LV": 3,
        "RA": 4,
        "RV": 5,
        "Aorta": 6,
        "PA": 7,
    },
}


def convert_prediction_to_stl(
    nifti_path: Path,
    output_dir: Path,
    labels: Dict[str, int],
    smoothing_iterations: int = 5,
    smoothing_method: str = "taubin",
    decimate_ratio: Optional[float] = None,
):
    """Convert a single nnU-Net prediction .nii.gz to per-structure STL files.

    Parameters
    ----------
    nifti_path : Path to the prediction .nii.gz file
    output_dir : Directory to save STL files
    labels : dict mapping structure name -> label integer
    smoothing_iterations : number of smoothing passes
    smoothing_method : 'taubin' or 'laplacian'
    decimate_ratio : fraction of faces to keep (None = no decimation)
    """
    img = nib.load(str(nifti_path))
    mask = np.asarray(img.dataobj, dtype=np.int16)
    spacing = tuple(img.header.get_zooms()[:3])

    # NIfTI is (X, Y, Z) but mask_to_mesh expects (Z, Y, X)
    mask = np.transpose(mask, (2, 1, 0))
    spacing = spacing[::-1]  # reverse to (dz, dy, dx)

    case_name = nifti_path.name.replace(".nii.gz", "")
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    unique_labels = set(np.unique(mask)) - {0}
    exported = []

    for struct_name, label_id in labels.items():
        if label_id not in unique_labels:
            continue

        out_path = case_dir / f"{struct_name}.stl"
        mesh = mask_to_mesh(
            mask=mask,
            spacing=spacing,
            out_path=out_path,
            class_id=label_id,
            postprocess=True,
            smoothing_iterations=smoothing_iterations,
            smoothing_method=smoothing_method,
            decimate_ratio=decimate_ratio,
        )
        if mesh is not None:
            exported.append(struct_name)

    return exported


def convert_folder(
    input_dir: Path,
    output_dir: Path,
    labels: Dict[str, int],
    smoothing_iterations: int = 5,
    smoothing_method: str = "taubin",
    decimate_ratio: Optional[float] = None,
):
    """Convert all .nii.gz predictions in a folder to STL meshes."""
    nifti_files = sorted(input_dir.glob("*.nii.gz"))
    if not nifti_files:
        print(f"No .nii.gz files found in {input_dir}")
        return

    print(f"Found {len(nifti_files)} prediction(s) in {input_dir}")
    print(f"Labels: {labels}")
    print(f"Output: {output_dir}\n")

    for nifti_path in nifti_files:
        print(f"Processing {nifti_path.name}...")
        exported = convert_prediction_to_stl(
            nifti_path=nifti_path,
            output_dir=output_dir,
            labels=labels,
            smoothing_iterations=smoothing_iterations,
            smoothing_method=smoothing_method,
            decimate_ratio=decimate_ratio,
        )
        if exported:
            print(f"  Exported: {', '.join(exported)}")
        else:
            print(f"  No structures found in mask")

    print(f"\nDone. STL meshes saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert nnU-Net predictions to STL meshes"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Input: prediction .nii.gz file or folder containing predictions"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output directory for STL meshes"
    )
    parser.add_argument(
        "--dataset", type=str, choices=list(DATASET_LABELS.keys()), required=True,
        help="Dataset type for label mapping"
    )
    parser.add_argument(
        "--smoothing_iterations", type=int, default=5,
        help="Smoothing iterations (default: 5)"
    )
    parser.add_argument(
        "--smoothing_method", type=str, default="taubin",
        choices=["taubin", "laplacian"],
        help="Smoothing method (default: taubin)"
    )
    parser.add_argument(
        "--decimate_ratio", type=float, default=None,
        help="Fraction of faces to keep (e.g. 0.5 for 50%%). None = no decimation"
    )

    args = parser.parse_args()
    labels = DATASET_LABELS[args.dataset]
    input_path = Path(args.input)

    if input_path.is_file():
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        exported = convert_prediction_to_stl(
            nifti_path=input_path,
            output_dir=output_dir,
            labels=labels,
            smoothing_iterations=args.smoothing_iterations,
            smoothing_method=args.smoothing_method,
            decimate_ratio=args.decimate_ratio,
        )
        if exported:
            print(f"Exported: {', '.join(exported)}")
        else:
            print("No structures found in mask")
    elif input_path.is_dir():
        convert_folder(
            input_dir=input_path,
            output_dir=Path(args.output),
            labels=labels,
            smoothing_iterations=args.smoothing_iterations,
            smoothing_method=args.smoothing_method,
            decimate_ratio=args.decimate_ratio,
        )
    else:
        print(f"Error: {input_path} is not a file or directory")


if __name__ == "__main__":
    main()
