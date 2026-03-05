"""
Split a 4D cine NIfTI into individual 3D frame files for nnU-Net inference.

nnU-Net expects each input as a separate 3D .nii.gz file with a _0000 channel
suffix. This script takes a 4D cine MRI (X, Y, Z, T) and writes one file per
temporal frame.

Usage:
    python -m twin_core.nnunet_pipeline.split_4d_to_frames \
        -i /path/to/patient006_4d.nii.gz \
        -o /path/to/frames_output/
"""

import argparse
from pathlib import Path

import nibabel as nib


def split_4d(nifti_path: Path, output_dir: Path) -> int:
    """Split a 4D NIfTI into per-frame 3D files.

    Parameters
    ----------
    nifti_path : Path to the 4D .nii.gz file
    output_dir : Directory to save individual frame files

    Returns
    -------
    Number of frames written.
    """
    nifti_path = Path(nifti_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(nifti_path))
    data = img.get_fdata()

    if data.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI, got {data.ndim}D with shape {data.shape}")

    T = data.shape[-1]

    # Extract patient ID from filename (e.g. patient006_4d.nii.gz -> patient006)
    stem = nifti_path.name.replace(".nii.gz", "")
    patient_id = stem.replace("_4d", "")

    print(f"Input: {nifti_path.name}")
    print(f"Shape: {data.shape} ({T} frames)")
    print(f"Patient ID: {patient_id}")
    print(f"Output: {output_dir}\n")

    for t in range(T):
        frame_data = data[..., t]
        frame_img = nib.Nifti1Image(frame_data, img.affine, img.header)
        out_path = output_dir / f"{patient_id}_frame{t:02d}_0000.nii.gz"
        nib.save(frame_img, str(out_path))
        print(f"  Frame {t:2d} -> {out_path.name}")

    print(f"\nDone. {T} frames saved to {output_dir}")
    return T


def main():
    parser = argparse.ArgumentParser(
        description="Split a 4D cine NIfTI into per-frame 3D files for nnU-Net"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to the 4D cine NIfTI file (e.g. patient006_4d.nii.gz)"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output directory for individual frame files"
    )

    args = parser.parse_args()
    split_4d(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
