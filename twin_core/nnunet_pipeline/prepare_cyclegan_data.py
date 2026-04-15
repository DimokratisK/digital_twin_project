"""
Prepare MM-WHS CT and MRI volumes as 2D PNG slices for CycleGAN training.

Extracts axial slices from NIfTI volumes, normalises intensities to [0, 255],
and writes them into the folder layout that CycleGAN expects:

    <output_dir>/
        trainA/   CT slices   (domain A)
        trainB/   MRI slices  (domain B)
        testA/    CT slices   (same images, used for translation after training)

A JSON manifest is written alongside each split so that the translated slices
can later be reassembled into 3-D NIfTI volumes (see translate_and_reassemble.py).

Intensity normalisation:
    CT  — clip to a soft-tissue window [−200, 400] HU, then linear map to [0, 255].
    MRI — per-volume 1st / 99th percentile clip, then linear map to [0, 255].

Empty slices (< 1 % foreground after thresholding) are skipped.

Usage (on GPU VM):
    python -m twin_core.nnunet_pipeline.prepare_cyclegan_data

    # Only CT (skip MRI):
    python -m twin_core.nnunet_pipeline.prepare_cyclegan_data --modality ct

    # Custom output:
    python -m twin_core.nnunet_pipeline.prepare_cyclegan_data --output-dir /tmp/cyc
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from twin_core.nnunet_pipeline.set_environment import set_env_vars, create_directories

set_env_vars()
create_directories()

import os

RAW_BASE = Path(os.environ["nnUNet_raw"])

DATASET_CT = "Dataset029_MMWHS_CT"
DATASET_MRI = "Dataset028_MMWHS"

# Intensity normalisation parameters
CT_WINDOW = (-200, 400)   # Hounsfield units, soft-tissue window
MRI_PCTILE = (1, 99)      # percentile clip for MRI
FOREGROUND_THRESHOLD = 0.01  # skip slices with < 1 % non-zero voxels


def normalise_ct(volume: np.ndarray) -> np.ndarray:
    """Clip to soft-tissue HU window and scale to [0, 255]."""
    lo, hi = CT_WINDOW
    vol = np.clip(volume.astype(np.float32), lo, hi)
    vol = (vol - lo) / (hi - lo) * 255.0
    return vol.astype(np.uint8)


def normalise_mri(volume: np.ndarray) -> np.ndarray:
    """Per-volume percentile clip and scale to [0, 255]."""
    v = volume.astype(np.float32)
    lo = np.percentile(v, MRI_PCTILE[0])
    hi = np.percentile(v, MRI_PCTILE[1])
    if hi - lo < 1e-6:
        return np.zeros_like(v, dtype=np.uint8)
    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo) * 255.0
    return v.astype(np.uint8)


def extract_slices(
    images_dir: Path,
    output_dir: Path,
    modality: str,
    fg_threshold: float = FOREGROUND_THRESHOLD,
) -> list:
    """Extract axial slices from all NIfTI volumes in *images_dir*.

    Returns a manifest: list of dicts with keys
        filename, volume, slice_idx, shape, affine (as nested list).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    normalise = normalise_ct if modality == "ct" else normalise_mri

    manifest = []
    nifti_files = sorted(images_dir.glob("*.nii.gz"))
    if not nifti_files:
        raise FileNotFoundError(f"No .nii.gz files in {images_dir}")

    total_slices = 0
    skipped = 0

    for nifti_path in nifti_files:
        # Volume name: mmwhs_1001_0000.nii.gz -> mmwhs_1001
        vol_name = nifti_path.name.replace("_0000.nii.gz", "")

        img = nib.load(str(nifti_path))
        data = np.asarray(img.dataobj)
        affine = img.affine.tolist()
        norm = normalise(data)

        n_slices = norm.shape[2]  # axial = last axis for MM-WHS
        for z in range(n_slices):
            slc = norm[:, :, z]

            # Skip near-empty slices
            fg_frac = np.count_nonzero(slc) / slc.size
            if fg_frac < fg_threshold:
                skipped += 1
                continue

            fname = f"{vol_name}_z{z:04d}.png"
            Image.fromarray(slc).save(str(output_dir / fname))
            manifest.append({
                "filename": fname,
                "volume": vol_name,
                "slice_idx": z,
                "shape": list(data.shape),
                "affine": affine,
            })
            total_slices += 1

    print(f"  {modality.upper()}: {total_slices} slices from {len(nifti_files)} volumes "
          f"({skipped} empty slices skipped)")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MM-WHS volumes as 2D PNG slices for CycleGAN"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cyclegan/datasets/ct2mri",
        help="Output directory for CycleGAN datasets (default: cyclegan/datasets/ct2mri)",
    )
    parser.add_argument(
        "--modality",
        choices=["both", "ct", "mri"],
        default="both",
        help="Which modality to prepare (default: both)",
    )
    parser.add_argument(
        "--fg-threshold",
        type=float,
        default=FOREGROUND_THRESHOLD,
        help="Skip slices with less than this fraction of foreground (default: 0.01)",
    )
    args = parser.parse_args()
    output_base = Path(args.output_dir)

    print("Preparing CycleGAN data from MM-WHS volumes")
    print(f"  Output: {output_base.resolve()}")

    manifests = {}

    if args.modality in ("both", "ct"):
        ct_images = RAW_BASE / DATASET_CT / "imagesTr"
        print(f"\n--- CT (domain A) ---")
        print(f"  Source: {ct_images}")

        # trainA: CT slices for CycleGAN training
        ct_manifest = extract_slices(
            ct_images, output_base / "trainA", "ct", args.fg_threshold
        )
        manifests["trainA"] = ct_manifest

        # testA: same CT slices (used for translation after CycleGAN training)
        # Symlink or copy — use copy for portability
        print("  Copying CT slices to testA/ for translation...")
        test_a = output_base / "testA"
        test_a.mkdir(parents=True, exist_ok=True)
        import shutil
        for entry in ct_manifest:
            src = output_base / "trainA" / entry["filename"]
            shutil.copy2(str(src), str(test_a / entry["filename"]))
        manifests["testA"] = ct_manifest  # same manifest
        print(f"  testA: {len(ct_manifest)} slices")

    if args.modality in ("both", "mri"):
        mri_images = RAW_BASE / DATASET_MRI / "imagesTr"
        print(f"\n--- MRI (domain B) ---")
        print(f"  Source: {mri_images}")

        mri_manifest = extract_slices(
            mri_images, output_base / "trainB", "mri", args.fg_threshold
        )
        manifests["trainB"] = mri_manifest

        # testB: not strictly needed for CT→MRI translation, but CycleGAN expects it
        test_b = output_base / "testB"
        test_b.mkdir(parents=True, exist_ok=True)

    # Save manifests for reassembly
    for split, manifest in manifests.items():
        manifest_path = output_base / f"{split}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n  Manifest saved: {manifest_path} ({len(manifest)} entries)")

    print(f"\nDone. CycleGAN data ready at {output_base.resolve()}")
    print("\nTo train CycleGAN:")
    print(f"  cd cyclegan")
    print(f"  python train.py \\")
    print(f"    --dataroot ./datasets/ct2mri \\")
    print(f"    --name ct2mri_cardiac \\")
    print(f"    --model cycle_gan \\")
    print(f"    --input_nc 1 --output_nc 1 \\")
    print(f"    --no_dropout \\")
    print(f"    --load_size 286 --crop_size 256 \\")
    print(f"    --n_epochs 100 --n_epochs_decay 100 \\")
    print(f"    --gpu_ids 0 --no_html")


if __name__ == "__main__":
    main()
