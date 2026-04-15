"""
Reassemble CycleGAN-translated 2D PNG slices back into 3D NIfTI volumes.

After CycleGAN training and translation (test.py), the fake MRI slices live at:
    cyclegan/results/<experiment>/test_latest/images/<stem>_fake_B.png

This script:
    1. Reads the manifest written by prepare_cyclegan_data.py
    2. Collects the corresponding fake_B PNGs per volume
    3. Reassembles each volume into a 3D NIfTI with the original affine/shape
    4. Writes to an output directory, ready for convert_fakemri_dataset.py

Usage (on GPU VM):
    # After CycleGAN translation:
    python -m twin_core.nnunet_pipeline.translate_and_reassemble

    # Custom paths:
    python -m twin_core.nnunet_pipeline.translate_and_reassemble \\
        --manifest cyclegan/datasets/ct2mri/testA_manifest.json \\
        --results-dir cyclegan/results/ct2mri_cardiac/test_latest/images \\
        --output-dir outputs/fakemri_volumes
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def reassemble_volumes(
    manifest_path: Path,
    results_dir: Path,
    output_dir: Path,
    suffix: str = "fake_B",
):
    """Reassemble translated 2D slices into 3D NIfTI volumes.

    Args:
        manifest_path: Path to the JSON manifest from prepare_cyclegan_data.py.
        results_dir: Directory containing CycleGAN output PNGs (*_fake_B.png).
        output_dir: Where to write the reassembled NIfTI volumes.
        suffix: CycleGAN output suffix (default: "fake_B" for A→B translation).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group manifest entries by volume
    volumes = defaultdict(list)
    vol_meta = {}  # shape and affine per volume
    for entry in manifest:
        vol = entry["volume"]
        volumes[vol].append(entry)
        if vol not in vol_meta:
            vol_meta[vol] = {
                "shape": entry["shape"],
                "affine": np.array(entry["affine"]),
            }

    print(f"Reassembling {len(volumes)} volumes from {results_dir}")

    for vol_name, entries in sorted(volumes.items()):
        meta = vol_meta[vol_name]
        shape = meta["shape"]
        affine = meta["affine"]

        # Initialise output volume (uint8, same as normalised input)
        volume = np.zeros(shape, dtype=np.uint8)

        missing = 0
        filled = 0
        for entry in sorted(entries, key=lambda e: e["slice_idx"]):
            stem = Path(entry["filename"]).stem  # e.g. mmwhs_1001_z0042
            fake_path = results_dir / f"{stem}_{suffix}.png"

            if not fake_path.exists():
                missing += 1
                continue

            slc = np.array(Image.open(str(fake_path)).convert("L"))

            # CycleGAN may have resized the slice (load_size/crop_size).
            # Resize back to original spatial dims if needed.
            orig_h, orig_w = shape[0], shape[1]
            if slc.shape != (orig_h, orig_w):
                slc = np.array(
                    Image.fromarray(slc).resize((orig_w, orig_h), Image.BILINEAR)
                )

            volume[:, :, entry["slice_idx"]] = slc
            filled += 1

        # Save as NIfTI
        out_path = output_dir / f"{vol_name}.nii.gz"
        nii = nib.Nifti1Image(volume, affine)
        nib.save(nii, str(out_path))

        print(f"  {vol_name}: {filled} slices filled, {missing} missing, "
              f"shape {tuple(shape)} → {out_path.name}")

    print(f"\nDone. {len(volumes)} volumes saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Reassemble CycleGAN-translated slices into NIfTI volumes"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="cyclegan/datasets/ct2mri/testA_manifest.json",
        help="Path to the testA manifest JSON from prepare_cyclegan_data.py",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="cyclegan/results/ct2mri_cardiac/test_latest/images",
        help="Directory containing CycleGAN output PNGs (fake_B images)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/fakemri_volumes",
        help="Where to write reassembled NIfTI volumes",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="fake_B",
        help="CycleGAN output suffix (default: fake_B for A→B translation)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run prepare_cyclegan_data.py first."
        )
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            f"Run CycleGAN test.py first:\n"
            f"  cd cyclegan && python test.py \\\n"
            f"    --dataroot ./datasets/ct2mri \\\n"
            f"    --name ct2mri_cardiac \\\n"
            f"    --model test --no_dropout \\\n"
            f"    --input_nc 1 --output_nc 1 \\\n"
            f"    --preprocess none \\\n"
            f"    --num_test 99999"
        )

    reassemble_volumes(manifest_path, results_dir, output_dir, args.suffix)


if __name__ == "__main__":
    main()
