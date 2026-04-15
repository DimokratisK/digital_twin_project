"""
Create an nnU-Net dataset from CycleGAN-translated fake MRI volumes.

Takes the reassembled NIfTI volumes (CT translated to look like MRI) and
pairs them with the original CT segmentation labels to create
Dataset030_MMWHS_FakeMRI in nnU-Net v2 format.

The key insight: CycleGAN only changes appearance, not anatomy.
The segmentation labels from the original CT volumes are still valid.

Usage (on GPU VM):
    python -m twin_core.nnunet_pipeline.convert_fakemri_dataset

    # Custom paths:
    python -m twin_core.nnunet_pipeline.convert_fakemri_dataset \\
        --fakemri-dir outputs/fakemri_volumes \\
        --dataset-id 30
"""

import argparse
import shutil
from pathlib import Path

from twin_core.nnunet_pipeline.set_environment import set_env_vars, create_directories

set_env_vars()
create_directories()

import os

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

RAW_BASE = Path(nnUNet_raw)

# Same label names as CT (Dataset029) — anatomy is preserved by CycleGAN
LABEL_NAMES = {
    "background": 0,
    "LV_Myo": 1,
    "LA": 2,
    "LV": 3,
    "RA": 4,
    "RV": 5,
    "Aorta": 6,
    "PA": 7,
}

# Source of ground truth labels (original CT dataset)
SOURCE_CT_DATASET = "Dataset029_MMWHS_CT"


def convert_fakemri(
    fakemri_dir: str,
    dataset_id: int = 30,
    dataset_suffix: str = "MMWHS_FakeMRI",
):
    """
    Create nnU-Net dataset from fake MRI volumes + original CT labels.

    Args:
        fakemri_dir: Path to reassembled fake MRI NIfTI volumes.
        dataset_id: nnU-Net dataset ID (default: 30).
        dataset_suffix: Suffix for dataset name.
    """
    fakemri_dir = Path(fakemri_dir)
    ct_labels_dir = RAW_BASE / SOURCE_CT_DATASET / "labelsTr"

    if not fakemri_dir.exists():
        raise FileNotFoundError(
            f"Fake MRI volumes not found at {fakemri_dir}\n"
            f"Run translate_and_reassemble.py first."
        )
    if not ct_labels_dir.exists():
        raise FileNotFoundError(
            f"CT labels not found at {ct_labels_dir}\n"
            f"Dataset029_MMWHS_CT must exist in nnUNet_raw."
        )

    dataset_name = f"Dataset{dataset_id:03d}_{dataset_suffix}"
    out_dir = RAW_BASE / dataset_name
    out_images = out_dir / "imagesTr"
    out_labels = out_dir / "labelsTr"

    # Clean previous conversion
    if out_dir.exists():
        print(f"Removing existing dataset at {out_dir}")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True)
    out_images.mkdir()
    out_labels.mkdir()

    # Discover fake MRI volumes
    fake_volumes = sorted(fakemri_dir.glob("*.nii.gz"))
    if not fake_volumes:
        raise FileNotFoundError(f"No .nii.gz files found in {fakemri_dir}")

    print(f"Found {len(fake_volumes)} fake MRI volumes")
    num_cases = 0

    for vol_path in fake_volumes:
        # Volume name: mmwhs_1001.nii.gz
        case_id = vol_path.stem.replace(".nii", "")  # handle .nii.gz

        # Check that matching CT label exists
        label_path = ct_labels_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            print(f"  WARNING: No CT label for {case_id}, skipping")
            continue

        # Copy fake MRI image with _0000 channel suffix
        shutil.copy(vol_path, out_images / f"{case_id}_0000.nii.gz")

        # Copy original CT label (already remapped to contiguous 1-7)
        shutil.copy(label_path, out_labels / f"{case_id}.nii.gz")

        print(f"  {case_id}: image + label copied")
        num_cases += 1

    # Generate dataset.json
    generate_dataset_json(
        str(out_dir),
        channel_names={0: "FakeMRI_from_CT"},
        labels=LABEL_NAMES,
        num_training_cases=num_cases,
        file_ending=".nii.gz",
        dataset_name=dataset_suffix,
        description=(
            f"MM-WHS CycleGAN domain-adapted: CT→fake MRI appearance, "
            f"original CT labels — {num_cases} patients, 7 structures"
        ),
        license="See MM-WHS challenge terms",
        reference="Zhuang, IEEE TPAMI 2019; Zhu et al., ICCV 2017 (CycleGAN)",
        converted_by="twin_core.nnunet_pipeline.convert_fakemri_dataset",
    )

    print(f"\nDataset created: {dataset_name}")
    print(f"  Cases: {num_cases}")
    print(f"  Output: {out_dir}")
    print(f"\nNext steps:")
    print(f"  nnUNetv2_plan_and_preprocess -d {dataset_id} -np 1 1 1 --verify_dataset_integrity")
    print(f"  nnUNetv2_train {dataset_id} 3d_fullres 0")

    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create nnU-Net dataset from CycleGAN fake MRI + CT labels"
    )
    parser.add_argument(
        "--fakemri-dir",
        type=str,
        default="outputs/fakemri_volumes",
        help="Directory with reassembled fake MRI NIfTI volumes",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=30,
        help="nnU-Net dataset ID (default: 30)",
    )
    args = parser.parse_args()

    convert_fakemri(
        fakemri_dir=args.fakemri_dir,
        dataset_id=args.dataset_id,
    )


if __name__ == "__main__":
    main()
