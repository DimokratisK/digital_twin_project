"""
Convert MM-WHS (Multi-Modality Whole Heart Segmentation) MRI dataset to nnU-Net v2 format.

The MM-WHS dataset uses non-standard label values (205, 420, 500, etc.)
which must be remapped to contiguous integers (1-7) for nnU-Net.

Source structure:
    mr_train/
        mr_train_1001_image.nii.gz
        mr_train_1001_label.nii.gz
        ...
        mr_train_1020_image.nii.gz
        mr_train_1020_label.nii.gz

Target structure (nnU-Net v2):
    nnunet_data/raw/Dataset028_MMWHS/
        imagesTr/
            mmwhs_1001_0000.nii.gz
            ...
        labelsTr/
            mmwhs_1001.nii.gz
            ...
        dataset.json

Usage:
    python -m twin_core.nnunet_pipeline.convert_mmwhs_dataset \
        --dataset_root "C:\...\MM-WHS 2017 Dataset\mr_train"

    # Smoke test (only 5 patients):
    python -m twin_core.nnunet_pipeline.convert_mmwhs_dataset \
        --dataset_root "C:\...\MM-WHS 2017 Dataset\mr_train" \
        --smoke_test 5
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from twin_core.nnunet_pipeline.set_environment import set_env_vars, create_directories
set_env_vars()
create_directories()

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

# MM-WHS original label values → nnU-Net contiguous labels
LABEL_MAPPING = {
    0:   0,   # background
    205: 1,   # LV myocardium
    420: 2,   # Left atrium
    421: 2,   # Left atrium (annotation artifact in patient 1010)
    500: 3,   # LV cavity
    550: 4,   # Right atrium
    600: 5,   # RV cavity
    820: 6,   # Ascending aorta
    850: 7,   # Pulmonary artery
}

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


def remap_labels(label_path: Path, output_path: Path):
    """Load a label NIfTI, remap MM-WHS values to contiguous integers, and save."""
    img = nib.load(str(label_path))
    data = np.asarray(img.dataobj, dtype=np.int16)

    remapped = np.zeros_like(data, dtype=np.uint8)
    unknown_values = set()

    for orig_val, new_val in LABEL_MAPPING.items():
        remapped[data == orig_val] = new_val

    # Check for unexpected label values
    all_original_values = set(np.unique(data))
    expected_values = set(LABEL_MAPPING.keys())
    unknown_values = all_original_values - expected_values
    if unknown_values:
        print(f"  WARNING: Unexpected label values in {label_path.name}: {unknown_values}")

    out_img = nib.Nifti1Image(remapped, img.affine, img.header)
    nib.save(out_img, str(output_path))


def convert_mmwhs(
    dataset_root: str,
    dataset_id: int = 28,
    smoke_test: Optional[int] = None,
):
    """
    Convert MM-WHS MRI dataset to nnU-Net v2 format.

    Args:
        dataset_root: Path to the mr_train folder containing *_image.nii.gz and *_label.nii.gz files.
        dataset_id: nnU-Net dataset ID (default: 28).
        smoke_test: If set, only convert this many patients.
    """
    dataset_root = Path(dataset_root)
    dataset_name = f"Dataset{dataset_id:03d}_MMWHS"

    out_dir = Path(nnUNet_raw) / dataset_name
    out_images_tr = out_dir / "imagesTr"
    out_labels_tr = out_dir / "labelsTr"

    # Discover patients
    image_files = sorted(dataset_root.glob("mr_train_*_image.nii.gz"))
    if not image_files:
        raise FileNotFoundError(
            f"No mr_train_*_image.nii.gz files found in {dataset_root}. "
            f"Make sure --dataset_root points to the mr_train folder."
        )

    total_found = len(image_files)
    if smoke_test is not None:
        image_files = image_files[:smoke_test]
        print(f"SMOKE TEST: using {len(image_files)} of {total_found} patients")
    else:
        print(f"Found {total_found} patients")

    # Clean previous conversion if exists
    if out_dir.exists():
        print(f"Removing existing dataset at {out_dir}")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True)
    out_images_tr.mkdir()
    out_labels_tr.mkdir()

    num_training_cases = 0

    for img_file in image_files:
        # Extract patient ID: mr_train_1001_image.nii.gz → 1001
        patient_id = img_file.name.split("_")[2]
        case_id = f"mmwhs_{patient_id}"

        label_file = dataset_root / f"mr_train_{patient_id}_label.nii.gz"
        if not label_file.exists():
            print(f"  WARNING: No label for patient {patient_id}, skipping")
            continue

        # Copy image with _0000 channel suffix
        shutil.copy(img_file, out_images_tr / f"{case_id}_0000.nii.gz")

        # Remap and save label
        print(f"  Processing {case_id}...")
        remap_labels(label_file, out_labels_tr / f"{case_id}.nii.gz")

        num_training_cases += 1

    # Generate dataset.json
    generate_dataset_json(
        str(out_dir),
        channel_names={0: "bSSFP_MRI"},
        labels=LABEL_NAMES,
        num_training_cases=num_training_cases,
        file_ending=".nii.gz",
        dataset_name="MMWHS",
        description=(
            f"MM-WHS 2017 whole heart MRI segmentation - "
            f"{num_training_cases} patients, 7 structures"
        ),
        license="See MM-WHS challenge terms (registration required)",
        reference="Zhuang, IEEE TPAMI 2019; Zhuang & Shen, MedIA 2016",
        converted_by="twin_core.nnunet_pipeline.convert_mmwhs_dataset",
    )

    print(f"\nConversion complete:")
    print(f"  Patients: {num_training_cases}")
    print(f"  Output: {out_dir}")
    print(f"  Structures: {list(LABEL_NAMES.keys())}")
    print(f"  dataset.json written to {out_dir / 'dataset.json'}")

    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert MM-WHS MRI dataset to nnU-Net v2 format"
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True,
        help="Path to the mr_train folder"
    )
    parser.add_argument(
        "--dataset_id", type=int, default=28,
        help="nnU-Net dataset ID (default: 28)"
    )
    parser.add_argument(
        "--smoke_test", type=int, default=None,
        help="Only convert this many patients (for quick testing)"
    )
    args = parser.parse_args()

    convert_mmwhs(
        dataset_root=args.dataset_root,
        dataset_id=args.dataset_id,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
