"""
Convert the Bjonze PV+LAA labelmaps (over ImageCAS CCTA images) to nnU-Net v2 format.

The Bjonze dataset (Hansen et al., STACOM 2025) provides 10-class labelmaps computed
on the public ImageCAS CCTA cohort. For our cardiac digital twin we only need the two
classes ImageCAS/TSHC licensing lets us use AND that are relevant for LA+PV modelling:

    Bjonze class → our class
        0  Background       → 0
        1  Myocardium       → 0   (TSHC license — cannot redistribute these as labels)
        2  LA               → 0
        3  LV               → 0
        4  RA               → 0
        5  RV               → 0
        6  Aorta            → 0
        7  PA               → 0
        8  LAA              → 2   (kept)
        9  Coronary         → 0
        10 PV               → 1   (kept — lumped; per-vein split done downstream)

Filtering:
    By default only the 685 cases listed in all_full_laa_segmentations_id.txt are
    converted (cases where the LAA does not touch the scan FOV edge). Use
    --use_all_cases to include all 998 cases instead.

Source structure:
    <bjonze_root>/
        info/
            all_full_laa_segmentations_id.txt    # 685 IDs, one per line, format "100.img"
            all_segmentations_id.txt             # 998 IDs
            readme.txt
        segmentations/
            100.img.nii.gz                       # labelmap with values 0..10
            ...

    <imagecas_root>/
        1-200/100.img.nii.gz                     # matching CCTA image
        201-400/.../...
        401-600/...
        601-800/...
        801-1000/...

Target structure (nnU-Net v2):
    $nnUNet_raw/Dataset031_Bjonze_PV_LAA/
        imagesTr/bjonze_100_0000.nii.gz
        labelsTr/bjonze_100.nii.gz
        dataset.json

Usage:
    python -m twin_core.nnunet_pipeline.convert_bjonze_dataset \
        --bjonze_root   ~/datasets/bjonze/ImageCAS-STACOM2025-02-10-2025 \
        --imagecas_root ~/datasets/imagecas/raw

    # Include all 998 cases (some with truncated LAAs):
    python -m twin_core.nnunet_pipeline.convert_bjonze_dataset ... --use_all_cases

    # Smoke test (5 patients):
    python -m twin_core.nnunet_pipeline.convert_bjonze_dataset ... --smoke_test 5
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


LABEL_MAPPING = {
    0:  0,  # background
    1:  0,  # myocardium      — TSHC license, drop
    2:  0,  # LA              — TSHC license, drop
    3:  0,  # LV              — TSHC license, drop
    4:  0,  # RA              — TSHC license, drop
    5:  0,  # RV              — TSHC license, drop
    6:  0,  # aorta           — TSHC license, drop
    7:  0,  # PA              — TSHC license, drop
    8:  2,  # LAA             — keep
    9:  0,  # coronary        — drop (not needed for LA CFD)
    10: 1,  # PV              — keep (lumped; per-vein split done later)
}

LABEL_NAMES = {
    "background": 0,
    "PV": 1,
    "LAA": 2,
}

IMAGECAS_BATCHES = ["1-200", "201-400", "401-600", "601-800", "801-1000"]


def parse_id_list(txt_path: Path) -> list[str]:
    """Read an ID list file. Lines look like '100.img' — return ['100', ...]."""
    ids = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith(".img"):
            line = line[:-4]
        ids.append(line)
    return ids


def locate_imagecas_image(imagecas_root: Path, patient_id: str) -> Optional[Path]:
    """Find the matching CCTA image across the 5 batch folders."""
    for batch in IMAGECAS_BATCHES:
        candidate = imagecas_root / batch / f"{patient_id}.img.nii.gz"
        if candidate.is_file():
            return candidate
    return None


def remap_labels(label_path: Path, output_path: Path) -> set:
    """Load a Bjonze labelmap, remap to our 3-class scheme, save. Returns any unknown values."""
    img = nib.load(str(label_path))
    data = np.asarray(img.dataobj).astype(np.int16)

    remapped = np.zeros_like(data, dtype=np.uint8)
    for orig_val, new_val in LABEL_MAPPING.items():
        if new_val != 0:
            remapped[data == orig_val] = new_val

    unknown = set(np.unique(data).tolist()) - set(LABEL_MAPPING.keys())

    out_img = nib.Nifti1Image(remapped, img.affine, img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(output_path))
    return unknown


def convert_bjonze(
    bjonze_root: str,
    imagecas_root: str,
    dataset_id: int = 31,
    use_all_cases: bool = False,
    smoke_test: Optional[int] = None,
):
    bjonze_root = Path(bjonze_root).expanduser()
    imagecas_root = Path(imagecas_root).expanduser()

    seg_dir = bjonze_root / "segmentations"
    info_dir = bjonze_root / "info"
    if not seg_dir.is_dir():
        raise FileNotFoundError(f"Expected segmentations/ under {bjonze_root}")
    if not info_dir.is_dir():
        raise FileNotFoundError(f"Expected info/ under {bjonze_root}")

    id_file_name = (
        "all_segmentations_id.txt" if use_all_cases
        else "all_full_laa_segmentations_id.txt"
    )
    id_file = info_dir / id_file_name
    if not id_file.is_file():
        raise FileNotFoundError(f"ID list not found: {id_file}")

    patient_ids = parse_id_list(id_file)
    expected_count = 998 if use_all_cases else 685
    if len(patient_ids) != expected_count:
        print(f"  WARNING: {id_file.name} has {len(patient_ids)} IDs (expected {expected_count})")

    dataset_name = f"Dataset{dataset_id:03d}_Bjonze_PV_LAA"
    out_dir = Path(nnUNet_raw) / dataset_name
    out_images_tr = out_dir / "imagesTr"
    out_labels_tr = out_dir / "labelsTr"

    if out_dir.exists():
        print(f"Removing existing dataset at {out_dir}")
        shutil.rmtree(out_dir)
    out_images_tr.mkdir(parents=True)
    out_labels_tr.mkdir()

    if smoke_test is not None:
        patient_ids = patient_ids[:smoke_test]
        print(f"SMOKE TEST: converting {len(patient_ids)} patients")
    else:
        print(f"Converting {len(patient_ids)} patients from {id_file.name}")

    converted = 0
    missing_image = []
    missing_label = []
    unknown_label_values: dict[str, set] = {}

    for pid in patient_ids:
        label_src = seg_dir / f"{pid}.img.nii.gz"
        if not label_src.is_file():
            missing_label.append(pid)
            continue

        image_src = locate_imagecas_image(imagecas_root, pid)
        if image_src is None:
            missing_image.append(pid)
            continue

        case_id = f"bjonze_{pid}"
        shutil.copy(image_src, out_images_tr / f"{case_id}_0000.nii.gz")
        unknown = remap_labels(label_src, out_labels_tr / f"{case_id}.nii.gz")
        if unknown:
            unknown_label_values[pid] = unknown

        converted += 1
        if converted % 50 == 0:
            print(f"  {converted}/{len(patient_ids)}")

    generate_dataset_json(
        str(out_dir),
        channel_names={0: "CT"},
        labels=LABEL_NAMES,
        num_training_cases=converted,
        file_ending=".nii.gz",
        dataset_name="Bjonze_PV_LAA",
        description=(
            f"Pulmonary veins (lumped) + LAA on ImageCAS CCTA. Labels from Bjonze et al. "
            f"(STACOM 2025); images from ImageCAS (Xu et al.). {converted} cases "
            f"({'all' if use_all_cases else 'full-LAA only'})."
        ),
        license="Bjonze labels: see STACOM 2025 publication. Images: ImageCAS (Kaggle, CC BY-NC-SA).",
        reference="Hansen et al., STACOM 2025; Xu et al., ImageCAS",
        converted_by="twin_core.nnunet_pipeline.convert_bjonze_dataset",
    )

    print(f"\nConversion complete:")
    print(f"  Output:    {out_dir}")
    print(f"  Converted: {converted}")
    print(f"  Missing labels: {len(missing_label)}")
    print(f"  Missing images: {len(missing_image)}")
    if missing_label:
        print(f"    (first 5 label IDs missing: {missing_label[:5]})")
    if missing_image:
        print(f"    (first 5 image IDs missing: {missing_image[:5]})")
    if unknown_label_values:
        print(f"  WARNING: {len(unknown_label_values)} cases had unexpected label values")
        for pid, vals in list(unknown_label_values.items())[:5]:
            print(f"    {pid}: {sorted(vals)}")

    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert Bjonze PV+LAA labels (over ImageCAS images) to nnU-Net v2 format"
    )
    parser.add_argument("--bjonze_root", type=str, required=True,
                        help="Path to ImageCAS-STACOM2025-02-10-2025/ (contains info/ + segmentations/)")
    parser.add_argument("--imagecas_root", type=str, required=True,
                        help="Path to ImageCAS raw/ (contains batch dirs 1-200, 201-400, ...)")
    parser.add_argument("--dataset_id", type=int, default=31,
                        help="nnU-Net dataset ID (default: 31)")
    parser.add_argument("--use_all_cases", action="store_true",
                        help="Include all 998 cases (default: 685 full-LAA cases only)")
    parser.add_argument("--smoke_test", type=int, default=None,
                        help="Only convert this many patients")
    args = parser.parse_args()

    convert_bjonze(
        bjonze_root=args.bjonze_root,
        imagecas_root=args.imagecas_root,
        dataset_id=args.dataset_id,
        use_all_cases=args.use_all_cases,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
