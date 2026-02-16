"""
Convert ACDC cardiac MRI dataset to nnU-Net v2 format.

Supports two dataset layouts:
  --mode merged : All patients in a single folder (user's merged 150-patient pool)
  --mode split  : Original ACDC layout with 'training/' and 'testing/' subfolders

Usage:
    # Full conversion (all 150 patients):
    python -m twin_core.nnunet_pipeline.convert_acdc_dataset \
        --dataset_root "C:\...\merged_dataset_150" \
        --dataset_id 27

    # Smoke test (only 5 patients):
    python -m twin_core.nnunet_pipeline.convert_acdc_dataset \
        --dataset_root "C:\...\merged_dataset_150" \
        --dataset_id 27 \
        --smoke_test 5
"""

import argparse
import os
import shutil
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Set up environment before importing nnU-Net
from twin_core.nnunet_pipeline.set_environment import set_env_vars, create_directories
set_env_vars()
create_directories()

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


def parse_info_cfg(info_path: Path) -> Dict:
    """Parse a patient's info.cfg file into a dictionary."""
    metadata = {}
    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            # Try to parse as number
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
            metadata[key] = value
    return metadata


def discover_patients(dataset_root: Path, mode: str) -> List[Path]:
    """Find all patient directories based on the dataset mode."""
    if mode == "merged":
        # All patients directly under dataset_root (or under a 'training' subfolder)
        # Check if there's a 'training' subfolder first
        training_dir = dataset_root / "training"
        if training_dir.is_dir():
            search_dir = training_dir
        else:
            search_dir = dataset_root
        patients = sorted([
            d for d in search_dir.iterdir()
            if d.is_dir() and d.name.startswith("patient")
        ])
    elif mode == "split":
        # Original ACDC layout: training/ and testing/ subfolders
        patients = []
        for subfolder in ["training", "testing"]:
            sub_dir = dataset_root / subfolder
            if sub_dir.is_dir():
                patients.extend(sorted([
                    d for d in sub_dir.iterdir()
                    if d.is_dir() and d.name.startswith("patient")
                ]))
        patients = sorted(patients, key=lambda p: p.name)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'merged' or 'split'.")
    return patients


def convert_acdc(
    dataset_root: str,
    dataset_id: int = 27,
    mode: str = "merged",
    smoke_test: Optional[int] = None,
):
    """
    Convert ACDC dataset to nnU-Net v2 format.

    Args:
        dataset_root: Path to the ACDC dataset root.
        dataset_id: nnU-Net dataset ID (default: 27).
        mode: 'merged' (all patients in one folder) or 'split' (training/testing subfolders).
        smoke_test: If set, only convert this many patients.
    """
    dataset_root = Path(dataset_root)
    dataset_name = f"Dataset{dataset_id:03d}_ACDC"

    out_dir = Path(nnUNet_raw) / dataset_name
    out_images_tr = out_dir / "imagesTr"
    out_labels_tr = out_dir / "labelsTr"

    # Clean previous conversion if exists
    if out_dir.exists():
        print(f"Removing existing dataset at {out_dir}")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True)
    out_images_tr.mkdir()
    out_labels_tr.mkdir()

    # Discover patients
    patients = discover_patients(dataset_root, mode)
    total_found = len(patients)

    if smoke_test is not None:
        patients = patients[:smoke_test]
        print(f"SMOKE TEST: using {len(patients)} of {total_found} patients")
    else:
        print(f"Found {total_found} patients")

    # Process each patient
    patient_metadata = {}
    num_training_cases = 0
    skipped_files = []

    for patient_dir in patients:
        patient_id = patient_dir.name  # e.g. "patient001"

        # Parse info.cfg
        info_path = patient_dir / "info.cfg"
        if info_path.exists():
            metadata = parse_info_cfg(info_path)
            patient_metadata[patient_id] = metadata
        else:
            print(f"  WARNING: No info.cfg found for {patient_id}")
            patient_metadata[patient_id] = {}

        # Find frame NIfTI files (exclude _4d and _gt files)
        frame_files = sorted([
            f for f in patient_dir.glob("*.nii.gz")
            if "_4d" not in f.name and "_gt" not in f.name
        ])

        for frame_file in frame_files:
            # e.g. patient001_frame01.nii.gz
            case_id = frame_file.name.replace(".nii.gz", "")  # patient001_frame01

            # Corresponding ground truth
            gt_file = patient_dir / f"{case_id}_gt.nii.gz"
            if not gt_file.exists():
                skipped_files.append(str(frame_file))
                continue

            # Copy image with _0000 channel suffix
            shutil.copy(frame_file, out_images_tr / f"{case_id}_0000.nii.gz")

            # Copy label (no _0000 suffix, no _gt)
            shutil.copy(gt_file, out_labels_tr / f"{case_id}.nii.gz")

            num_training_cases += 1

    # Compute group distribution
    group_counts = Counter(
        meta.get("Group", "unknown")
        for meta in patient_metadata.values()
    )

    print(f"\nConversion complete:")
    print(f"  Patients: {len(patients)}")
    print(f"  Training cases (frames): {num_training_cases}")
    print(f"  Groups: {dict(group_counts)}")
    if skipped_files:
        print(f"  Skipped (no GT): {len(skipped_files)} files")

    # Generate dataset.json with extended metadata
    generate_dataset_json(
        str(out_dir),
        channel_names={0: "cineMRI"},
        labels={
            "background": 0,
            "RV": 1,
            "MYO": 2,
            "LV": 3,
        },
        num_training_cases=num_training_cases,
        file_ending=".nii.gz",
        dataset_name="ACDC",
        description=f"ACDC cardiac segmentation - {len(patients)} patients, ED+ES frames",
        license="See ACDC challenge terms",
        converted_by="twin_core.nnunet_pipeline.convert_acdc_dataset",
        # Extra fields stored via **kwargs
        patient_metadata=patient_metadata,
        group_distribution=dict(group_counts),
    )

    print(f"\n  dataset.json written to {out_dir / 'dataset.json'}")

    # Create patient-level cross-validation splits (stratified by Group)
    _create_patient_splits(out_dir, out_labels_tr, patient_metadata, dataset_name)

    return out_dir


def _create_patient_splits(
    out_dir: Path,
    labels_dir: Path,
    patient_metadata: Dict,
    dataset_name: str,
    n_folds: int = 5,
    seed: int = 1234,
):
    """
    Create nnU-Net cross-validation splits that respect patient boundaries.

    Ensures ED/ES frames from the same patient are always in the same fold.
    Stratifies by patient Group when possible.
    """
    from batchgenerators.utilities.file_and_folder_operations import (
        nifti_files, maybe_mkdir_p, save_json
    )

    nii_files = nifti_files(str(labels_dir), join=False)
    # Extract unique patient IDs
    patients = sorted(set(f.split("_frame")[0] for f in nii_files))

    # Group patients for stratified splitting
    groups = {}
    for patient_id in patients:
        group = patient_metadata.get(patient_id, {}).get("Group", "unknown")
        groups.setdefault(group, []).append(patient_id)

    # Shuffle within each group, then distribute across folds
    rs = np.random.RandomState(seed)
    for group_patients in groups.values():
        rs.shuffle(group_patients)

    # Round-robin assignment of patients to folds, stratified by group
    fold_patients = [[] for _ in range(n_folds)]
    for group_name, group_patients in sorted(groups.items()):
        for i, patient_id in enumerate(group_patients):
            fold_patients[i % n_folds].append(patient_id)

    # Build splits: each fold is validation, rest is training
    splits = []
    for fold in range(n_folds):
        val_patients = set(fold_patients[fold])
        train_patients = set(patients) - val_patients

        val_cases = [f[:-7] for f in nii_files if f.split("_frame")[0] in val_patients]
        train_cases = [f[:-7] for f in nii_files if f.split("_frame")[0] in train_patients]

        splits.append({"train": sorted(train_cases), "val": sorted(val_cases)})

        print(f"  Fold {fold}: {len(train_cases)} train, {len(val_cases)} val cases "
              f"({len(train_patients)} / {len(val_patients)} patients)")

    # Save to nnUNet_preprocessed
    preprocessed_dir = Path(nnUNet_preprocessed) / dataset_name
    maybe_mkdir_p(str(preprocessed_dir))
    splits_path = preprocessed_dir / "splits_final.json"
    save_json(splits, str(splits_path), sort_keys=False)
    print(f"\n  splits_final.json written to {splits_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ACDC dataset to nnU-Net v2 format"
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True,
        help="Path to the ACDC dataset root folder"
    )
    parser.add_argument(
        "--dataset_id", type=int, default=27,
        help="nnU-Net dataset ID (default: 27)"
    )
    parser.add_argument(
        "--mode", type=str, default="merged", choices=["merged", "split"],
        help="'merged' = all patients in one folder, 'split' = original training/testing subfolders"
    )
    parser.add_argument(
        "--smoke_test", type=int, default=None,
        help="Only convert this many patients (for quick testing)"
    )
    args = parser.parse_args()

    convert_acdc(
        dataset_root=args.dataset_root,
        dataset_id=args.dataset_id,
        mode=args.mode,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
