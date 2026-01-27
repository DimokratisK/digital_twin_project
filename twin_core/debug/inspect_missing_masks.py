from pathlib import Path
import numpy as np

root = Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\preprocessed")  # adjust path

for split in ["train", "val"]:
    split_root = root / split
    patients = sorted([d for d in split_root.iterdir() if d.is_dir()])
    print(f"\n=== Checking {split} ({len(patients)} patients) ===")

    missing_masks = 0
    total = 0

    for patient in patients:
        data_files = sorted((patient / "data").glob("*.npy"))
        mask_files = {f.name for f in (patient / "masks").glob("*.npy")}
        for img in data_files:
            total += 1
            if img.name not in mask_files:
                missing_masks += 1

    print(f"Total slices: {total}")
    print(f"Missing masks: {missing_masks}")
