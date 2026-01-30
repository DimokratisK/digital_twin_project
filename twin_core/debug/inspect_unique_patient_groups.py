#!/usr/bin/env python3
"""
Print unique Group values found in training/*/info.cfg and the percentage of patients in each.
Run from the repository root (where the `training/` folder lives).
"""
from pathlib import Path
from collections import Counter

TRAIN_DIR = Path("C:/Users/dimok/Downloads/PhD/Digital Twin/Data/Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)/Resources/database/merged_dataset_150/training/")  

groups = []
for patient_dir in sorted(TRAIN_DIR.iterdir()):
    if not patient_dir.is_dir():
        continue
    info_file = patient_dir / "info.cfg"
    group_val = "UNKNOWN"
    if info_file.exists():
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    key, val = line.split(":", 1)
                    if key.strip().lower() == "group":
                        group_val = val.strip() or "UNKNOWN"
                        break
        except Exception:
            group_val = "UNKNOWN"
    groups.append(group_val)

total = len(groups)
counter = Counter(groups)

print(f"Total patients: {total}\n")
# Print sorted by descending count
for grp, cnt in counter.most_common():
    pct = (cnt / total * 100) if total > 0 else 0.0
    print(f"{grp}\t{cnt}\t{pct:.1f}%")
