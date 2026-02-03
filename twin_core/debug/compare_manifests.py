# compare_manifests.py
import json
from pathlib import Path

old_path = Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\merged_dataset_150\preprocessed\mask_manifest.json")   # update
new_path = Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\merged_dataset_150\preprocessed_new\mask_manifest.json")   # update

old = json.load(open(old_path, "r", encoding="utf-8"))
new = json.load(open(new_path, "r", encoding="utf-8"))

all_ids = sorted(set(list(old.keys()) + list(new.keys())))
differences = []
total_old_real = total_new_real = 0
total_old_saved = total_new_saved = 0

for pid in all_ids:
    o = old.get(pid, {})
    n = new.get(pid, {})
    or_real = int(o.get("real_masks", 0))
    nr_real = int(n.get("real_masks", 0))
    or_saved = int(o.get("saved_masks", 0))
    nr_saved = int(n.get("saved_masks", 0))
    total_old_real += or_real
    total_new_real += nr_real
    total_old_saved += or_saved
    total_new_saved += nr_saved
    if or_real != nr_real or or_saved != nr_saved:
        differences.append((pid, or_saved, nr_saved, or_real, nr_real, o.get("labeled_frame_indices"), n.get("labeled_frame_indices")))

print("Totals:")
print(" old saved_masks:", total_old_saved, " old real_masks:", total_old_real)
print(" new saved_masks:", total_new_saved, " new real_masks:", total_new_real)
print("\nChanged patients (saved_old, saved_new, real_old, real_new):")
for d in differences:
    pid, osv, nsv, orr, nrr, olf, nlf = d
    print(f"{pid}: saved {osv} -> {nsv}, real {orr} -> {nrr}")
    print("  old frames:", olf)
    print("  new frames:", nlf)
