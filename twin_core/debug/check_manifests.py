# check_manifests.py
import json, sys, pathlib
def load(p): return json.load(open(p,'r',encoding='utf-8'))
preproc = pathlib.Path("C:/Users/dimok/Downloads/PhD/Digital Twin/Data/Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)/Resources/database/merged_dataset_150/preprocessed")  # change this
print("split_manifest exists:", (preproc/"split_manifest.json").exists())
print("mask_manifest exists:", (preproc/"mask_manifest.json").exists())
sm = load(preproc/"split_manifest.json")
mm = load(preproc/"mask_manifest.json")
print("train len:", len(sm["train"]), "val len:", len(sm["val"]))
tot_saved = sum(int(mm[p]["saved_masks"]) for p in mm)
tot_real = sum(int(mm[p]["real_masks"]) for p in mm)
print("manifest totals saved_masks:", tot_saved, "real_masks:", tot_real)
# print a few patient names
print("sample train:", sm["train"][:10])
