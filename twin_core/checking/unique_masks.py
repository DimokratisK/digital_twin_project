import glob, os
base = r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database"

# 1) look for common label folders
for name in ("masks","labels","gt","seg","ground_truth"):
    hits = glob.glob(os.path.join(base, "**", name, "*.*"), recursive=True)
    print(name, "found:", len(hits))
    if hits:
        print(" example:", hits[:3])

# 2) look for any .nii/.nii.gz/.png label-like files
candidates = glob.glob(os.path.join(base, "**", "*.nii*"), recursive=True) + \
             glob.glob(os.path.join(base, "**", "*.png"), recursive=True) + \
             glob.glob(os.path.join(base, "**", "*.nii.gz"), recursive=True)
print("other candidate label files:", len(candidates))
print("examples:", candidates[:5])
