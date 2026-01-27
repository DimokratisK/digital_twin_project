import numpy as np
from pathlib import Path

pre = Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\preprocessed\val\patient001\masks")
# iterate all saved masks and sum nonzero per time index
counts = {}
for p in sorted(pre.glob("t*_z*.npy")):
    stem = p.stem  # e.g., t00_z02
    t = int(stem.split("_")[0][1:])
    arr = np.load(p)
    counts.setdefault(t, 0)
    counts[t] += int(np.count_nonzero(arr))

for t in sorted(counts.keys()):
    print(f"t={t:02d} preprocessed nonzero voxels (sum over z): {counts[t]}")
