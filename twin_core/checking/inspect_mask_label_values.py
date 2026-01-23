import numpy as np, glob
p = glob.glob(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\preprocessed\train\*\masks\*.npy", recursive=True)
print("example mask:", p[:3])
print("unique labels (example):", np.unique(np.load(p[0])))
