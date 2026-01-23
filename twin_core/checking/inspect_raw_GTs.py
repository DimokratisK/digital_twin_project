import nibabel as nib, numpy as np, pathlib
p = pathlib.Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\training\patient001")
for f in sorted(p.glob("*_frame*_gt.nii*")):
    arr = nib.load(str(f)).get_fdata()
    print(f.name, "shape", arr.shape, "nonzero", int(np.count_nonzero(arr)), "unique", np.unique(arr)[:10])
    # per-Z nonzero counts
    if arr.ndim == 3:
        print(" per-Z nonzero:", [int(np.count_nonzero(arr[z])) for z in range(arr.shape[0])])
