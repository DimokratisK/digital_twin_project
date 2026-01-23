import nibabel as nib, numpy as np, pathlib
p = pathlib.Path(r"c:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\training\patient001")
for f in sorted(p.glob("*_frame*_gt.nii*")):
    arr = nib.load(str(f)).get_fdata()
    nz = int(np.count_nonzero(arr))
    print(f.name, "shape", arr.shape, "nonzero voxels", nz, "unique", np.unique(arr)[:10])
