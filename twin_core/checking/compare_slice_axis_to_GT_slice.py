import nibabel as nib
import numpy as np
from pathlib import Path
from twin_core.utils.preprocessing_pipeline import reorder_to_tzyx

p = Path(r"c:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\training\patient001")
img_nii = nib.load(str(p / "patient001_4d.nii.gz"))
img_r = reorder_to_tzyx(img_nii.get_fdata(), img_nii.header)  # (T,Z,Y,X)
print("image shape (T,Z,Y,X):", img_r.shape)

for f in sorted(p.glob("*_frame*_gt.nii*")):
    gt_nii = nib.load(str(f))
    gt = gt_nii.get_fdata()
    print(f.name, "raw shape:", gt.shape, "raw nonzero:", int(np.count_nonzero(gt)))
    # show nonzero counts along each axis to identify slice axis
    print(" nonzero per axis0 (len=%d):" % gt.shape[0], [int(np.count_nonzero(gt[i,:,:])) for i in range(gt.shape[0])])
    print(" nonzero per axis1 (len=%d):" % gt.shape[1], [int(np.count_nonzero(gt[:,i,:])) for i in range(gt.shape[1])])
    print(" nonzero per axis2 (len=%d):" % gt.shape[2], [int(np.count_nonzero(gt[:,:,i])) for i in range(gt.shape[2])])
    # reorder and show shape and nonzero bbox
    gt_r = reorder_to_tzyx(gt, gt_nii.header)  # should be (T?, Z, Y, X) or (1,Z,Y,X)
    print(" after reorder shape:", gt_r.shape, "nonzero:", int(np.count_nonzero(gt_r)))
    nz = np.nonzero(gt_r)
    if len(nz[0])==0:
        print("  -> no nonzero after reorder")
    else:
        print("  -> bbox z", (int(nz[0].min()), int(nz[0].max())), "y", (int(nz[1].min()), int(nz[1].max())), "x", (int(nz[2].min()), int(nz[2].max())))
    print()
