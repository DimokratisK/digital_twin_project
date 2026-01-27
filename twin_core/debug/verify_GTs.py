import nibabel as nib
import numpy as np
from pathlib import Path
from twin_core.utils.preprocessing_pipeline import reorder_to_tzyx, _parse_frame_index_from_name

p = Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\training\patient001")
img_nii = nib.load(str(p / "patient001_4d.nii.gz"))
img_r = reorder_to_tzyx(img_nii.get_fdata(), img_nii.header)
T, Z, Y, X = img_r.shape

lbl_stack = np.zeros((T, Z, Y, X), dtype=np.uint8)

for f in sorted(p.glob("*_frame*_gt.nii*")):
    t = _parse_frame_index_from_name(f.name)
    nii = nib.load(str(f))
    gt = reorder_to_tzyx(nii.get_fdata(), nii.header)
    # normalize to (Z,Y,X)
    if gt.ndim == 4 and gt.shape[0] == 1:
        gt3 = gt[0]
    elif gt.ndim == 3:
        gt3 = gt
    else:
        gt3 = np.squeeze(gt)
        if gt3.ndim == 2:
            gt3 = gt3[None,:,:]
    z_copy = min(gt3.shape[0], Z)
    y_copy = min(gt3.shape[1], Y)
    x_copy = min(gt3.shape[2], X)
    lbl_stack[t, :z_copy, :y_copy, :x_copy] = gt3[:z_copy, :y_copy, :x_copy]

for t in range(T):
    print(f"t={t:02d} nonzero voxels: {int(np.count_nonzero(lbl_stack[t]))}")
