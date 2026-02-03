# inspect_patient124.py
import nibabel as nib
import numpy as np
from pathlib import Path
from pprint import pprint
import sys
proj = Path(r"C:\Users\dimok\VSCodeProjects\digital_twin_project")   # update if needed
pdir = Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\merged_dataset_150\training\patient124")  # update if needed

mask_files = sorted(list(pdir.glob("*_frame*_gt.nii*")))
print("Found mask files:", mask_files)
for p in mask_files:
    nii = nib.load(str(p))
    raw = np.asarray(nii.get_fdata())
    raw_nnz = int(np.count_nonzero(raw))
    raw_shape = raw.shape
    # canonicalized
    try:
        nii_c = nib.as_closest_canonical(nii)
        can = np.asarray(nii_c.get_fdata())
        can_nnz = int(np.count_nonzero(can))
        can_shape = can.shape
    except Exception as e:
        can_nnz = None
        can_shape = None
    print("----")
    print("file:", p.name)
    print(" raw shape:", raw_shape, " raw nnz:", raw_nnz)
    print(" can shape:", can_shape, " can nnz:", can_nnz)
    # apply same mask_to_zyx as pipeline:
    def mask_to_zyx_local(a):
        a = np.asarray(a)
        nd = a.ndim
        if nd == 2:
            return a[None,:,:].astype(np.uint8)
        if nd == 3:
            z,y,x = a.shape
            if z <= 20 and x > 20:
                return a.astype(np.uint8)
            if x <= 20 and z > 20:
                return np.transpose(a, (2,1,0)).astype(np.uint8)
            if x > 20:
                return a.astype(np.uint8)
            return np.transpose(a,(2,1,0)).astype(np.uint8)
        if nd == 4:
            if a.shape[0] == 1:
                b = a[0]; 
                if b.ndim == 3:
                    return b.astype(np.uint8)
            try:
                b = a[...,0]; return np.transpose(b,(2,1,0)).astype(np.uint8)
            except:
                pass
        a_s = np.squeeze(a)
        if a_s.ndim == 3: return mask_to_zyx_local(a_s)
        if a_s.ndim == 2: return a_s[None,:,:].astype(np.uint8)
        return np.zeros((1,1,1),dtype=np.uint8)
    try:
        arr3_raw = mask_to_zyx_local(raw)
        arr3_can = mask_to_zyx_local(can) if can_shape is not None else None
        print(" mask_to_zyx raw ->", arr3_raw.shape, " nnz:", int(np.count_nonzero(arr3_raw)))
        if arr3_can is not None:
            print(" mask_to_zyx can ->", arr3_can.shape, " nnz:", int(np.count_nonzero(arr3_can)))
    except Exception as e:
        print("mask_to_zyx failed:", e)

# Also inspect the canonicalized image slices sizes used by pipeline
img_file = next(pdir.glob("patient124_*_4d.nii*"), None)
if img_file:
    nii_img = nib.load(str(img_file))
    niic = nib.as_closest_canonical(nii_img)
    print("Image canonical shape (nib):", niic.shape)
    # show example slice shape for t=0,z=0
    data = np.asarray(niic.get_fdata())
    if data.ndim==4:
        X,Y,Z,T = niic.shape
        print("canonical nib shape (X,Y,Z,T):", niic.shape)
        slice_xy = np.asarray(niic.dataobj[:,:,0,0])  # (X,Y) then transpose
        print("slice (after transpose) shape expected by pipeline:", slice_xy.T.shape)
