import nibabel as nib
import pathlib
from twin_core.utils.preprocessing_pipeline import reorder_to_tzyx  # adjust import path

p = pathlib.Path(r"c:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\training\patient001")
img_nii = nib.load(str(p / "patient001_4d.nii.gz"))
img_r = reorder_to_tzyx(img_nii.get_fdata(), img_nii.header)
print("image after reorder:", img_r.shape)   # expect (T, Z, Y, X)

for f in sorted(p.glob("*_frame*_gt.nii*")):
    nii = nib.load(str(f))
    gt_r = reorder_to_tzyx(nii.get_fdata(), nii.header)
    print(f.name, "after reorder:", gt_r.shape)
