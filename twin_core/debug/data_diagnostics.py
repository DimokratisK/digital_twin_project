import nibabel as nib
from pathlib import Path
import numpy as np



root = Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\training")
img_root= Path(r"c:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\training\patient001\patient001_4d.nii.gz")

img = nib.load(img_root)
data = img.get_fdata()
print("Shape:", data.shape)
print("Affine:", img.affine)
print("Header:", img.header)





gt_path = Path(r"c:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\training\patient001\patient001_frame01_gt.nii.gz")
seg = nib.load(gt_path).get_fdata().astype(np.uint8)

print(seg.shape)
print(np.unique(seg))


# for p in sorted(root.iterdir()):
#     img = next(p.glob("*_4d.nii.gz"), None)
#     if img:
#         data = nib.load(img).get_fdata()
#         print(p.name, data.shape)
#         break
