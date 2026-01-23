from twin_core.utils.preprocessing_pipeline import reorder_to_tzyx
import nibabel as nib, numpy as np, pathlib
p = pathlib.Path(r"c:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\training\patient001")
img = nib.load(str(p / "patient001_4d.nii.gz"))
img_r = reorder_to_tzyx(img.get_fdata(), img.header)
print("image shape", img_r.shape)
for f in sorted(p.glob("*_frame*_gt.nii*")):
    gt = nib.load(str(f))
    gt_r = reorder_to_tzyx(gt.get_fdata(), gt.header)
    nz = np.nonzero(gt_r)
    if len(nz[0])==0:
        print(f.name, "has no nonzero voxels after reorder")
    else:
        zmin, zmax = nz[0].min(), nz[0].max()
        ymin, ymax = nz[1].min(), nz[1].max()
        xmin, xmax = nz[2].min(), nz[2].max()
        print(f.name, "bbox z", (zmin,zmax), "y", (ymin,ymax), "x", (xmin,xmax))
