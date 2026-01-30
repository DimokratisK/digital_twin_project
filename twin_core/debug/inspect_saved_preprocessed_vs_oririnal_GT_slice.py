import numpy as np, nibabel as nib, pathlib
pre_mask = pathlib.Path(r"c:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\preprocessed\val\patient001\masks\t00_z02.npy")
orig_gt = pathlib.Path(r"c:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\training\patient001\patient001_frame01_gt.nii.gz")
mask = np.load(pre_mask)
gt = nib.load(str(orig_gt)).get_fdata()
# select the same z slice after reorder if needed; adapt indices to the pipeline
print("pre mask nonzero", int(np.count_nonzero(mask)), "unique", np.unique(mask)[:10])
# inspect corresponding slice in gt (example: gt[2] if z=2)
print("orig gt nonzero per z", [int(np.count_nonzero(gt[z])) for z in range(gt.shape[0])])
