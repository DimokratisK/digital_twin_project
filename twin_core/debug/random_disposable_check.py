import nibabel as nib
import numpy as np
img = nib.load(r'C:\Users\dimok\Downloads\PhD\Digital Twin\Data\MM-WHS Multi-Modality Whole Heart Segmentation\MM-WHS 2017 Dataset\mr_train\mr_train_1010_label.nii.gz')
data = np.asarray(img.dataobj)
total = data.size
n421 = np.sum(data == 421)
n420 = np.sum(data == 420)
print(f'Total voxels: {total:,}')
print(f'LA voxels (420): {n420:,}')
print(f'Stray voxels (421): {n421:,}')
print(f'Ratio 421/420: {n421/max(n420,1)*100:.2f}%')