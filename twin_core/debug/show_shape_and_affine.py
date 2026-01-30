import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Define the directory where the NIfTI file is located
# Replace 'path/to/the/data/folder' with the actual path
data_dir = Path("c:/Users/dimok/Downloads/PhD/Udemy_courses/Deep_Learning_for_medical_imaging/AI-IN-MEDICAL-MATERIALS_NEW/AI-IN-MEDICAL-MATERIALS/03-Data-Formats/03-Preprocessing/IXI662-Guys-1120-T1.nii.gz") 
file_path = data_dir


brain_mri = nib.load(file_path)
brain_mri_data = brain_mri.get_fdata()


shape = brain_mri.shape
affine = brain_mri.affine
print(affine)
print(shape)

print(brain_mri.header.get_zooms())

