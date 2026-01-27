import numpy as np
import glob

import numpy as np
import glob

files = glob.glob(
    r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\preprocessed\*\*\data\*.npy"
)

shapes = {np.load(f).shape for f in files}
print(shapes)
