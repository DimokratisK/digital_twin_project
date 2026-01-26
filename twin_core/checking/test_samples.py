from twin_core.data_ingestion.dataset import CardiacDataset
from pathlib import Path
import numpy as np

# ds = CardiacDataset(Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\preprocessed\train"), exclude_missing_masks=True)
# print("train samples:", len(ds))
# print("example masks uniques:", np.unique(np.load(ds.samples[0][4])))


val_root = Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\preprocessed\val")
ds_val = CardiacDataset(val_root, exclude_missing_masks=True)
print("val samples:", len(ds_val))