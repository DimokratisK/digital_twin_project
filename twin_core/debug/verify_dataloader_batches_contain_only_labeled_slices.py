from twin_core.data_ingestion.dataset import CardiacDataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch

train_root = Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\preprocessed\train")
ds_train = CardiacDataset(train_root, exclude_missing_masks=True)

loader = DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=0)

images, masks = next(iter(loader))
print("image shape:", images.shape)
print("mask uniques per sample:", [torch.unique(m).tolist() for m in masks])
