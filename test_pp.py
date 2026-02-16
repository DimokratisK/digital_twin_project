import os
os.environ["nnUNet_raw"] = "C:/Users/dimok/VSCodeProjects/digital_twin_project/nnunet_data/raw"
os.environ["nnUNet_preprocessed"] = "C:/Users/dimok/VSCodeProjects/digital_twin_project/nnunet_data/preprocessed"
os.environ["nnUNet_results"] = "C:/Users/dimok/VSCodeProjects/digital_twin_project/nnunet_data/results"
print("Step 1: env set")
import torch
print("Step 2: torch OK", torch.__version__)
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask
print("Step 3: acvl_utils OK")
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints
print("Step 4: nnunet imports OK")
