import torch
from twin_core.utils.UNET_model import UNet
from twin_core.utils.segmentation_model import _robust_load_state_dict_into

ckpt = r"checkpoints/ckpt_best.pt"

# instantiate with the correct args
model = UNet(in_channels=1, out_channels=4, base_features=64)

# load weights safely on CPU and print diagnostics
model = _robust_load_state_dict_into(model, ckpt, device="cpu")

# move to GPU if available for inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print("Model ready on", device)

