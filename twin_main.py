import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
from twin_core.utils.fix_path import fix_path