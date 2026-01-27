import torch
ck = torch.load(r"checkpoints/ckpt_best.pt", map_location="cpu")
ms = ck.get("model_state", ck.get("state_dict", None))
print("model_state present:", ms is not None)
if isinstance(ms, dict):
    print("sample keys:", list(ms.keys())[:40])
else:
    print("top-level keys:", list(ck.keys()))

