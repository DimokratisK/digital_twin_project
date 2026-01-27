import torch
from typing import Union, Dict
from pathlib import Path
from .UNET_model import UNet


def _strip_prefixes(state_dict: Dict[str, torch.Tensor], prefixes=("module.", "model.", "net.")) -> Dict[str, torch.Tensor]:
    """
    Remove common prefixes from state_dict keys if present.
    """
    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
                break
        new_state[new_k] = v
    return new_state


def _extract_state_dict(ck: object) -> Dict[str, torch.Tensor]:
    """
    Given a loaded checkpoint object, try to extract a state_dict-like mapping.
    Handles common training checkpoint layouts (model_state, state_dict, etc.).
    """
    if isinstance(ck, dict):
        # common candidates where the model weights are stored
        for candidate in ("model_state", "state_dict", "model", "model_state_dict"):
            if candidate in ck and isinstance(ck[candidate], dict):
                return ck[candidate]
        # If the dict itself looks like a state_dict (keys -> tensors), return it
        # Heuristic: check a few keys for tensor-like values
        sample_keys = list(ck.keys())[:5]
        if sample_keys:
            sample_val = ck[sample_keys[0]]
            if hasattr(sample_val, "shape") or isinstance(sample_val, torch.Tensor):
                return ck  # assume it's already a state_dict
    raise RuntimeError("Checkpoint does not contain a recognizable state_dict mapping")


def _robust_load_state_dict_into(model: torch.nn.Module, ckpt_path: Union[str, Path], device: str = "cpu") -> torch.nn.Module:
    """
    Robustly load a checkpoint into model:
      - extract nested state dict if present
      - strip common prefixes (module., model., net.)
      - try strict load, fallback to non-strict and print diagnostics
      - move model to device and set eval()
    """
    ck = torch.load(str(ckpt_path), map_location="cpu")

    # Extract nested state dict
    state = _extract_state_dict(ck)

    # Strip common prefixes
    stripped = _strip_prefixes(state, prefixes=("module.", "model.", "net."))

    # Attempt strict load first, then fallback to non-strict with diagnostics
    try:
        model.load_state_dict(stripped, strict=True)
        print("[load_model] Loaded checkpoint (strict=True)")
    except RuntimeError as e:
        print("[load_model] strict load failed:", e)
        missing, unexpected = model.load_state_dict(stripped, strict=False)
        print("[load_model] missing keys:", missing)
        print("[load_model] unexpected keys:", unexpected)
        # continue with partially loaded model

    model.to(device)
    model.eval()
    return model


def load_model(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    n_classes: int = 4,
) -> torch.nn.Module:
    """
    Load a UNet model for multi-class segmentation.

    - checkpoint_path: path to a saved state_dict or a training checkpoint.
    - device: "cpu" or "cuda".
    - n_classes: number of output classes (including background).

    This implementation is robust to common checkpoint formats:
    - raw state_dict saved with torch.save(model.state_dict())
    - training checkpoint dict containing "model_state" or "state_dict"
    - state_dicts with prefixes like "module.", "model.", "net."
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Instantiate model with correct number of output channels
    model = UNet(out_channels=int(n_classes))

    # Robustly load weights into model
    model = _robust_load_state_dict_into(model, checkpoint_path, device=device)

    return model
