import torch
from typing import Union
from pathlib import Path
from .UNET_model import UNet


def _strip_lightning_prefix(state_dict: dict, prefix: str = "model.") -> dict:
    """
    Remove a common Lightning prefix from state_dict keys if present.
    """
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state[k[len(prefix):]] = v
        else:
            new_state[k] = v
    return new_state


def load_model(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    n_classes: int = 4,
) -> torch.nn.Module:
    """
    Load a UNet model for multi-class segmentation.

    - checkpoint_path: path to a saved state_dict or a Lightning checkpoint.
    - device: "cpu" or "cuda".
    - n_classes: number of output classes (including background).

    The function handles:
    - raw state_dict saved with torch.save(model.state_dict())
    - Lightning checkpoint dict containing "state_dict"
    - state_dicts with a "model." prefix on keys (common in Lightning)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Instantiate model with correct number of output channels
    model = UNet(out_channels=int(n_classes))

    # Load checkpoint
    ck = torch.load(str(checkpoint_path), map_location=device)

    # Determine how to extract a state_dict
    if isinstance(ck, dict):
        # Lightning-style checkpoint often contains 'state_dict'
        if "state_dict" in ck and isinstance(ck["state_dict"], dict):
            state = ck["state_dict"]
            # Remove Lightning prefixes if present
            state = _strip_lightning_prefix(state, prefix="model.")
            state = _strip_lightning_prefix(state, prefix="net.")
            model.load_state_dict(state)
        else:
            # Might already be a raw state_dict
            try:
                model.load_state_dict(ck)
            except Exception as e:
                # Try stripping common prefixes and retry
                stripped = _strip_lightning_prefix(ck, prefix="model.")
                stripped = _strip_lightning_prefix(stripped, prefix="net.")
                model.load_state_dict(stripped)
    else:
        # ck is likely a raw state_dict mapping
        model.load_state_dict(ck)

    model.to(device)
    model.eval()
    return model
