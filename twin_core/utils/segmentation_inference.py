import torch
import numpy as np
from typing import Optional, Union, Tuple

from ..data_ingestion.dataset import pad_to_multiple_2d  # relative import; dataset.py must expose this helper


def _prepare_slice_tensor(slice_arr: np.ndarray, device: str) -> torch.Tensor:
    """
    Convert a 2D numpy slice (H, W) to a float tensor (1, 1, H, W) on device.
    """
    x = torch.from_numpy(slice_arr).float().unsqueeze(0).unsqueeze(0).to(device)
    return x


def _unpad_2d(arr: np.ndarray, pad: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Remove padding given (pad_top, pad_bottom, pad_left, pad_right).
    Works for 2D arrays.
    """
    pad_top, pad_bottom, pad_left, pad_right = pad
    h, w = arr.shape
    top = pad_top
    bottom = h - pad_bottom if pad_bottom > 0 else h
    left = pad_left
    right = w - pad_right if pad_right > 0 else w
    return arr[top:bottom, left:right]


def segment_volume(
    model,
    volume: Union[np.ndarray, torch.Tensor],
    device: str = "cpu",
    batch_slices: Optional[int] = None,
    pad_multiple: int = 16,
) -> np.ndarray:
    """
    Run inference on a volume and return integer label map.

    - model: PyTorch model that returns logits with shape (B, C, H, W).
    - volume: numpy array with shape (Z, H, W) or (H, W) for single-slice.
    - device: "cpu" or "cuda".
    - batch_slices: optional int to process multiple slices in a single forward pass.
    - pad_multiple: pad H and W to multiples of this value (default 16).

    Returns:
    - labels: numpy array of dtype uint8 with shape (Z, H, W) or (H, W) if input was 2D.
    """
    # Convert torch.Tensor input to numpy if needed
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()

    # Handle single 2D slice
    if volume.ndim == 2:
        # pad
        padded_slice, pad = pad_to_multiple_2d(volume, multiple=pad_multiple, mode="constant", cval=0)
        x = _prepare_slice_tensor(padded_slice, device)
        with torch.no_grad():
            logits = model(x)  # (1, C, H_p, W_p)
            probs = torch.softmax(logits, dim=1)
            labels_padded = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        labels = _unpad_2d(labels_padded, pad)
        return labels

    # Expecting 3D: (Z, H, W)
    if volume.ndim != 3:
        raise ValueError(f"Unsupported volume ndim {volume.ndim}; expected 2 or 3")

    Z, H, W = volume.shape

    # Compute padding once (all slices share H,W)
    _, pad = pad_to_multiple_2d(volume[0], multiple=pad_multiple, mode="constant", cval=0)
    pad_top, pad_bottom, pad_left, pad_right = pad

    # Pad entire volume efficiently: pad along H and W only
    padded = np.pad(
        volume,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )  # shape (Z, H_p, W_p)

    labels_list = []

    # If batch_slices provided, process in batches for speed
    if batch_slices is None or batch_slices <= 1:
        # slice-by-slice
        with torch.no_grad():
            for z in range(Z):
                slice_ = padded[z]
                x = _prepare_slice_tensor(slice_, device)
                logits = model(x)  # (1, C, H_p, W_p)
                probs = torch.softmax(logits, dim=1)
                lbl_padded = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                lbl = _unpad_2d(lbl_padded, pad)
                labels_list.append(lbl)
    else:
        # batched processing
        with torch.no_grad():
            for start in range(0, Z, batch_slices):
                end = min(start + batch_slices, Z)
                batch = padded[start:end]  # (B, H_p, W_p)
                # stack into tensor (B, 1, H_p, W_p)
                x = torch.from_numpy(batch).float().unsqueeze(1).to(device)
                logits = model(x)  # (B, C, H_p, W_p)
                probs = torch.softmax(logits, dim=1)
                lbls_padded = probs.argmax(dim=1).cpu().numpy().astype(np.uint8)  # (B, H_p, W_p)
                for b in range(lbls_padded.shape[0]):
                    lbl = _unpad_2d(lbls_padded[b], pad)
                    labels_list.append(lbl)

    labels = np.stack(labels_list, axis=0)  # (Z, H, W)
    return labels
