"""
twin_core.utils.metrics

Utility functions for segmentation metrics used across training, validation and
inference scripts.

Provided functions
- dice_per_class(pred, target, num_classes=4, ignore_index=None)
- iou_per_class(pred, target, num_classes=4, ignore_index=None)
- confusion_matrix(pred, target, num_classes=4)
- accumulate_inter_union_from_logits(logits, target, eps=1e-6)

All functions accept either NumPy arrays or PyTorch tensors for `pred`, `target`
and return NumPy arrays (except accumulate_inter_union_from_logits which returns
torch.Tensors to be convenient for training accumulation).
"""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(x):
    """Convert torch Tensor to numpy, or pass through numpy arrays."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def dice_per_class(pred, target, num_classes: int = 4, ignore_index: Optional[int] = None) -> np.ndarray:
    """
    Compute per-class Dice coefficient (not loss) between integer label arrays.

    Parameters
    - pred: (H,W) or (N,H,W) integer labels or one-hot/prob maps (will be argmaxed)
    - target: same shape as pred (integer labels)
    - num_classes: number of classes
    - ignore_index: label value to ignore in both pred and target (e.g., 255)

    Returns
    - numpy array shape (num_classes,) with Dice in [0,1]
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)

    # If probabilities or logits provided, reduce to labels
    if p.ndim >= 3 and p.dtype != np.integer:
        # assume shape (C,H,W) or (N,C,H,W) or (C,H,W) -> argmax over channel dim
        if p.ndim == 3:
            p = np.argmax(p, axis=0)
        else:
            p = np.argmax(p, axis=1)
    if t.ndim >= 3 and t.dtype != np.integer:
        if t.ndim == 3:
            t = np.argmax(t, axis=0)
        else:
            t = np.argmax(t, axis=1)

    # If batch dimension present, flatten across batch
    if p.ndim == 3:
        # (N,H,W) -> flatten to (N*H*W,)
        p = p.reshape(-1)
    else:
        p = p.reshape(-1)
    if t.ndim == 3:
        t = t.reshape(-1)
    else:
        t = t.reshape(-1)

    scores = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if ignore_index is not None:
            mask = (t != ignore_index)
            p_c = (p[mask] == c).astype(np.uint8)
            t_c = (t[mask] == c).astype(np.uint8)
        else:
            p_c = (p == c).astype(np.uint8)
            t_c = (t == c).astype(np.uint8)

        inter = (p_c & t_c).sum()
        denom = p_c.sum() + t_c.sum()
        if denom == 0:
            scores[c] = 1.0  # class absent in both -> perfect
        else:
            scores[c] = (2.0 * inter) / float(denom)
    return scores


def iou_per_class(pred, target, num_classes: int = 4, ignore_index: Optional[int] = None) -> np.ndarray:
    """
    Compute per-class Intersection-over-Union (Jaccard) between integer label arrays.

    Returns numpy array shape (num_classes,) with IoU in [0,1].
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)

    # reduce probabilities/logits to labels if necessary
    if p.ndim >= 3 and p.dtype != np.integer:
        if p.ndim == 3:
            p = np.argmax(p, axis=0)
        else:
            p = np.argmax(p, axis=1)
    if t.ndim >= 3 and t.dtype != np.integer:
        if t.ndim == 3:
            t = np.argmax(t, axis=0)
        else:
            t = np.argmax(t, axis=1)

    p = p.reshape(-1)
    t = t.reshape(-1)

    scores = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if ignore_index is not None:
            mask = (t != ignore_index)
            p_c = (p[mask] == c).astype(np.uint8)
            t_c = (t[mask] == c).astype(np.uint8)
        else:
            p_c = (p == c).astype(np.uint8)
            t_c = (t == c).astype(np.uint8)

        inter = (p_c & t_c).sum()
        union = (p_c | t_c).sum()
        if union == 0:
            scores[c] = 1.0
        else:
            scores[c] = inter / float(union)
    return scores


def confusion_matrix(pred, target, num_classes: int = 4, ignore_index: Optional[int] = None) -> np.ndarray:
    """
    Compute confusion matrix (predicted x true) as numpy array shape (num_classes, num_classes).
    Rows = predicted class, Columns = true class.
    """
    p = _to_numpy(pred).reshape(-1)
    t = _to_numpy(target).reshape(-1)

    if ignore_index is not None:
        mask = (t != ignore_index)
        p = p[mask]
        t = t[mask]

    # clamp labels to valid range
    p = np.clip(p, 0, num_classes - 1).astype(np.int64)
    t = np.clip(t, 0, num_classes - 1).astype(np.int64)

    idx = p * num_classes + t
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    conf = counts.reshape((num_classes, num_classes)).astype(np.int64)
    return conf


def accumulate_inter_union_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given logits (B, C, H, W) and integer target (B, H, W) compute per-class
    soft intersection and (pred_sum + target_sum) terms as torch tensors.

    Returns (inter, union_term) each shape (C,) on the same device as logits.
    - inter = sum_over_pixels(softmax_probs * target_onehot)
    - union_term = sum_over_pixels(softmax_probs) + sum_over_pixels(target_onehot)
    """
    if target.dim() == 4:
        target = target.squeeze(1)
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)  # (B,C,H,W)
    target_onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * target_onehot).sum(dims)  # (C,)
    union_term = probs.sum(dims) + target_onehot.sum(dims)  # (C,)
    return inter, union_term
