#!/usr/bin/env python3
"""
preprocess_smoke_test.py

Usage examples:
  python preprocess_smoke_test.py --preprocessed_root "path/to/preprocessed" --n_classes 4
  python preprocess_smoke_test.py --preprocessed_root "path/to/preprocessed" --n_classes 4 --check_model twin_core.utils.UNET_model.UNet

This script is intentionally conservative and prints clear diagnostics.
"""
import argparse
import glob
import importlib
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# -------------------------
# Helpers
# -------------------------
def sample_mask_paths(preprocessed_root: Path, max_samples: int = 16):
    pattern = str(preprocessed_root / "train" / "*" / "masks" / "*.npy")
    all_masks = glob.glob(pattern)
    if not all_masks:
        return []
    return random.sample(all_masks, min(len(all_masks), max_samples))


def load_np(path: str):
    try:
        arr = np.load(path)
        return arr
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        return None


def print_mask_info(path: str):
    arr = load_np(path)
    if arr is None:
        return
    print(f"Mask: {path}")
    print(f"  dtype: {arr.dtype}, shape: {arr.shape}, size: {arr.size}")
    try:
        uniques = np.unique(arr)
        print(f"  unique labels (count {len(uniques)}): {uniques}")
    except Exception as e:
        print(f"  Could not compute unique labels: {e}")


def print_image_info(path: str):
    arr = load_np(path)
    if arr is None:
        return
    print(f"Image: {path}")
    print(f"  dtype: {arr.dtype}, shape: {arr.shape}, min/max: {np.min(arr):.6g}/{np.max(arr):.6g}")


def find_corresponding_image(mask_path: str) -> Optional[str]:
    # mask_path ends with .../patientXXX/masks/tXX_zYY.npy
    p = Path(mask_path)
    patient_dir = p.parents[1]  # .../patientXXX
    image_path = patient_dir / "data" / p.name
    return str(image_path) if image_path.exists() else None


# -------------------------
# Optional model forward check
# -------------------------
def run_model_forward_check(model_path: str, n_classes: int, sample_images: list):
    """
    model_path: dotted path to class, e.g. 'twin_core.utils.UNET_model.UNet'
    sample_images: list of image file paths to use as a small batch
    """
    print("\n=== Model forward check ===")
    try:
        module_path, class_name = model_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        ModelClass = getattr(mod, class_name)
    except Exception as e:
        print(f"Failed to import model {model_path}: {e}")
        return

    # instantiate model safely (try common kwarg names)
    model = None
    for kw in ("out_channels", "num_classes", "n_classes", "channels"):
        try:
            model = ModelClass(**{kw: n_classes})
            print(f"Instantiated {class_name} with kwarg {kw}={n_classes}")
            break
        except TypeError:
            continue
        except Exception as e:
            print(f"Model constructor raised for kw {kw}: {e}")
            raise

    if model is None:
        try:
            model = ModelClass(n_classes)
            print(f"Instantiated {class_name} positionally with {n_classes}")
        except Exception as e:
            print(f"Failed to instantiate {class_name}: {e}")
            return

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # build a small batch (B,1,H,W)
    imgs = []
    for p in sample_images:
        arr = np.load(p).astype(np.float32)
        # if preprocessing normalized, keep as-is; otherwise normalize minimally
        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected image shape {arr.shape} for {p}")
        imgs.append(arr)
    # pad/truncate to same shape if needed
    H = max(a.shape[0] for a in imgs)
    W = max(a.shape[1] for a in imgs)
    batch = []
    for a in imgs:
        pad_h = H - a.shape[0]
        pad_w = W - a.shape[1]
        if pad_h < 0 or pad_w < 0:
            a = a[:H, :W]
        else:
            a = np.pad(a, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        batch.append(a)
    batch_np = np.stack(batch, axis=0)  # (B,H,W)
    batch_t = torch.from_numpy(batch_np).unsqueeze(1).to(device)  # (B,1,H,W)

    with torch.no_grad():
        out = model(batch_t)
    print(f"Model output shape: {tuple(out.shape)} (expect B x {n_classes} x H x W)")
    if not torch.isfinite(out).all():
        print("  WARNING: model output contains non-finite values")
    else:
        print("  Model output is finite")


# -------------------------
# Main
# -------------------------
def main(preprocessed_root: str, n_classes: int, max_samples: int, check_model: Optional[str]):
    preprocessed_root = Path(preprocessed_root)
    if not preprocessed_root.exists():
        raise FileNotFoundError(f"Preprocessed root not found: {preprocessed_root}")

    print("Scanning masks (train)...")
    masks = sample_mask_paths(preprocessed_root, max_samples)
    if not masks:
        print("No mask files found under preprocessed/train/*/masks/*.npy")
        return

    # Print info for each sampled mask and its corresponding image
    for m in masks:
        print_mask_info(m)
        img = find_corresponding_image(m)
        if img:
            print_image_info(img)
        print("-" * 60)

    # Global label range check
    all_max = 0
    for m in masks:
        arr = load_np(m)
        if arr is None or arr.size == 0:
            continue
        v = int(np.max(arr))
        if v > all_max:
            all_max = v
    print(f"\nObserved max label across sampled masks: {all_max}")
    if all_max >= n_classes:
        print(f"ERROR: n_classes={n_classes} is too small. Use n_classes >= {all_max + 1}")
    else:
        print(f"n_classes={n_classes} is sufficient for sampled masks.")

    # Optional model check
    if check_model:
        # pick up to 4 images for a small batch
        sample_images = []
        for m in masks:
            img = find_corresponding_image(m)
            if img:
                sample_images.append(img)
            if len(sample_images) >= 4:
                break
        if not sample_images:
            print("No images found for model forward check.")
        else:
            run_model_forward_check(check_model, n_classes, sample_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick smoke tests for preprocessed dataset and optional UNet forward pass")
    parser.add_argument("--preprocessed_root", required=True, help="Path to preprocessed/ root")
    parser.add_argument("--n_classes", type=int, default=4, help="Number of classes including background")
    parser.add_argument("--max_samples", type=int, default=8, help="How many mask samples to inspect")
    parser.add_argument("--check_model", type=str, default=None, help="Optional dotted path to UNet class, e.g. twin_core.utils.UNET_model.UNet")
    args = parser.parse_args()

    main(args.preprocessed_root, args.n_classes, args.max_samples, args.check_model)
