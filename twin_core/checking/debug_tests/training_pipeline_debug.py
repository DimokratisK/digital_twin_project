#!/usr/bin/env python3
"""
training_pipeline_debug.py

Disposable debug training script that attempts to overfit a single batch.

Purpose:
- Quickly verify model, dataloader, loss, and optimizer are wired correctly.
- Fail fast and print informative messages.
- No checkpointing, no validation, minimal external dependencies.

Behavior:
- Build dataloaders via get_dataloaders(dl_cfg) when possible, otherwise create a fallback DataLoader.
- Grab one training batch and repeatedly run gradient updates on that batch.
- Prints loss / CE / Dice every `print_every` iterations.
"""
from pathlib import Path
import json
import time
import sys
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Try to import project utilities; fall back gracefully if something changed
try:
    from twin_core.data_ingestion.dataloaders import get_dataloaders
except Exception:
    get_dataloaders = None

try:
    from twin_core.data_ingestion.dataset import CardiacDataset
except Exception:
    CardiacDataset = None

from twin_core.utils.UNET_model import UNet

# -------------------------
# Utility losses
# -------------------------
def dice_loss_logits(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Mean (1 - Dice) across classes computed from logits.
    pred_logits: (B, C, H, W)
    target: (B, H, W) integer labels
    """
    if target.dim() == 4:
        target = target.squeeze(1)
    num_classes = pred_logits.shape[1]
    pred = torch.softmax(pred_logits, dim=1)
    target_onehot = torch.nn.functional.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (pred * target_onehot).sum(dims)
    union = pred.sum(dims) + target_onehot.sum(dims)
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


# -------------------------
# Build dataloaders (robust)
# -------------------------
def build_dataloaders(cfg: Dict[str, Any]):
    data_root = Path(cfg.get("data_root", "preprocessed"))
    manifest_path = cfg.get("manifest_path", str(data_root / "mask_manifest.json"))
    split_manifest_path = cfg.get("split_manifest_path", str(data_root / "split_manifest_labeled.json"))
    if not Path(split_manifest_path).exists():
        split_manifest_path = str(data_root / "split_manifest.json")

    dl_cfg = {
        "data_root": str(data_root),
        "manifest_path": manifest_path,
        "split_manifest_path": split_manifest_path,
        "use_labeled_only": cfg.get("use_labeled_only", True),
        "sampler_type": cfg.get("sampler_type", "none"),
        "batch_size": int(cfg.get("batch_size", 4)),
        "num_workers": int(cfg.get("num_workers", 0)),
        "prefer_ed_es": cfg.get("prefer_ed_es", False),
        "one_hot": cfg.get("one_hot", False),
        "n_classes": int(cfg.get("n_classes", 4)),
        "pad_multiple": int(cfg.get("pad_multiple", 16)),
        "pin_memory": bool(cfg.get("pin_memory", False)),
        "exclude_missing_masks": bool(cfg.get("exclude_missing_masks", False)),
    }

    # Attempt main factory first
    if get_dataloaders is not None:
        try:
            dls = get_dataloaders(dl_cfg)
            print("[debug] get_dataloaders succeeded")
            return dls
        except Exception as e:
            print(f"[debug] get_dataloaders failed: {e}; will fall back to CardiacDataset + DataLoader")

    # Fallback: create simple CardiacDataset and DataLoader
    if CardiacDataset is None:
        raise RuntimeError("Neither get_dataloaders nor CardiacDataset are importable. Cannot build dataloaders.")

    train_root = Path(data_root) / "train"
    val_root = Path(data_root) / "val"

    train_ds = CardiacDataset(train_root,
                              augment=None,
                              prefer_ed_es=cfg.get("prefer_ed_es", False),
                              metadata_index=None,
                              one_hot=cfg.get("one_hot", False),
                              n_classes=cfg.get("n_classes", 4),
                              exclude_missing_masks=cfg.get("exclude_missing_masks", True),
                              pad_multiple=cfg.get("pad_multiple", 16))

    val_ds = CardiacDataset(val_root,
                            augment=None,
                            prefer_ed_es=False,
                            metadata_index=None,
                            one_hot=cfg.get("one_hot", False),
                            n_classes=cfg.get("n_classes", 4),
                            exclude_missing_masks=cfg.get("exclude_missing_masks", True),
                            pad_multiple=cfg.get("pad_multiple", 16))

    train_loader = DataLoader(train_ds, batch_size=dl_cfg["batch_size"], shuffle=True, num_workers=dl_cfg["num_workers"], pin_memory=dl_cfg["pin_memory"])
    val_loader = DataLoader(val_ds, batch_size=dl_cfg["batch_size"], shuffle=False, num_workers=dl_cfg["num_workers"], pin_memory=dl_cfg["pin_memory"])

    return {"train": train_loader, "val": val_loader}


# -------------------------
# Overfit single batch loop
# -------------------------
def overfit_one_batch(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.get("seed", 42)))

    dls = build_dataloaders(cfg)
    train_loader: DataLoader = dls["train"]
    if len(train_loader) == 0:
        raise RuntimeError("Train loader is empty; nothing to debug.")

    # Grab one batch
    batch = None
    for b in train_loader:
        batch = b
        break
    if batch is None:
        raise RuntimeError("Failed to fetch a batch from train loader.")

    imgs, masks = batch
    # Masks expected shape (B,1,H,W) or (B,H,W)
    if masks.dim() == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    imgs = imgs.to(device, dtype=torch.float32)
    masks = masks.to(device, dtype=torch.long)

    print(f"[debug] using one batch with images {imgs.shape} masks {masks.shape} on device {device}")

    # Build model
    in_ch = int(cfg.get("in_channels", 1))
    n_classes = int(cfg.get("n_classes", 4))
    base_features = int(cfg.get("base_features", 64))
    model = UNet(in_channels=in_ch, out_channels=n_classes, base_features=base_features)
    model = model.to(device)

    # Optimizer & losses
    lr = float(cfg.get("lr", 1e-3))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce_fn = nn.CrossEntropyLoss()
    use_dice = bool(cfg.get("use_dice_loss", True))
    dice_w = float(cfg.get("dice_weight", 1.0))

    # AMP optional (disabled by default for debug)
    use_amp = bool(cfg.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    # Overfit loop params
    num_iters = int(cfg.get("debug_iters", 200))
    print_every = int(cfg.get("debug_print_every", 10))

    model.train()
    losses = []
    t0 = time.time()
    for it in range(1, num_iters + 1):
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                ce = ce_fn(logits, masks)
                dice = dice_loss_logits(logits, masks) if use_dice else torch.tensor(0.0, device=device)
                loss = ce + dice_w * dice
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            ce = ce_fn(logits, masks)
            dice = dice_loss_logits(logits, masks) if use_dice else torch.tensor(0.0, device=device)
            loss = ce + dice_w * dice
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))

        if it % print_every == 0 or it == 1:
            elapsed = time.time() - t0
            avg = sum(losses[-print_every:]) / min(len(losses), print_every)
            print(f"[iter {it:04d}/{num_iters}] loss={loss.item():.6f}  ce={float(ce.item()):.6f}  dice={float(dice.item() if isinstance(dice, torch.Tensor) else dice):.6f}  avg_last={avg:.6f}  elapsed={elapsed:.1f}s")

    print("[debug] finished overfit run. Loss history (last 10):", losses[-10:])
    return {"losses": losses, "model": model}


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Debug training: overfit one batch to validate pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON (optional).")
    args = parser.parse_args()

    # defaults tuned for fast debug
    cfg = {
        "data_root": "preprocessed",
        "manifest_path": "preprocessed/mask_manifest.json",
        "split_manifest_path": "preprocessed/split_manifest_labeled.json",
        "use_labeled_only": True,
        "batch_size": 4,
        "num_workers": 0,
        "lr": 1e-3,
        "n_classes": 4,
        "in_channels": 1,
        "base_features": 64,
        "use_amp": False,
        "use_dice_loss": True,
        "dice_weight": 1.0,
        "debug_iters": 200,
        "debug_print_every": 10,
        "exclude_missing_masks": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise RuntimeError(f"Config file not found: {cfg_path}")
        cfg.update(json.load(open(cfg_path, "r", encoding="utf-8")))

    print("[debug] config:", {k: cfg[k] for k in sorted(cfg) if k in ("data_root", "batch_size", "debug_iters", "use_amp", "device", "n_classes")})
    out = overfit_one_batch(cfg)
    print("[debug] done.")
