#!/usr/bin/env python3
"""
Refactored training pipeline (supervised) that uses labeled patients by default.

- Uses the dataloader factory `get_dataloaders(cfg)` to build train/val loaders.
- Defaults to the labeled-only split (split_manifest_labeled.json) when available.
- Uses the repository UNet implementation (utils/UNET_model.py).
- Stable pipeline interface: build(cfg) -> state, train_epoch(state,cfg), validate(state,cfg), save(state,path), load(path,device).
- Checkpoints include model/optim/epoch/cfg to allow future semi-supervised extensions.
"""
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from twin_core.data_ingestion.dataloaders import get_dataloaders
from twin_core.utils.UNET_model import UNet
# Dataset wrapper (metadata helpers)
from twin_core.data_ingestion.dataset_wrapper import CardiacDatasetWithMeta



# -------------------------
# Loss helpers
# -------------------------
def dice_loss_logits(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute mean (1 - Dice) across classes from logits.
    pred_logits: (B, C, H, W)
    target: (B, 1, H, W) integer labels or (B, H, W)
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
# Pipeline interface
# -------------------------
def build(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build model, optimizer, scheduler, dataloaders and initial state.
    Returns a state dict consumed by train_epoch and validate.
    """
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    data_root = Path(cfg.get("data_root", "preprocessed"))
    manifest_path = cfg.get("manifest_path", str(data_root / "mask_manifest.json"))

    # Prefer labeled split by default (Option 1)
    split_manifest_path = cfg.get("split_manifest_path", str(data_root / "split_manifest_labeled.json"))
    if not Path(split_manifest_path).exists():
        split_manifest_path = str(data_root / "split_manifest.json")

    # Build dataloaders via factory
    dl_cfg = {
        "data_root": data_root,
        "manifest_path": manifest_path,
        "split_manifest_path": split_manifest_path,
        "use_labeled_only": True,
        "sampler_type": cfg.get("sampler_type", "none"),
        "labeled_weight": cfg.get("labeled_weight", 5.0),
        "batch_size": cfg.get("batch_size", 4),
        "num_workers": cfg.get("num_workers", 4),
        "use_manifest_for_label_filter": cfg.get("use_manifest_for_label_filter", True),
        "val_use_labeled_only": cfg.get("val_use_labeled_only", True),
        "prefer_ed_es": cfg.get("prefer_ed_es", False),
        "one_hot": cfg.get("one_hot", False),
        "n_classes": cfg.get("n_classes", 4),
        "pad_multiple": cfg.get("pad_multiple", 16),
    }

    dataloaders = get_dataloaders(dl_cfg)
    train_loader: DataLoader = dataloaders["train"]
    val_loader: DataLoader = dataloaders["val"]

    # Fail-fast checks
    if len(train_loader.dataset) == 0:
        raise RuntimeError("Train dataset is empty. Check split_manifest_labeled.json and mask_manifest.json.")
    if cfg.get("require_val_labels", True) and len(val_loader.dataset) == 0:
        raise RuntimeError("Validation dataset is empty (no labeled samples). Check split_manifest_labeled.json and mask_manifest.json.")

    # Model: use repository UNet
    in_ch = int(cfg.get("in_channels", 1))
    n_classes = int(cfg.get("n_classes", 4))
    base_features = int(cfg.get("base_features", 64))
    use_bn = bool(cfg.get("use_bn", True))
    dropout = cfg.get("dropout", None)

    model = UNet(
        in_channels=in_ch,
        out_channels=n_classes,
        base_features=base_features,
        use_bn=use_bn,
        dropout=dropout,
    )
    model = model.to(device)

    # Optimizer and scheduler
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5) if cfg.get("use_plateau_scheduler", True) else None

    # AMP support
    use_amp = bool(cfg.get("use_amp", False))
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    state = {
        "cfg": cfg,
        "device": device,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "dataloaders": dataloaders,
        "start_epoch": 0,
        "best_val_loss": float("inf"),
        "use_amp": use_amp,
        "scaler": scaler,
    }
    return state


def train_epoch(state: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    """
    Run one training epoch. Returns updated state with metrics.
    """
    model: nn.Module = state["model"]
    optimizer: optim.Optimizer = state["optimizer"]
    device = state["device"]
    train_loader: DataLoader = state["dataloaders"]["train"]

    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    n_batches = 0
    log_every = 10
    ce_loss_fn = nn.CrossEntropyLoss()
    running_loss = torch.tensor(0.0, device=device)
    running_ce = torch.tensor(0.0, device=device)
    running_dice = torch.tensor(0.0, device=device)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train epoch {epoch}", leave=False)

    for i, batch in pbar:
        imgs, masks = batch
        imgs = imgs.to(device, dtype=torch.float32, non_blocking=False)
        masks = masks.to(device, dtype=torch.long).squeeze(1)

        optimizer.zero_grad()

        if state.get("use_amp", False) and device.type == "cuda":
            scaler = state.get("scaler")
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                ce = ce_loss_fn(logits, masks)
                dice = dice_loss_logits(logits, masks) if state["cfg"].get("use_dice_loss", True) else torch.tensor(0.0, device=device)
                loss = ce + float(state["cfg"].get("dice_weight", 1.0)) * dice
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            ce = ce_loss_fn(logits, masks)
            dice = dice_loss_logits(logits, masks) if state["cfg"].get("use_dice_loss", True) else torch.tensor(0.0, device=device)
            loss = ce + float(state["cfg"].get("dice_weight", 1.0)) * dice
            loss.backward()
            optimizer.step()

        running_loss += loss.detach()
        running_ce += ce.detach()
        running_dice += dice.detach()

        if (i + 1) % log_every == 0 or (i + 1) == len(train_loader):
            if device.type == "cuda":
                torch.cuda.synchronize()
            avg_loss = float(running_loss.item()) / log_every
            avg_ce = float(running_ce.item()) / log_every
            avg_dice = float(running_dice.item()) / log_every
            total_loss += avg_loss * log_every
            total_ce += avg_ce * log_every
            total_dice += avg_dice * log_every
            n_batches += log_every
            running_loss = torch.tensor(0.0, device=device)
            running_ce = torch.tensor(0.0, device=device)
            running_dice = torch.tensor(0.0, device=device)
            pbar.set_postfix({"loss": total_loss / n_batches, "ce": total_ce / n_batches, "dice": total_dice / max(1, n_batches)})

    avg_loss = total_loss / max(1, n_batches)
    avg_ce = total_ce / max(1, n_batches)
    avg_dice = total_dice / max(1, n_batches)

    state["last_train_loss"] = avg_loss
    state["last_train_ce"] = avg_ce
    state["last_train_dice"] = avg_dice
    return state


def validate(state: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    """
    Run validation over labeled validation set. Returns updated state with validation metrics.
    """
    model: nn.Module = state["model"]
    device = state["device"]
    val_loader: DataLoader = state["dataloaders"]["val"]

    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    n_batches = 0
    ce_loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validate epoch {epoch}", leave=False)
        for i, batch in pbar:
            imgs, masks = batch
            imgs = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long).squeeze(1)

            logits = model(imgs)
            ce = ce_loss_fn(logits, masks)
            dice = dice_loss_logits(logits, masks) if state["cfg"].get("use_dice_loss", True) else torch.tensor(0.0, device=device)
            loss = ce + float(state["cfg"].get("dice_weight", 1.0)) * dice

            total_loss += float(loss.item())
            total_ce += float(ce.item())
            total_dice += float(dice.item()) if isinstance(dice, torch.Tensor) else float(dice)
            n_batches += 1

            pbar.set_postfix({"val_loss": total_loss / n_batches})

    avg_loss = total_loss / max(1, n_batches)
    avg_ce = total_ce / max(1, n_batches)
    avg_dice = total_dice / max(1, n_batches)

    state["last_val_loss"] = avg_loss
    state["last_val_ce"] = avg_ce
    state["last_val_dice"] = avg_dice

    # Scheduler step if using ReduceLROnPlateau
    if state.get("scheduler") is not None:
        try:
            state["scheduler"].step(avg_loss)
        except Exception:
            pass

    # Track best
    if avg_loss < state.get("best_val_loss", float("inf")):
        state["best_val_loss"] = avg_loss
        state["best_epoch"] = epoch
        state["is_best"] = True
    else:
        state["is_best"] = False

    return state


def save(state: Dict[str, Any], path: str) -> None:
    """
    Save checkpoint. Includes model/optimizer/epoch/cfg and best metric.
    """
    ckpt = {
        "model_state": state["model"].state_dict(),
        "optimizer_state": state["optimizer"].state_dict(),
        "epoch": state.get("epoch", 0),
        "best_val_loss": state.get("best_val_loss", float("inf")),
        "cfg": state.get("cfg", {}),
    }
    scaler = state.get("scaler", None)
    if scaler is not None:
        try:
            ckpt["scaler_state"] = scaler.state_dict()
        except Exception:
            pass
    torch.save(ckpt, path)


def load(path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load checkpoint and return dict with keys matching save().
    """
    map_location = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=map_location)
    return ckpt


# -------------------------
# High-level train loop entrypoint
# -------------------------
def train(cfg: Dict[str, Any]) -> None:
    """
    High-level training entrypoint. Builds state, runs epochs, saves checkpoints.
    """
    import signal
    import sys
    state = build(cfg)
    model = state["model"]
    optimizer = state["optimizer"]
    device = state["device"]

    max_epochs = int(cfg.get("max_epochs", 100))
    ckpt_dir = Path(cfg.get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_interval = int(cfg.get("ckpt_interval", 1))

    # initial diagnostics
    print("Training start. Device:", device)
    print("Train samples:", len(state["dataloaders"]["train"].dataset), "Val samples:", len(state["dataloaders"]["val"].dataset))
    print("Config summary:", {k: cfg[k] for k in sorted(cfg) if k in ("batch_size", "lr", "max_epochs", "use_labeled_only", "sampler_type")})

    # Register signal handler to save checkpoint on termination
    def _save_on_signal(signum, frame):
        try:
            ckpt_path = ckpt_dir / f"ckpt_epoch{state.get('epoch', 0):03d}_signal.pt"
            save(state, str(ckpt_path))
            print(f"[signal] saved checkpoint to {ckpt_path}", flush=True)
        except Exception as e:
            print(f"[signal] failed to save checkpoint: {e}", flush=True)
        finally:
            sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _save_on_signal)
        signal.signal(signal.SIGINT, _save_on_signal)
    except Exception:
        pass

    try:
        for epoch in range(state.get("start_epoch", 0), max_epochs):
            state["epoch"] = epoch
            t0 = time.time()
            state = train_epoch(state, epoch)
            state = validate(state, epoch)

            # Save checkpoint every ckpt_interval epochs and when best
            if (epoch + 1) % ckpt_interval == 0:
                ckpt_path = ckpt_dir / f"ckpt_epoch{epoch:03d}.pt"
                save(state, str(ckpt_path))
            if state.get("is_best", False):
                best_path = ckpt_dir / "ckpt_best.pt"
                save(state, str(best_path))

            t1 = time.time()
            print(f"Epoch {epoch:03d} finished in {t1 - t0:.1f}s  train_loss={state.get('last_train_loss'):.4f}  val_loss={state.get('last_val_loss'):.4f}")
    except KeyboardInterrupt:
        ckpt_path = ckpt_dir / f"ckpt_epoch{state.get('epoch', 0):03d}_interrupt.pt"
        save(state, str(ckpt_path))
        print(f"[interrupt] saved checkpoint to {ckpt_path}", flush=True)
        raise

    print("Training finished. Best val loss:", state.get("best_val_loss"))


# -------------------------
# CLI entrypoint for quick runs
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train segmentation model (supervised, labeled-only split by default)")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (optional). If omitted, defaults are used.")
    args = parser.parse_args()

    # Default config (can be overridden by JSON file)
    cfg: Dict[str, Any] = {
        "data_root": "preprocessed",
        "manifest_path": "preprocessed/mask_manifest.json",
        "split_manifest_path": "preprocessed/split_manifest_labeled.json",
        "use_labeled_only": True,
        "sampler_type": "none",
        "labeled_weight": 5.0,
        "batch_size": 4,
        "num_workers": 4,
        "lr": 1e-3,
        "max_epochs": 50,
        "ckpt_dir": "checkpoints",
        "ckpt_interval": 1,
        "n_classes": 4,
        "in_channels": 1,
        "base_features": 64,
        "use_bn": True,
        "dropout": None,
        "use_dice_loss": True,
        "dice_weight": 1.0,
        "require_val_labels": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise RuntimeError(f"Config file not found: {cfg_path}")
        cfg_from_file = json.load(open(cfg_path, "r", encoding="utf-8"))
        cfg.update(cfg_from_file)

    train(cfg)
