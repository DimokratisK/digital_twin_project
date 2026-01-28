#!/usr/bin/env python3
"""
Refactored training pipeline (supervised) with AMP, robust checkpointing,
and publication-grade metrics:

- Per-class Dice score and Dice loss (epoch-aggregated).
- Mean Dice (optionally excluding background).
- Confusion matrix (predicted x true) saved per epoch.
- TensorBoard scalars and CSV logging (detailed per-class columns).
"""
from pathlib import Path
import json
import time
import signal
import sys
import csv
from typing import Dict, Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from twin_core.data_ingestion.dataloaders import get_dataloaders
from twin_core.utils.UNET_model import UNet

# Optional TensorBoard writer
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# -------------------------
# Helpers: Dice / confusion utilities
# -------------------------
def _safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return a / (b + eps)


def accumulate_inter_union_from_logits(
    logits: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given logits (B, C, H, W) and integer target (B, H, W),
    compute class-wise intersection and (pred_sum + target_sum) terms.

    Returns:
      inter: Tensor[C]  (sum over batch and pixels of prob * target_onehot)
      union_term: Tensor[C]  (pred_prob_sum + target_onehot_sum)
    """
    if target.dim() == 4:
        target = target.squeeze(1)
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
    # target one-hot -> (B, C, H, W)
    target_onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)  # batch + H + W
    inter = (probs * target_onehot).sum(dims)  # per-class soft intersection
    union_term = probs.sum(dims) + target_onehot.sum(dims)
    return inter, union_term


def compute_per_class_dice_from_inter_union(inter: torch.Tensor, union_term: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Given per-class inter and union_term (pred_sum + target_sum),
    return per-class Dice score: (2*inter + eps) / (union_term + eps)
    """
    dice = (2.0 * inter + eps) / (union_term + eps)
    return dice


def confusion_matrix_from_preds_and_targets(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:
    """
    Build confusion matrix for a batch.
    preds: (B, H, W) integer predicted labels
    targets: (B, H, W) integer true labels
    Returns numpy array shape (num_classes, num_classes) where
      rows = predicted class, cols = true class
    """
    # flatten and move to cpu ints
    p = preds.view(-1).cpu().to(torch.int64)
    t = targets.view(-1).cpu().to(torch.int64)
    # filter out any impossible labels (>num_classes-1) by clipping
    p = torch.clamp(p, 0, num_classes - 1)
    t = torch.clamp(t, 0, num_classes - 1)
    idx = p * num_classes + t  # linear index
    counts = torch.bincount(idx, minlength=num_classes * num_classes)
    conf = counts.reshape(num_classes, num_classes).numpy().astype(np.int64)
    return conf


# -------------------------
# Pipeline interface
# -------------------------
def build(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build model, optimizer, scheduler, dataloaders and initial state.
    Returns a state dict consumed by train_epoch and validate.
    """
    # Device
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Optional cuDNN tuning
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # Data
    data_root = Path(cfg.get("data_root", "preprocessed"))
    manifest_path = cfg.get("manifest_path", str(data_root / "mask_manifest.json"))

    split_manifest_path = cfg.get("split_manifest_path", str(data_root / "split_manifest_labeled.json"))
    if not Path(split_manifest_path).exists():
        split_manifest_path = str(data_root / "split_manifest.json")

    dl_cfg = {
        "data_root": data_root,
        "manifest_path": manifest_path,
        "split_manifest_path": split_manifest_path,
        "use_labeled_only": cfg.get("use_labeled_only", True),
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
        "pin_memory": cfg.get("pin_memory", False),
        "exclude_missing_masks": cfg.get("exclude_missing_masks", False),
    }

    dataloaders = get_dataloaders(dl_cfg)
    train_loader: DataLoader = dataloaders["train"]
    val_loader: DataLoader = dataloaders["val"]

    # Fail-fast checks
    if len(train_loader.dataset) == 0:
        raise RuntimeError("Train dataset is empty. Check split_manifest_labeled.json and mask_manifest.json.")
    if cfg.get("require_val_labels", True) and len(val_loader.dataset) == 0:
        raise RuntimeError("Validation dataset is empty (no labeled samples). Check split_manifest_labeled.json and mask_manifest.json.")

    # Model
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

    # class names and dice options
    class_names: Sequence[str] = cfg.get("class_names", ["BG", "RV", "MYO", "LV"])
    exclude_background_from_dice: bool = bool(cfg.get("exclude_background_from_dice", True))

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
        # logging placeholders
        "writer": None,
        "metrics_csv": None,
        "class_names": list(class_names),
        "exclude_background_from_dice": exclude_background_from_dice,
    }
    return state


# -------------------------
# Training / validation epoch implementations
# -------------------------
def train_epoch(state: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    model: nn.Module = state["model"]
    optimizer: optim.Optimizer = state["optimizer"]
    device = state["device"]
    train_loader: DataLoader = state["dataloaders"]["train"]
    cfg = state["cfg"]

    model.train()
    # epoch accumulators
    total_loss = 0.0
    total_ce = 0.0
    n_batches = 0

    num_classes = int(cfg.get("n_classes", 4))
    # accumulate inter and union across epoch on device
    inter_acc = torch.zeros(num_classes, device=device)
    union_acc = torch.zeros(num_classes, device=device)
    # confusion matrix accum on CPU
    conf_acc = np.zeros((num_classes, num_classes), dtype=np.int64)

    ce_loss_fn = nn.CrossEntropyLoss()

    use_amp = state.get("use_amp", False) and device.type == "cuda"
    scaler = state.get("scaler", None)

    log_every = int(cfg.get("log_every", 10))
    sync_for_logging = bool(cfg.get("sync_for_logging", False))

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train epoch {epoch}", leave=False)

    for i, batch in pbar:
        imgs, masks = batch
        imgs = imgs.to(device, dtype=torch.float32, non_blocking=False)
        masks = masks.to(device, dtype=torch.long).squeeze(1)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                ce = ce_loss_fn(logits, masks)
                # compute inter/union for dice accumulation
                inter_b, union_b = accumulate_inter_union_from_logits(logits, masks)
                # compute dice loss scalar (mean across classes or excluding background later)
                per_class_dice_b = compute_per_class_dice_from_inter_union(inter_b, union_b)
                if state["exclude_background_from_dice"] and num_classes > 1:
                    mask_cls = torch.arange(num_classes, device=device) != 0
                    dice_loss_b = 1.0 - per_class_dice_b[mask_cls].mean()
                else:
                    dice_loss_b = 1.0 - per_class_dice_b.mean()
                loss = ce + float(cfg.get("dice_weight", 1.0)) * dice_loss_b
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            ce = ce_loss_fn(logits, masks)
            inter_b, union_b = accumulate_inter_union_from_logits(logits, masks)
            per_class_dice_b = compute_per_class_dice_from_inter_union(inter_b, union_b)
            if state["exclude_background_from_dice"] and num_classes > 1:
                mask_cls = torch.arange(num_classes, device=device) != 0
                dice_loss_b = 1.0 - per_class_dice_b[mask_cls].mean()
            else:
                dice_loss_b = 1.0 - per_class_dice_b.mean()
            loss = ce + float(cfg.get("dice_weight", 1.0)) * dice_loss_b
            loss.backward()
            optimizer.step()

        # detach inter/union before accumulation to keep metrics out of autograd graph
        inter_acc += inter_b.detach()
        union_acc += union_b.detach()

        # confusion matrix (CPU)
        preds = logits.argmax(dim=1)  # (B, H, W)
        conf_batch = confusion_matrix_from_preds_and_targets(preds, masks, num_classes=num_classes)
        conf_acc += conf_batch

        # accumulate scalar metrics
        total_loss += float(loss.detach().item())
        total_ce += float(ce.detach().item())
        n_batches += 1

        # periodic progress postfix
        if (i + 1) % log_every == 0 or (i + 1) == len(train_loader):
            # compute interim epoch-averages
            avg_loss = total_loss / n_batches
            avg_ce = total_ce / n_batches
            # compute interim per-class dice (device->cpu). detach to be safe.
            with torch.no_grad():
                per_class_dice_epoch = compute_per_class_dice_from_inter_union(inter_acc, union_acc).detach().cpu().numpy()
                if state["exclude_background_from_dice"] and num_classes > 1:
                    mask_cls_np = np.array([j for j in range(num_classes) if j != 0])
                    mean_dice_loss = 1.0 - float(per_class_dice_epoch[mask_cls_np].mean())
                    mean_dice_score = float(per_class_dice_epoch[mask_cls_np].mean())
                else:
                    mean_dice_loss = 1.0 - float(per_class_dice_epoch.mean())
                    mean_dice_score = float(per_class_dice_epoch.mean())
            postfix = {"loss": avg_loss, "ce": avg_ce, "dice": mean_dice_loss}
            if sync_for_logging and device.type == "cuda":
                torch.cuda.synchronize()
            pbar.set_postfix(postfix)

    # finalize epoch-level per-class dice
    per_class_dice = compute_per_class_dice_from_inter_union(inter_acc, union_acc).detach().cpu().numpy()  # (C,)
    per_class_dice_loss = 1.0 - per_class_dice
    if state["exclude_background_from_dice"] and num_classes > 1:
        mask_idx = [i for i in range(num_classes) if i != 0]
        mean_dice_loss = float(per_class_dice_loss[mask_idx].mean())
        mean_dice_score = float(per_class_dice[mask_idx].mean())
    else:
        mean_dice_loss = float(per_class_dice_loss.mean())
        mean_dice_score = float(per_class_dice.mean())

    avg_loss = total_loss / max(1, n_batches)
    avg_ce = total_ce / max(1, n_batches)

    # Save epoch metrics into state
    state["last_train_loss"] = avg_loss
    state["last_train_ce"] = avg_ce
    state["last_train_dice_loss"] = mean_dice_loss
    state["last_train_dice"] = mean_dice_score
    state["last_train_per_class_dice"] = per_class_dice
    state["last_train_confusion"] = conf_acc

    return state


def validate(state: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    model: nn.Module = state["model"]
    device = state["device"]
    val_loader: DataLoader = state["dataloaders"]["val"]
    cfg = state["cfg"]

    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    n_batches = 0

    num_classes = int(cfg.get("n_classes", 4))
    inter_acc = torch.zeros(num_classes, device=device)
    union_acc = torch.zeros(num_classes, device=device)
    conf_acc = np.zeros((num_classes, num_classes), dtype=np.int64)

    ce_loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validate epoch {epoch}", leave=False)
        for i, batch in pbar:
            imgs, masks = batch
            imgs = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long).squeeze(1)

            logits = model(imgs)
            ce = ce_loss_fn(logits, masks)
            inter_b, union_b = accumulate_inter_union_from_logits(logits, masks)
            # detach even under no_grad for safety/consistency
            inter_acc += inter_b.detach()
            union_acc += union_b.detach()

            preds = logits.argmax(dim=1)
            conf_batch = confusion_matrix_from_preds_and_targets(preds, masks, num_classes=num_classes)
            conf_acc += conf_batch

            if state["exclude_background_from_dice"] and num_classes > 1:
                per_class_dice_b = compute_per_class_dice_from_inter_union(inter_b, union_b)
                mask_cls = torch.arange(num_classes, device=device) != 0
                dice_loss_b = 1.0 - per_class_dice_b[mask_cls].mean()
            else:
                per_class_dice_b = compute_per_class_dice_from_inter_union(inter_b, union_b)
                dice_loss_b = 1.0 - per_class_dice_b.mean()

            loss = ce + float(cfg.get("dice_weight", 1.0)) * dice_loss_b

            total_loss += float(loss.item())
            total_ce += float(ce.item())
            n_batches += 1

            pbar.set_postfix({"val_loss": total_loss / n_batches})

    per_class_dice = compute_per_class_dice_from_inter_union(inter_acc, union_acc).detach().cpu().numpy()
    per_class_dice_loss = 1.0 - per_class_dice
    if state["exclude_background_from_dice"] and num_classes > 1:
        mask_idx = [i for i in range(num_classes) if i != 0]
        mean_dice_loss = float(per_class_dice_loss[mask_idx].mean())
        mean_dice_score = float(per_class_dice[mask_idx].mean())
    else:
        mean_dice_loss = float(per_class_dice_loss.mean())
        mean_dice_score = float(per_class_dice.mean())

    avg_loss = total_loss / max(1, n_batches)
    avg_ce = total_ce / max(1, n_batches)

    state["last_val_loss"] = avg_loss
    state["last_val_ce"] = avg_ce
    state["last_val_dice_loss"] = mean_dice_loss
    state["last_val_dice"] = mean_dice_score
    state["last_val_per_class_dice"] = per_class_dice
    state["last_val_confusion"] = conf_acc

    # Scheduler step if using ReduceLROnPlateau
    if state.get("scheduler") is not None:
        try:
            state["scheduler"].step(avg_loss)
        except Exception:
            pass

    # Track best by validation loss
    if avg_loss < state.get("best_val_loss", float("inf")):
        state["best_val_loss"] = avg_loss
        state["best_epoch"] = epoch
        state["is_best"] = True
    else:
        state["is_best"] = False

    return state


# -------------------------
# Checkpoint / save / load
# -------------------------
def save(state: Dict[str, Any], path: str) -> None:
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
    map_location = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=map_location)
    return ckpt


# -------------------------
# High-level train loop
# -------------------------
def train(cfg: Dict[str, Any]) -> None:
    state = build(cfg)
    model = state["model"]
    optimizer = state["optimizer"]
    device = state["device"]
    class_names = state.get("class_names", ["BG", "RV", "MYO", "LV"])
    num_classes = int(cfg.get("n_classes", 4))
    exclude_bg = state.get("exclude_background_from_dice", True)

    max_epochs = int(cfg.get("max_epochs", 100))
    ckpt_dir = Path(cfg.get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_interval = int(cfg.get("ckpt_interval", 1))

    # Initialize TensorBoard writer and CSV
    log_dir = Path(cfg.get("log_dir", ckpt_dir / "logs")) if cfg.get("log_dir") is not None else (ckpt_dir / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(log_dir=str(log_dir))
        except Exception:
            writer = None
    state["writer"] = writer

    metrics_csv = ckpt_dir / "metrics.csv"
    state["metrics_csv"] = str(metrics_csv)

    # Build CSV header (basic metrics + per-class dice for train and val)
    header = [
        "epoch",
        "train_loss",
        "train_ce",
        "train_dice_loss",
        "train_dice_score",
        "val_loss",
        "val_ce",
        "val_dice_loss",
        "val_dice_score",
    ]
    # per-class train dice columns
    for cname in class_names:
        header.append(f"train_dice_{cname}")
    for cname in class_names:
        header.append(f"val_dice_{cname}")

    if not metrics_csv.exists():
        with open(metrics_csv, "w", newline="", encoding="utf-8") as cf:
            w = csv.writer(cf)
            w.writerow(header)

    # Optional resume
    resume_path = cfg.get("resume_from", None)
    if resume_path:
        ckpt = load(resume_path, device=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        state["start_epoch"] = int(ckpt.get("epoch", 0)) + 1
        state["best_val_loss"] = ckpt.get("best_val_loss", float("inf"))
        if "scaler_state" in ckpt and state.get("scaler") is not None:
            try:
                state["scaler"].load_state_dict(ckpt["scaler_state"])
            except Exception:
                pass
        print(f"[resume] loaded checkpoint {resume_path}, starting at epoch {state['start_epoch']}", flush=True)

    # Diagnostics
    print("Training start. Device:", device)
    print("Train samples:", len(state["dataloaders"]["train"].dataset), "Val samples:", len(state["dataloaders"]["val"].dataset))
    print("Config summary:", {k: cfg[k] for k in sorted(cfg) if k in ("batch_size", "lr", "max_epochs", "use_labeled_only", "sampler_type")})

    def _save_on_signal(signum, frame):
        try:
            ckpt_path = ckpt_dir / f"ckpt_epoch{state.get('epoch', 0):03d}_signal.pt"
            save(state, str(ckpt_path))
            print(f"[signal] saved checkpoint to {ckpt_path}", flush=True)
        except Exception as e:
            print(f"[signal] failed to save checkpoint: {e}", flush=True)
        finally:
            try:
                if state.get("writer") is not None:
                    state["writer"].close()
            except Exception:
                pass
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

            # Save checkpoints
            if (epoch + 1) % ckpt_interval == 0:
                ckpt_path = ckpt_dir / f"ckpt_epoch{epoch:03d}.pt"
                save(state, str(ckpt_path))
            if state.get("is_best", False):
                best_path = ckpt_dir / "ckpt_best.pt"
                save(state, str(best_path))

            t1 = time.time()

            # Gather metrics
            tr_loss = state.get("last_train_loss", float("nan"))
            tr_ce = state.get("last_train_ce", float("nan"))
            tr_dice_loss = state.get("last_train_dice_loss", float("nan"))
            tr_dice_score = state.get("last_train_dice", float("nan"))
            tr_per_class = state.get("last_train_per_class_dice", np.zeros(num_classes))

            val_loss = state.get("last_val_loss", float("nan"))
            val_ce = state.get("last_val_ce", float("nan"))
            val_dice_loss = state.get("last_val_dice_loss", float("nan"))
            val_dice_score = state.get("last_val_dice", float("nan"))
            val_per_class = state.get("last_val_per_class_dice", np.zeros(num_classes))

            # Print summary line
            print(
                f"Epoch {epoch:03d} finished in {t1 - t0:.1f}s  "
                f"train_loss={tr_loss:.4f} train_ce={tr_ce:.6f} train_dice_loss={tr_dice_loss:.6f} train_dice={tr_dice_score:.4f}  "
                f"val_loss={val_loss:.4f} val_ce={val_ce:.6f} val_dice_loss={val_dice_loss:.6f} val_dice={val_dice_score:.4f}"
            )

            # Save confusion matrices (npy)
            try:
                train_conf = state.get("last_train_confusion", None)
                val_conf = state.get("last_val_confusion", None)
                if train_conf is not None:
                    np.save(ckpt_dir / f"confusion_train_epoch{epoch:03d}.npy", train_conf)
                if val_conf is not None:
                    np.save(ckpt_dir / f"confusion_val_epoch{epoch:03d}.npy", val_conf)
            except Exception:
                pass

            # TensorBoard logging
            writer = state.get("writer")
            if writer is not None:
                try:
                    writer.add_scalar("train/loss", tr_loss, epoch)
                    writer.add_scalar("train/ce", tr_ce, epoch)
                    writer.add_scalar("train/dice_loss", tr_dice_loss, epoch)
                    writer.add_scalar("train/dice", tr_dice_score, epoch)

                    writer.add_scalar("val/loss", val_loss, epoch)
                    writer.add_scalar("val/ce", val_ce, epoch)
                    writer.add_scalar("val/dice_loss", val_dice_loss, epoch)
                    writer.add_scalar("val/dice", val_dice_score, epoch)

                    # per-class scalars
                    for ci in range(num_classes):
                        cname = class_names[ci] if ci < len(class_names) else f"class{ci}"
                        writer.add_scalar(f"train/dice_{cname}", float(tr_per_class[ci]), epoch)
                        writer.add_scalar(f"val/dice_{cname}", float(val_per_class[ci]), epoch)

                    # optionally log confusion matrix image as heatmap (if matplotlib available)
                    try:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
                        sns.heatmap(state.get("last_train_confusion", np.zeros((num_classes, num_classes))), ax=ax[0], cmap="viridis", annot=False)
                        ax[0].set_title("Conf Train (pred x true)")
                        sns.heatmap(state.get("last_val_confusion", np.zeros((num_classes, num_classes))), ax=ax[1], cmap="viridis", annot=False)
                        ax[1].set_title("Conf Val (pred x true)")
                        writer.add_figure("confusion_matrices", fig, epoch)
                        plt.close(fig)
                    except Exception:
                        # optional plotting libs may not exist; ignore
                        pass
                except Exception:
                    pass

            # Append CSV row
            try:
                with open(state["metrics_csv"], "a", newline="", encoding="utf-8") as cf:
                    w = csv.writer(cf)
                    row = [
                        epoch,
                        f"{tr_loss:.6f}",
                        f"{tr_ce:.6f}",
                        f"{tr_dice_loss:.6f}",
                        f"{tr_dice_score:.6f}",
                        f"{val_loss:.6f}",
                        f"{val_ce:.6f}",
                        f"{val_dice_loss:.6f}",
                        f"{val_dice_score:.6f}",
                    ]
                    # add per-class train dice
                    for v in tr_per_class:
                        row.append(f"{float(v):.6f}")
                    for v in val_per_class:
                        row.append(f"{float(v):.6f}")
                    w.writerow(row)
            except Exception:
                pass

        # close writer if present
        try:
            if state.get("writer") is not None:
                state["writer"].close()
        except Exception:
            pass

    except KeyboardInterrupt:
        ckpt_path = ckpt_dir / f"ckpt_epoch{state.get('epoch', 0):03d}_interrupt.pt"
        save(state, str(ckpt_path))
        print(f"[interrupt] saved checkpoint to {ckpt_path}", flush=True)
        try:
            if state.get("writer") is not None:
                state["writer"].close()
        except Exception:
            pass
        raise

    print("Training finished. Best val loss:", state.get("best_val_loss"))


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train segmentation model (supervised) with publication-grade metrics")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (optional). If omitted, defaults are used.")
    args = parser.parse_args()

    # Default config
    cfg: Dict[str, Any] = {
        "data_root": "preprocessed",
        "manifest_path": "preprocessed/mask_manifest.json",
        "split_manifest_path": "preprocessed/split_manifest_labeled.json",
        "use_labeled_only": True,
        "sampler_type": "none",
        "labeled_weight": 5.0,
        "batch_size": 4,
        "num_workers": 4,
        "pin_memory": False,
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
        # AMP and logging options
        "use_amp": True,
        "log_every": 10,
        "sync_for_logging": False,
        "cudnn_benchmark": False,
        # Optional resume path (set in JSON or CLI)
        "resume_from": None,
        # Exclude samples with missing masks
        "exclude_missing_masks": False,
        # Per-class names (index 0..n-1)
        "class_names": ["BG", "RV", "MYO", "LV"],
        # Exclude background when averaging Dice (recommended)
        "exclude_background_from_dice": True,
        # Logging: where to write TensorBoard logs (defaults to <ckpt_dir>/logs)
        "log_dir": None,
    }

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise RuntimeError(f"Config file not found: {cfg_path}")
        cfg_from_file = json.load(open(cfg_path, "r", encoding="utf-8"))
        cfg.update(cfg_from_file)

    train(cfg)
