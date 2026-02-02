#!/usr/bin/env python3
"""
Training pipeline with:
 - per-class Dice metrics and confusion matrices
 - TensorBoard + CSV logging
 - robust checkpointing and AMP support
 - k-fold cross-validation stratified by patient Group (from mask_manifest.json)
 - early stopping (monitor val_loss) and ReduceLROnPlateau scheduler support

Usage: provide --config path/to/config.json (same config pattern as before).
"""
from pathlib import Path
import json
import time
import signal
import sys
import csv
import shutil
from typing import Dict, Any, Optional, Sequence, List, Tuple
import random
import math

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
    if target.dim() == 4:
        target = target.squeeze(1)
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
    target_onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * target_onehot).sum(dims)
    union_term = probs.sum(dims) + target_onehot.sum(dims)
    return inter, union_term


def compute_per_class_dice_from_inter_union(inter: torch.Tensor, union_term: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dice = (2.0 * inter + eps) / (union_term + eps)
    return dice


def confusion_matrix_from_preds_and_targets(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:
    p = preds.view(-1).cpu().to(torch.int64)
    t = targets.view(-1).cpu().to(torch.int64)
    p = torch.clamp(p, 0, num_classes - 1)
    t = torch.clamp(t, 0, num_classes - 1)
    idx = p * num_classes + t
    counts = torch.bincount(idx, minlength=num_classes * num_classes)
    conf = counts.reshape(num_classes, num_classes).numpy().astype(np.int64)
    return conf


# -------------------------
# Utility: write split manifest
# -------------------------
def write_split_manifest(train_list: List[str], val_list: List[str], out_path: Path, extra: Dict[str, Any] = None) -> None:
    payload = {
        "train": list(train_list),
        "val": list(val_list),
    }
    if extra:
        payload.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -------------------------
# Build k-folds stratified by Group mapping
# -------------------------
def make_stratified_kfolds_from_manifest(manifest_path: Path, pool: Optional[List[str]], k: int, seed: int) -> List[List[str]]:
    """
    Reads mask_manifest.json and returns k folds (list of patient name lists),
    stratified by Group. If 'pool' is provided, limit to that patient subset for k-fold (e.g., pre-split train list).
    """
    mm = json.load(open(manifest_path, "r", encoding="utf-8"))
    # mm keys are patient IDs (folder names)
    # build group -> list mapping restricted to pool if provided
    groups: Dict[str, List[str]] = {}
    pool_set = None
    if pool is not None:
        pool_set = set(pool)
    for patient, info in mm.items():
        if pool_set is not None and patient not in pool_set:
            continue
        grp = info.get("Group") or "UNKNOWN"
        groups.setdefault(grp, []).append(patient)

    rng = random.Random(seed)
    # initialize empty folds
    folds: List[List[str]] = [[] for _ in range(k)]
    # distribute each group's patients round-robin into folds after shuffling
    for grp, patients in groups.items():
        rng.shuffle(patients)
        for idx, p in enumerate(patients):
            folds[idx % k].append(p)

    # final shuffle within each fold for randomness (but deterministic from seed)
    for fold in folds:
        rng.shuffle(fold)
    return folds


# -------------------------
# Pipeline interface (build model, dataloaders, optimizer...)
# -------------------------
def build(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

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

    if len(train_loader.dataset) == 0:
        raise RuntimeError("Train dataset is empty. Check split_manifest and mask_manifest.")
    if cfg.get("require_val_labels", True) and len(val_loader.dataset) == 0:
        raise RuntimeError("Validation dataset is empty (no labeled samples). Check split_manifest and mask_manifest.")

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg.get("plateau_factor", 0.5), patience=int(cfg.get("plateau_patience", 5))) if cfg.get("use_plateau_scheduler", True) else None

    # AMP support
    use_amp = bool(cfg.get("use_amp", False))
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

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
        "writer": None,
        "metrics_csv": None,
        "class_names": list(class_names),
        "exclude_background_from_dice": exclude_background_from_dice,
    }
    return state


# -------------------------
# Training / validation epoch implementations
# (unchanged logic, but moved into functions used per-fold)
# -------------------------
def train_epoch(state: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    model: nn.Module = state["model"]
    optimizer: optim.Optimizer = state["optimizer"]
    device = state["device"]
    train_loader: DataLoader = state["dataloaders"]["train"]
    cfg = state["cfg"]

    model.train()
    total_loss = 0.0
    total_ce = 0.0
    n_batches = 0

    num_classes = int(cfg.get("n_classes", 4))
    inter_acc = torch.zeros(num_classes, device=device)
    union_acc = torch.zeros(num_classes, device=device)
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
                logits = state["model"](imgs)
                ce = ce_loss_fn(logits, masks)
                inter_b, union_b = accumulate_inter_union_from_logits(logits, masks)
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
            logits = state["model"](imgs)
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

        inter_acc += inter_b.detach()
        union_acc += union_b.detach()

        preds = logits.argmax(dim=1)
        conf_batch = confusion_matrix_from_preds_and_targets(preds, masks, num_classes=num_classes)
        conf_acc += conf_batch

        total_loss += float(loss.detach().item())
        total_ce += float(ce.detach().item())
        n_batches += 1

        if (i + 1) % log_every == 0 or (i + 1) == len(train_loader):
            avg_loss = total_loss / n_batches
            avg_ce = total_ce / n_batches
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

            logits = state["model"](imgs)
            ce = ce_loss_fn(logits, masks)
            inter_b, union_b = accumulate_inter_union_from_logits(logits, masks)
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

    if state.get("scheduler") is not None:
        try:
            state["scheduler"].step(avg_loss)
        except Exception:
            pass

    if avg_loss < state.get("best_val_loss", float("inf")) - cfg.get("early_stopping_min_delta", 1e-6):
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
# High-level train loop with k-fold and early stopping
# -------------------------
def train(cfg: Dict[str, Any]) -> None:
    # reproducibility seeds
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_root = Path(cfg.get("data_root", "preprocessed"))
    manifest_path = Path(cfg.get("manifest_path", data_root / "mask_manifest.json"))
    if not manifest_path.exists():
        raise RuntimeError(f"mask_manifest not found at {manifest_path}")

    # read existing split_manifest if present (to detect pre-split mode)
    split_manifest_path_cfg = cfg.get("split_manifest_path", str(data_root / "split_manifest_labeled.json"))
    split_manifest_path = Path(split_manifest_path_cfg)
    split_manifest = None
    if split_manifest_path.exists():
        try:
            split_manifest = json.load(open(split_manifest_path, "r", encoding="utf-8"))
        except Exception:
            split_manifest = None

    # choose pool for k-fold:
    # if split_manifest provided and contains "train", perform k-fold inside that train list (typical pre-split case)
    pool_patients: Optional[List[str]] = None
    if split_manifest and isinstance(split_manifest, dict) and "train" in split_manifest:
        pool_patients = list(split_manifest["train"])
        print(f"[kfold] Detected existing split_manifest: performing k-fold on provided 'train' pool of size {len(pool_patients)}")
    else:
        # full manifest patient list
        pool_patients = sorted(list(json.load(open(manifest_path, "r", encoding="utf-8")).keys()))
        print(f"[kfold] No train list in split_manifest. Performing k-fold over all patients in manifest (count={len(pool_patients)})")

    k_folds = int(cfg.get("k_folds", 1))
    if k_folds < 1:
        k_folds = 1

    # If only one fold requested, fallback to single-run using existing split_manifest (or default)
    if k_folds == 1:
        folds = [pool_patients]
    else:
        folds = make_stratified_kfolds_from_manifest(manifest_path, pool_patients, k_folds, seed)

    # Directory for top-level checkpoints/logs
    top_ckpt_dir = Path(cfg.get("ckpt_dir", "checkpoints"))
    top_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # early stopping options
    use_early_stop = bool(cfg.get("early_stopping", True))
    early_patience = int(cfg.get("early_stopping_patience", 10))
    early_min_delta = float(cfg.get("early_stopping_min_delta", 1e-4))

    # accumulate fold results
    fold_summaries: List[Dict[str, Any]] = []

    # If split_manifest originally had 'val' (two-input mode) and user requested k-fold,
    # we keep that val/test as separate holdout if desired. Here we implement k-fold on train pool only,
    # and the val produced by the fold is the held-out fold.
    for fold_idx in range(len(folds) if k_folds > 1 else 1):
        if k_folds == 1:
            # single-run: use existing split manifest if available, otherwise rely on get_dataloaders default split files
            if split_manifest and "train" in split_manifest and "val" in split_manifest:
                train_list = list(split_manifest["train"])
                val_list = list(split_manifest["val"])
            else:
                # no explicit split; rely on split_manifest.json in preprocessed/ (build will fallback)
                train_list = pool_patients
                val_list = []
        else:
            # build train/val for this fold (val = folds[fold_idx], train = union of others)
            val_list = list(folds[fold_idx])
            train_list = []
            for j in range(len(folds)):
                if j == fold_idx:
                    continue
                train_list.extend(folds[j])

        # prepare fold-specific ckpt dir and split_manifest file
        fold_ckpt_dir = top_ckpt_dir / f"fold_{fold_idx:02d}"
        if fold_ckpt_dir.exists() and cfg.get("overwrite_ckpt_dir", False):
            shutil.rmtree(fold_ckpt_dir)
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

        split_for_fold_path = fold_ckpt_dir / "split_manifest_kfold.json"
        extra = {"train_ratio": len(train_list) / max(1, (len(train_list) + len(val_list))), "seed": seed, "k_fold_index": fold_idx, "k_folds": k_folds}
        write_split_manifest(train_list, val_list, split_for_fold_path, extra=extra)

        # update cfg for this fold (copy to avoid mutating original)
        cfg_fold = dict(cfg)
        cfg_fold["split_manifest_path"] = str(split_for_fold_path)
        cfg_fold["ckpt_dir"] = str(fold_ckpt_dir)
        # logs saved under fold-specific ckpt dir automatically by build/train
        # set early stopping parameters into cfg_fold for check in validate
        cfg_fold["early_stopping_min_delta"] = early_min_delta

        # Build state for this fold
        state = build(cfg_fold)
        model = state["model"]
        optimizer = state["optimizer"]
        device = state["device"]
        class_names = state.get("class_names", ["BG", "RV", "MYO", "LV"])
        num_classes = int(cfg_fold.get("n_classes", 4))

        max_epochs = int(cfg_fold.get("max_epochs", 100))
        ckpt_interval = int(cfg_fold.get("ckpt_interval", 1))

        # Prepare logging: writer + CSV per fold
        log_dir = Path(cfg_fold.get("log_dir")) if cfg_fold.get("log_dir") is not None else (Path(state["cfg"]["ckpt_dir"]) / "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = None
        if SummaryWriter is not None:
            try:
                writer = SummaryWriter(log_dir=str(log_dir))
            except Exception:
                writer = None
        state["writer"] = writer

        metrics_csv = Path(state["cfg"]["ckpt_dir"]) / "metrics.csv"
        state["metrics_csv"] = str(metrics_csv)
        if not metrics_csv.exists():
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
            for cname in class_names:
                header.append(f"train_dice_{cname}")
            for cname in class_names:
                header.append(f"val_dice_{cname}")
            with open(metrics_csv, "w", newline="", encoding="utf-8") as cf:
                w = csv.writer(cf)
                w.writerow(header)

        # Optional resume: if resume_from points to a checkpoint and user wants to resume per-fold, user can supply resume path.
        resume_path = cfg_fold.get("resume_from", None)
        if resume_path:
            ckpt = load(resume_path, device=device)
            try:
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
            except Exception as e:
                print(f"[resume] failed to load checkpoint cleanly: {e}", flush=True)

        # diagnostics
        print(f"Fold {fold_idx:02d}/{(k_folds or 1)-1:02d} Training start. Device: {device}")
        print("Train samples:", len(state["dataloaders"]["train"].dataset), "Val samples:", len(state["dataloaders"]["val"].dataset))
        print("Fold config summary:", {k: cfg_fold[k] for k in sorted(cfg_fold) if k in ("batch_size", "lr", "max_epochs", "use_labeled_only", "sampler_type")})

        # signal handler to save
        def _save_on_signal(signum, frame):
            try:
                ckpt_path = Path(state["cfg"]["ckpt_dir"]) / f"ckpt_epoch{state.get('epoch', 0):03d}_signal.pt"
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

        # Early stopping trackers
        best_fold_val = float("inf")
        epochs_since_improve = 0
        fold_best_epoch = -1

        try:
            for epoch in range(state.get("start_epoch", 0), max_epochs):
                state["epoch"] = epoch
                t0 = time.time()
                state = train_epoch(state, epoch)
                state = validate(state, epoch)

                # Save checkpoints
                if (epoch + 1) % ckpt_interval == 0:
                    ckpt_path = Path(state["cfg"]["ckpt_dir"]) / f"ckpt_epoch{epoch:03d}.pt"
                    save(state, str(ckpt_path))
                if state.get("is_best", False):
                    best_path = Path(state["cfg"]["ckpt_dir"]) / "ckpt_best.pt"
                    save(state, str(best_path))

                t1 = time.time()
                # gather metrics
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

                print(
                    f"[fold {fold_idx:02d}] Epoch {epoch:03d} finished in {t1 - t0:.1f}s  "
                    f"train_loss={tr_loss:.4f} train_ce={tr_ce:.6f} train_dice_loss={tr_dice_loss:.6f} train_dice={tr_dice_score:.4f}  "
                    f"val_loss={val_loss:.4f} val_ce={val_ce:.6f} val_dice_loss={val_dice_loss:.6f} val_dice={val_dice_score:.4f}"
                )

                # Save confusion matrices
                try:
                    train_conf = state.get("last_train_confusion", None)
                    val_conf = state.get("last_val_confusion", None)
                    if train_conf is not None:
                        np.save(Path(state["cfg"]["ckpt_dir"]) / f"confusion_train_epoch{epoch:03d}.npy", train_conf)
                    if val_conf is not None:
                        np.save(Path(state["cfg"]["ckpt_dir"]) / f"confusion_val_epoch{epoch:03d}.npy", val_conf)
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

                        for ci in range(num_classes):
                            cname = class_names[ci] if ci < len(class_names) else f"class{ci}"
                            writer.add_scalar(f"train/dice_{cname}", float(tr_per_class[ci]), epoch)
                            writer.add_scalar(f"val/dice_{cname}", float(val_per_class[ci]), epoch)

                        # optionally log confusion matrix as image
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
                        for v in tr_per_class:
                            row.append(f"{float(v):.6f}")
                        for v in val_per_class:
                            row.append(f"{float(v):.6f}")
                        w.writerow(row)
                except Exception:
                    pass

                # Early stopping logic (monitor val_loss)
                if use_early_stop:
                    if val_loss + early_min_delta < best_fold_val:
                        best_fold_val = val_loss
                        epochs_since_improve = 0
                        fold_best_epoch = epoch
                    else:
                        epochs_since_improve += 1

                    if epochs_since_improve >= early_patience:
                        print(f"[fold {fold_idx:02d}] Early stopping after {epochs_since_improve} epochs without improvement (patience={early_patience}).")
                        break

            # close writer if present
            try:
                if state.get("writer") is not None:
                    state["writer"].close()
            except Exception:
                pass

        except KeyboardInterrupt:
            ckpt_path = Path(state["cfg"]["ckpt_dir"]) / f"ckpt_epoch{state.get('epoch', 0):03d}_interrupt.pt"
            save(state, str(ckpt_path))
            print(f"[interrupt] saved checkpoint to {ckpt_path}", flush=True)
            try:
                if state.get("writer") is not None:
                    state["writer"].close()
            except Exception:
                pass
            raise

        # record fold summary
        fold_summary = {
            "fold_idx": fold_idx,
            "train_count": len(train_list),
            "val_count": len(val_list),
            "best_val_loss": float(state.get("best_val_loss", float("nan"))),
            "best_epoch": int(state.get("best_epoch", -1)),
            "last_val_dice": float(state.get("last_val_dice", float("nan"))),
            "last_train_dice": float(state.get("last_train_dice", float("nan"))),
            "ckpt_dir": str(state["cfg"]["ckpt_dir"]),
        }
        fold_summaries.append(fold_summary)

    # after folds
    try:
        # aggregate fold results
        avg_best_val = float(np.mean([fs["best_val_loss"] for fs in fold_summaries]))
        avg_last_val_dice = float(np.mean([fs["last_val_dice"] for fs in fold_summaries]))
        print("K-Fold training finished.")
        print("Fold summaries:")
        for fs in fold_summaries:
            print(fs)
        print(f"Average best val_loss across folds: {avg_best_val:.6f}")
        print(f"Average last val_dice across folds: {avg_last_val_dice:.6f}")
    except Exception:
        pass


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train segmentation model with k-fold CV, early stopping and publication-grade metrics")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (optional). If omitted, defaults are used.")
    args = parser.parse_args()

    # Default config (add k-fold and early stopping defaults)
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
        # K-fold options
        "k_folds": 1,  # number of folds, set >1 to run k-fold CV
        "seed": 42,
        "kfold_on_train_only": True,  # if split_manifest has a train list, use that pool for k-fold
        # Early stopping (monitor val_loss)
        "early_stopping": True,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 1e-4,
        # ReduceLROnPlateau options
        "use_plateau_scheduler": True,
        "plateau_patience": 5,
        "plateau_factor": 0.5,
        # Whether to allow overwriting fold ckpt dirs
        "overwrite_ckpt_dir": False,
    }

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise RuntimeError(f"Config file not found: {cfg_path}")
        cfg_from_file = json.load(open(cfg_path, "r", encoding="utf-8"))
        cfg.update(cfg_from_file)

    train(cfg)
