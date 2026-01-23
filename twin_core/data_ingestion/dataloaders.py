# dataloaders.py
from pathlib import Path
import json
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import SimpleITK as sitk
from twin_core.data_ingestion.dataset_wrapper import CardiacDatasetWithMeta

def load_4d_image(filepath):
    img = sitk.ReadImage(str(filepath))
    arr = sitk.GetArrayFromImage(img)  # (T, Z, Y, X)
    spacing = img.GetSpacing()         # (X, Y, Z, T)
    return arr, spacing


def _load_manifest(manifest_path: Optional[Path]) -> Dict[str, dict]:
    if manifest_path is None:
        return {}
    try:
        return json.load(open(manifest_path, "r", encoding="utf-8"))
    except Exception:
        return {}

def _indices_for_patients(dataset: CardiacDatasetWithMeta, patients: List[str]) -> List[int]:
    """Return dataset indices that belong to any patient in patients list."""
    patient_set = set(patients)
    return [i for i, s in enumerate(dataset.samples) if s[0] in patient_set]

def _labeled_indices_from_manifest(dataset: CardiacDatasetWithMeta, manifest: Dict[str, dict]) -> List[int]:
    """Fast path: use manifest patient-level real_masks>0 to select indices."""
    labeled_patients = [p for p, info in manifest.items() if info.get("real_masks", 0) > 0]
    return _indices_for_patients(dataset, labeled_patients)

def _labeled_indices_by_check(dataset: CardiacDatasetWithMeta) -> List[int]:
    """Slow path: check each sample's mask file for nonzero content."""
    labeled = []
    for i in range(len(dataset)):
        # check_nonzero True loads .npy and counts nonzero
        if dataset.has_mask(i, check_nonzero=True):
            labeled.append(i)
    return labeled

def get_dataloaders(cfg: Dict) -> Dict[str, DataLoader]:
    """
    Build train and val DataLoaders.

    cfg keys used:
      - data_root: Path or str to preprocessed root (contains train/ and val/)
      - manifest_path: optional path to mask_manifest.json
      - split_manifest_path: optional path to split_manifest.json (not required)
      - use_labeled_only: bool
      - sampler_type: 'none' | 'weighted'
      - labeled_weight: float
      - batch_size, num_workers, pin_memory
      - use_manifest_for_label_filter: bool (default True)
      - prefer_ed_es, exclude_missing_masks, one_hot, n_classes, pad_multiple (passed to dataset)
    """
    data_root = Path(cfg.get("data_root", "preprocessed"))
    pre_train = data_root / "train"
    pre_val = data_root / "val"

    if not pre_train.exists() or not pre_val.exists():
        raise RuntimeError(f"Expected preprocessed train/ and val/ under {data_root}")

    manifest = _load_manifest(Path(cfg.get("manifest_path")) if cfg.get("manifest_path") else None)
    use_manifest_for_label_filter = bool(cfg.get("use_manifest_for_label_filter", True))

    ds_kwargs = {
        "prefer_ed_es": cfg.get("prefer_ed_es", False),
        "metadata_index": None,
        "one_hot": cfg.get("one_hot", False),
        "n_classes": cfg.get("n_classes", 4),
        "exclude_missing_masks": cfg.get("exclude_missing_masks", False),
        "pad_multiple": cfg.get("pad_multiple", 16),
    }

    # Instantiate datasets (they scan their respective folders)
    train_ds = CardiacDatasetWithMeta(pre_train, **ds_kwargs)
    val_ds = CardiacDatasetWithMeta(pre_val, **ds_kwargs)

    # Diagnostics: global manifest counts if available
    total_saved = sum(v.get("saved_masks", 0) for v in manifest.values()) if manifest else None
    total_real = sum(v.get("real_masks", 0) for v in manifest.values()) if manifest else None
    if total_saved is not None and total_real is not None:
        frac = total_real / total_saved if total_saved > 0 else 0.0
        print(f"[dataloaders] manifest totals: saved_masks={total_saved}, real_masks={total_real}, fraction_labeled={frac:.4f}")

    # Build train loader according to config
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = bool(cfg.get("pin_memory", True))

    use_labeled_only = bool(cfg.get("use_labeled_only", False))
    sampler_type = cfg.get("sampler_type", "none")
    labeled_weight = float(cfg.get("labeled_weight", 5.0))

    # Determine labeled indices for train dataset
    labeled_indices: List[int] = []
    if use_labeled_only or sampler_type == "weighted":
        if use_manifest_for_label_filter and manifest:
            # Fast patient-level selection using manifest
            labeled_indices = _labeled_indices_from_manifest(train_ds, manifest)
            # If we need per-sample nonzero guarantee, optionally refine by checking files
            if cfg.get("require_nonzero_mask_files", False):
                labeled_indices = [i for i in labeled_indices if train_ds.has_mask(i, check_nonzero=True)]
        else:
            # Slow but robust: check each sample's mask file for nonzero content
            labeled_indices = _labeled_indices_by_check(train_ds)

    # Build train_loader
    if use_labeled_only:
        if not labeled_indices:
            raise RuntimeError("use_labeled_only=True but no labeled samples found in train dataset.")
        train_subset = Subset(train_ds, labeled_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    elif sampler_type == "weighted":
        # Build weights: labeled samples get labeled_weight, others get 1.0
        weights = []
        for i in range(len(train_ds)):
            if i in set(labeled_indices):
                weights.append(float(labeled_weight))
            else:
                # fallback to quick existence check (mask_path not None) to avoid loading files
                weights.append(1.0 if train_ds.has_mask(i, check_nonzero=False) else 1.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    # Build val_loader (always deterministic, do not include unlabeled-only val)
    # Optionally filter val to only patients with real masks using manifest
    val_use_labeled_only = bool(cfg.get("val_use_labeled_only", True))
    if val_use_labeled_only and manifest:
        # find val patients with real masks
        val_patients_with_labels = [p for p, info in manifest.items() if info.get("real_masks", 0) > 0 and (data_root / "val" / p).exists()]
        if val_patients_with_labels:
            val_indices = _indices_for_patients(val_ds, val_patients_with_labels)
            if not val_indices:
                raise RuntimeError("No labeled samples found in val dataset after filtering by manifest.")
            val_loader = DataLoader(Subset(val_ds, val_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        else:
            # fallback to full val dataset
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    else:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Print quick composition diagnostics for first few batches
    def _print_batch_composition(loader, name: str, n_batches: int = 3):
        it = iter(loader)
        for i in range(n_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            # batch expected as (images, masks)
            masks = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
            if masks is None:
                print(f"[dataloaders] {name} batch {i}: cannot inspect masks (unexpected batch format)")
                continue
            # masks shape: (B,1,H,W) or (B,C,H,W)
            b = masks.size(0)
            labeled_in_batch = (masks.view(b, -1).sum(dim=1) > 0).sum().item()
            print(f"[dataloaders] {name} batch {i}: labeled_in_batch={labeled_in_batch}/{b}")

    print("[dataloaders] dataset sizes: train_samples=", len(train_ds), " val_samples=", len(val_ds))
    _print_batch_composition(train_loader, "train", n_batches=3)
    _print_batch_composition(val_loader, "val", n_batches=2)

    return {"train": train_loader, "val": val_loader}
