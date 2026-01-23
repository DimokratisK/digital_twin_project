#!/usr/bin/env python3
"""
Refactored preprocessing pipeline for cine-like 4D cardiac NIfTI datasets.

Changes from original:
- Writes an additional labeled-only split manifest: split_manifest_labeled.json
- Extends mask_manifest entries with labeled_frame_indices and per_frame_nonzero
- Adds optional end-to-end consistency checks (original vs saved nonzero counts)
- Adds CLI flags to control writing labeled split and consistency checks
- Emits startup and summary diagnostics
- Keeps original outputs and behavior unchanged by default
"""
from pathlib import Path
import argparse
import json
import random
import re
import numpy as np
import nibabel as nib
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List, Any

from .paths import dataset_paths, ensure_dir

# -------------------------
# Defaults (can be overridden via CLI)
# -------------------------
DEFAULT_CROP = 8
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_EPS = 1e-8
DEFAULT_SEED = 42

# -------------------------
# Normalization helpers
# -------------------------
EPS = DEFAULT_EPS


def normalize(volume: np.ndarray) -> np.ndarray:
    mu = volume.mean()
    std = volume.std() + EPS
    return (volume - mu) / std


def standardize(volume: np.ndarray) -> np.ndarray:
    vmin, vmax = volume.min(), volume.max()
    return (volume - vmin) / (vmax - vmin + EPS)


# -------------------------
# Helpers: canonical reorder to (T, Z, Y, X)
# -------------------------
def reorder_to_tzyx(img: np.ndarray, header: Any) -> np.ndarray:
    """
    Convert a nibabel-loaded array to canonical (T, Z, Y, X).
    Handles common layouts: (X,Y,Z,T), (T,X,Y,Z), (X,Y,Z) and many odd variations robustly.
    Returns a numpy array with ndim == 4 and axis 0 = time (T). For 3D inputs returns (1,Z,Y,X).
    """
    # If already 3D: assume (X,Y,Z) -> convert to (1, Z, Y, X)
    if img.ndim == 3:
        return np.transpose(img, (2, 1, 0))[None]

    # If not 4D now, fail
    if img.ndim != 4:
        raise RuntimeError(f"Unsupported image ndim: {img.ndim}")

    # Try to read header dims in a safe way
    try:
        hdr_dim = header.get('dim') if hasattr(header, 'get') else header['dim']
        hdr_x, hdr_y, hdr_z, hdr_t = int(hdr_dim[1]), int(hdr_dim[2]), int(hdr_dim[3]), int(hdr_dim[4])
    except Exception:
        # fallback: use shape heuristics
        shape = img.shape
        hdr_x, hdr_y, hdr_z, hdr_t = shape[0], shape[1], shape[2], shape[3]

    shape = img.shape

    # Common case: (X,Y,Z,T)
    if shape == (hdr_x, hdr_y, hdr_z, hdr_t):
        arr = np.moveaxis(img, -1, 0)   # (T,X,Y,Z)
        arr = np.transpose(arr, (0, 3, 2, 1))  # (T,Z,Y,X)
        return arr

    # Common case: (T, X, Y, Z)
    if shape[0] == hdr_t and shape[1:] == (hdr_x, hdr_y, hdr_z):
        arr = np.transpose(img, (0, 3, 2, 1))  # (T,Z,Y,X)
        return arr

    # Generic: find axis with length hdr_t and move it to front
    axis_idx_for_t = None
    for idx, length in enumerate(shape):
        if length == hdr_t:
            axis_idx_for_t = idx
            break

    if axis_idx_for_t is None:
        axis_idx_for_t = len(shape) - 1

    arr = np.moveaxis(img, axis_idx_for_t, 0)  # bring time to axis 0 -> (T,...)

    # Map remaining axes to (Z, Y, X) by matching to hdr_x/hdr_y/hdr_z
    rem = arr.shape[1:]
    rem_indices: Dict[str, int] = {}
    for i, dim_len in enumerate(rem):
        if dim_len == hdr_x and 'x' not in rem_indices:
            rem_indices['x'] = i
        elif dim_len == hdr_y and 'y' not in rem_indices:
            rem_indices['y'] = i
        elif dim_len == hdr_z and 'z' not in rem_indices:
            rem_indices['z'] = i

    if set(rem_indices.keys()) == {'x', 'y', 'z'}:
        z_idx = rem_indices['z'] + 1
        y_idx = rem_indices['y'] + 1
        x_idx = rem_indices['x'] + 1
        arr = np.transpose(arr, (0, z_idx, y_idx, x_idx))
        return arr

    # If rem already equals (hdr_z, hdr_y, hdr_x), assume correct
    if rem == (hdr_z, hdr_y, hdr_x):
        return arr

    # If rem equals (hdr_x, hdr_y, hdr_z), transpose to (Z,Y,X)
    if rem == (hdr_x, hdr_y, hdr_z):
        arr = np.transpose(arr, (0, 3, 2, 1))
        return arr

    # Fallback: try last three reversed
    try:
        arr = np.transpose(arr, (0, 3, 2, 1))
    except Exception:
        pass
    return arr


# -------------------------
# Helpers: parse info.cfg
# -------------------------
def parse_info_cfg(patient_dir: Path) -> Dict[str, Optional[str]]:
    """
    Parse a simple key: value info.cfg file. Returns a dict with keys as strings.
    Missing keys map to None.
    """
    info_file = patient_dir / "info.cfg"
    info = {
        "ED": None,
        "ES": None,
        "Group": None,
        "Height": None,
        "Weight": None,
        "NbFrame": None
    }
    if not info_file.exists():
        return info

    try:
        with open(info_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = [s.strip() for s in line.split(":", 1)]
                if key in info:
                    info[key] = val
    except Exception:
        return info

    return info


def info_values_as_ints(info: Dict[str, Optional[str]]) -> Dict[str, Optional[Any]]:
    """
    Convert certain fields to numbers when possible and keep others.
    """
    out: Dict[str, Optional[Any]] = {}
    for k in ("ED", "ES", "NbFrame"):
        v = info.get(k)
        try:
            out[k] = int(v) if v is not None else None
        except Exception:
            out[k] = None
    out["Group"] = info.get("Group")
    out["Height"] = float(info["Height"]) if info.get("Height") else None
    out["Weight"] = float(info["Weight"]) if info.get("Weight") else None
    return out


# -------------------------
# Parse per-frame filename to obtain frame index
# -------------------------
# Accepts "frame01", "frame-01", "f01" style variants
_FRAME_RE = re.compile(r"(?:[_\-]|^)(?:frame|f)0*([0-9]+)", flags=re.IGNORECASE)


def _parse_frame_index_from_name(fname: str) -> Optional[int]:
    """
    Returns 0-indexed frame index or None.
    Example matches: patient001_frame01_gt.nii.gz -> 0
                     patient001-frame12-gt.nii.gz -> 11
    """
    m = _FRAME_RE.search(fname)
    if not m:
        return None
    try:
        idx = int(m.group(1))
        return max(0, idx - 1)  # convert 1-indexed to 0-indexed
    except Exception:
        return None


# -------------------------
# Load per-frame GT files into mapping t -> (Z,Y,X) arrays
# -------------------------
def load_per_frame_gt_map(patient_dir: Path) -> Tuple[Dict[int, np.ndarray], List[str]]:
    """
    Returns (mapping, file_list):
      - mapping: {t_index: ndarray(Z,Y,X)}
      - file_list: list of filenames actually considered (for metadata)
    """
    frame_files = sorted(list(patient_dir.glob("*_frame*_gt.nii*")))
    mapping: Dict[int, np.ndarray] = {}
    used_files: List[str] = []
    for p in frame_files:
        t_idx = _parse_frame_index_from_name(p.name)
        if t_idx is None:
            print(f"Warning: could not parse frame index from {p.name}; skipping")
            continue
        try:
            nii = nib.load(str(p))
            arr = nii.get_fdata()
            arr = reorder_to_tzyx(arr, nii.header)  # often becomes (1, Z, Y, X)
            # Normalize arr to (Z, Y, X)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr3 = arr[0]  # (Z, Y, X)
            elif arr.ndim == 3:
                arr3 = arr  # (Z, Y, X)
            elif arr.ndim == 2:
                arr3 = arr[None, :, :]
            else:
                arr_s = np.squeeze(arr)
                if arr_s.ndim == 3:
                    arr3 = arr_s
                elif arr_s.ndim == 2:
                    arr3 = arr_s[None, :, :]
                else:
                    raise RuntimeError(f"Unexpected per-frame GT shape {arr.shape} for {p}")
            mapping[t_idx] = arr3.astype(np.uint8)
            used_files.append(p.name)
        except Exception as e:
            print(f"Warning: failed to load per-frame GT {p.name}: {e}; skipping")
            continue

    return mapping, used_files


# -------------------------
# Compose full label array aligned with image T,Z,Y,X
# -------------------------
def build_label_array_from_sources(
    image_shape: Tuple[int, int, int, int],
    patient_dir: Path
) -> Tuple[Optional[np.ndarray], str, List[str]]:
    """
    Try to load either:
      1) a single 4D GT file (patient_4d_gt.nii*), or
      2) per-frame GT files (patient_frameXX_gt.nii*).
    Returns:
      - label_array: ndarray (T,Z,Y,X) or None if no labels
      - label_source: "4d", "per_frame", or "none"
      - label_files_used: list of filenames used
    """
    T_img, Z_img, Y_img, X_img = image_shape
    # 1) Try single 4D GT
    gt4_path = next(patient_dir.glob("*_4d_gt.nii*"), None)
    if gt4_path is not None:
        try:
            nii = nib.load(str(gt4_path))
            lbl = nii.get_fdata().astype(np.uint8)
            lbl = reorder_to_tzyx(lbl, nii.header)  # (T', Z', Y', X') usually
            if lbl.ndim == 3:
                lbl = lbl[None]
            if lbl.ndim != 4:
                raise RuntimeError(f"Unexpected GT 4D shape {lbl.shape} for {gt4_path}")
            T_lbl, Z_lbl, Y_lbl, X_lbl = lbl.shape
            res = np.zeros((T_img, Z_img, Y_img, X_img), dtype=np.uint8)
            t_copy = min(T_lbl, T_img)
            z_copy = min(Z_lbl, Z_img)
            res[:t_copy, :z_copy, :Y_img, :X_img] = lbl[:t_copy, :z_copy, :Y_img, :X_img]
            return res, "4d", [gt4_path.name]
        except Exception as e:
            print(f"Warning: failed to load 4D GT {gt4_path.name}: {e}; falling back to per-frame.")

    # 2) Try per-frame GTs
    mapping, used_files = load_per_frame_gt_map(patient_dir)
    if not mapping:
        return None, "none", []

    res = np.zeros((T_img, Z_img, Y_img, X_img), dtype=np.uint8)
    for t_idx, arr3 in mapping.items():
        if t_idx < 0 or t_idx >= T_img:
            print(f"Warning: parsed frame index {t_idx} for a per-frame GT is out of bounds (T={T_img}); skipping")
            continue
        Z_lbl = arr3.shape[0]
        z_copy = min(Z_lbl, Z_img)
        res[t_idx, :z_copy, :Y_img, :X_img] = arr3[:z_copy, :Y_img, :X_img]
    return res, "per_frame", used_files


# -------------------------
# Core logic: process_volume
# -------------------------
def process_volume(
    image_path: Path,
    label_array: Optional[np.ndarray],
    label_source: str,
    label_files_used: List[str],
    out_root: Path,
    split: str,
    crop_cfg: int,
    patient_info: Dict[str, Optional[str]],
    mask_manifest: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Process a single 4D image and its aligned label_array (if any).
    Saves 2D slices and masks into out_root/split/patient_id/{data,masks}.
    Updates mask_manifest in-place.

    Returns:
      (original_nonzero, saved_nonzero)
    """
    patient_id = image_path.stem.split("_")[0]

    # Load image and reorder axes
    nifti = nib.load(str(image_path))
    img = nifti.get_fdata()
    img = reorder_to_tzyx(img, nifti.header)  # (T, Z, Y, X)
    if img.ndim != 4:
        raise RuntimeError(f"Image not 4D after reorder: {image_path} -> {img.shape}")

    T, Z, Y, X = img.shape

    # Prepare output dirs
    subject_dir = out_root / split / patient_id
    img_dir = ensure_dir(subject_dir / "data")
    msk_dir = ensure_dir(subject_dir / "masks") if label_array is not None else None

    # Save metadata for this patient (info.cfg + nifti header summary + label source info)
    meta = {
        "patient_id": patient_id,
        "source_image": str(image_path.name),
        "label_source": label_source,
        "label_files_used": label_files_used,
        "info_cfg": patient_info,
        "nifti_shape_after_reorder": img.shape,
        "pixdim": tuple(map(float, nifti.header.get_zooms()))
    }
    with open(subject_dir / "metadata.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    # Determine set of frames that had any nonzero mask in label_array
    mask_frames_with_data = set()
    if label_array is not None:
        for t in range(min(label_array.shape[0], T)):
            if label_array[t].sum() > 0:
                mask_frames_with_data.add(t)

    # Also ensure ED/ES frames from info.cfg are preserved if present (1-index in file -> convert)
    info_ints = info_values_as_ints(patient_info)
    ed_frame = info_ints.get("ED")
    es_frame = info_ints.get("ES")
    if ed_frame is not None:
        mask_frames_with_data.add(ed_frame - 1 if ed_frame > 0 else ed_frame)
    if es_frame is not None:
        mask_frames_with_data.add(es_frame - 1 if es_frame > 0 else es_frame)

    # Counters
    saved_images = 0
    saved_masks = 0
    real_masks = 0  # number of masks that contained any nonzero labels

    # Iterate and save slices
    for t in range(T):
        for z in range(Z):
            vol = img[t, z]  # (Y, X)
            h, w = vol.shape

            # safe crop
            max_safe_crop = min((h - 1) // 2, (w - 1) // 2)
            crop = min(crop_cfg, max_safe_crop)
            if crop <= 0:
                vol_cropped = vol
            else:
                vol_cropped = vol[crop:-crop, crop:-crop]

            if vol_cropped.size == 0:
                # skip impossible cropping
                continue

            # Normalize + standardize
            try:
                vol_norm = standardize(normalize(vol_cropped)).astype(np.float32)
            except Exception as e:
                print(f"Warning: normalization failed for {patient_id} t{t} z{z}: {e}; skipping")
                continue

            name = f"t{t:02d}_z{z:02d}.npy"
            np.save(img_dir / name, vol_norm)
            saved_images += 1

            # Save mask if label_array provided - always save a mask file (may be zero)
            if label_array is not None:
                if t < label_array.shape[0] and z < label_array.shape[1]:
                    mask = label_array[t, z]
                else:
                    mask = None

                if mask is None:
                    mask_cropped = np.zeros_like(vol_norm, dtype=np.uint8)
                else:
                    if crop <= 0:
                        mask_cropped = mask
                    else:
                        mask_cropped = mask[crop:-crop, crop:-crop]

                # Fix possible shape mismatches between vol_cropped and mask_cropped
                if mask_cropped is not None and vol_cropped.shape != mask_cropped.shape:
                    try:
                        if mask_cropped.T.shape == vol_cropped.shape:
                            mask_cropped = mask_cropped.T
                        else:
                            mh, mw = mask_cropped.shape
                            vh, vw = vol_cropped.shape
                            new_mask = np.zeros((vh, vw), dtype=np.uint8)
                            ch = min(mh, vh)
                            cw = min(mw, vw)
                            new_mask[:ch, :cw] = mask_cropped[:ch, :cw]
                            mask_cropped = new_mask
                    except Exception:
                        mask_cropped = np.zeros_like(vol_norm, dtype=np.uint8)
                    print(f"Warning: fixed crop shape mismatch for {patient_id} t{t} z{z}: vol {vol_cropped.shape} vs mask {mask_cropped.shape}")

                # Save mask (explicit zeros if empty)
                if mask_cropped.size == 0:
                    zero_mask = np.zeros_like(vol_norm, dtype=np.uint8)
                    np.save(msk_dir / name, zero_mask)
                else:
                    np.save(msk_dir / name, mask_cropped.astype(np.uint8))
                    if mask_cropped.sum() > 0:
                        real_masks += 1
                saved_masks += 1

    # compute labeled frame indices and per-frame nonzero counts (from label_array)
    labeled_frames = sorted(list(mask_frames_with_data))
    per_frame_nonzero: Dict[str, int] = {}
    if label_array is not None:
        for t in labeled_frames:
            per_frame_nonzero[str(int(t))] = int(label_array[t].sum())

    # update mask_manifest
    mask_manifest[patient_id] = {
        "saved_images": int(saved_images),
        "saved_masks": int(saved_masks),
        "real_masks": int(real_masks),
        "has_labels": label_array is not None,
        "label_source": label_source,
        "label_files_used": label_files_used,
        "labeled_frame_indices": labeled_frames,
        "per_frame_nonzero": per_frame_nonzero,
    }

    # compute original_nonzero and saved_nonzero for optional consistency checks
    original_nonzero = int(np.count_nonzero(label_array)) if label_array is not None else 0
    saved_nonzero = 0
    if msk_dir is not None and msk_dir.exists():
        for m in msk_dir.glob("*.npy"):
            try:
                arr = np.load(m)
                saved_nonzero += int(np.count_nonzero(arr))
            except Exception:
                continue

    print(f"Patient {patient_id} processed. Saved images: {saved_images}, saved masks: {saved_masks}, real masks: {real_masks}")
    return original_nonzero, saved_nonzero


# -------------------------
# Dataset traversal & splitting
# -------------------------
def collect_patients(training_root: Path) -> List[Path]:
    return sorted([p for p in training_root.iterdir() if p.is_dir()])


def stratified_split(patients: List[Path], train_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    """
    If each patient folder contains an info.cfg with Group, perform stratified split by Group.
    Otherwise perform deterministic shuffle split using seed.
    """
    groups: Dict[Optional[str], List[Path]] = {}
    for p in patients:
        info = parse_info_cfg(p)
        grp = info.get("Group") or "UNKNOWN"
        groups.setdefault(grp, []).append(p)

    if set(groups.keys()) == {"UNKNOWN"}:
        rng = random.Random(seed)
        pts = patients.copy()
        rng.shuffle(pts)
        n_train = int(len(pts) * train_ratio)
        return pts[:n_train], pts[n_train:]

    train_list: List[Path] = []
    val_list: List[Path] = []
    rng = random.Random(seed)
    for grp, grp_patients in groups.items():
        rng.shuffle(grp_patients)
        n_train_grp = int(len(grp_patients) * train_ratio)
        train_list.extend(grp_patients[:n_train_grp])
        val_list.extend(grp_patients[n_train_grp:])

    train_list = sorted(train_list, key=lambda p: p.name)
    val_list = sorted(val_list, key=lambda p: p.name)
    return train_list, val_list


# -------------------------
# Labeled split helper
# -------------------------
def build_labeled_split_from_manifest(mask_manifest: Dict[str, Any], train_ratio: float, seed: int) -> Dict[str, Any]:
    """Return a deterministic labeled-only split using patients with real_masks > 0."""
    labeled = [p for p, info in mask_manifest.items() if info.get("real_masks", 0) > 0]
    rng = random.Random(seed)
    rng.shuffle(labeled)
    n_train = int(len(labeled) * train_ratio)
    return {"train": labeled[:n_train], "val": labeled[n_train:], "train_ratio": train_ratio, "seed": seed}


# -------------------------
# Main run
# -------------------------
def run(
    dataset_root: Path,
    crop: int,
    train_ratio: float,
    seed: int,
    write_labeled_split: bool = True,
    require_manifest_consistency: bool = False,
    manifest_path_override: Optional[Path] = None,
):
    dataset_root = dataset_root.resolve()
    paths = dataset_paths(dataset_root)

    # training folder expected directly under dataset_root
    training_root = dataset_root / "training"
    if not training_root.exists():
        raise RuntimeError(f"'training' folder not found under {dataset_root}")

    preprocessed_root = ensure_dir(paths["preprocessed"])

    patients = collect_patients(training_root)
    if len(patients) == 0:
        raise RuntimeError("No patient folders found in training/")

    # Stratified split
    train_patients, val_patients = stratified_split(patients, train_ratio, seed)

    # Save split manifest for reproducibility
    split_manifest = {
        "train": [p.name for p in train_patients],
        "val": [p.name for p in val_patients],
        "train_ratio": train_ratio,
        "seed": seed
    }
    with open(preprocessed_root / "split_manifest.json", "w", encoding="utf-8") as sm:
        json.dump(split_manifest, sm, indent=2)

    print(f"Preprocessing {len(patients)} patients. Train/val split sizes: {len(train_patients)}/{len(val_patients)}")

    mask_manifest: Dict[str, Any] = {}

    # Process train
    for patient_dir in tqdm(train_patients, desc="Preprocessing train patients"):
        image_path = next(patient_dir.glob("*_4d.nii.gz"), None)
        if image_path is None:
            print(f"No 4D image found for {patient_dir.name}; skipping")
            continue

        # Determine image shape first
        try:
            nii_img = nib.load(str(image_path))
            img_arr = reorder_to_tzyx(nii_img.get_fdata(), nii_img.header)
            T_img, Z_img, Y_img, X_img = img_arr.shape
        except Exception as e:
            print(f"Warning: failed to load image {image_path.name}: {e}; skipping")
            continue

        # Build label array aligned with image shape
        label_array, label_source, label_files_used = build_label_array_from_sources((T_img, Z_img, Y_img, X_img), patient_dir)

        original_nonzero, saved_nonzero = process_volume(
            image_path=image_path,
            label_array=label_array,
            label_source=label_source,
            label_files_used=label_files_used,
            out_root=preprocessed_root,
            split="train",
            crop_cfg=crop,
            patient_info=parse_info_cfg(patient_dir),
            mask_manifest=mask_manifest,
        )

        # Optional per-patient consistency check
        if require_manifest_consistency:
            if saved_nonzero > original_nonzero:
                raise RuntimeError(f"Consistency check failed for {patient_dir.name}: saved_nonzero ({saved_nonzero}) > original_nonzero ({original_nonzero})")

    # Process val
    for patient_dir in tqdm(val_patients, desc="Preprocessing val patients"):
        image_path = next(patient_dir.glob("*_4d.nii.gz"), None)
        if image_path is None:
            print(f"No 4D image found for {patient_dir.name}; skipping")
            continue

        try:
            nii_img = nib.load(str(image_path))
            img_arr = reorder_to_tzyx(nii_img.get_fdata(), nii_img.header)
            T_img, Z_img, Y_img, X_img = img_arr.shape
        except Exception as e:
            print(f"Warning: failed to load image {image_path.name}: {e}; skipping")
            continue

        label_array, label_source, label_files_used = build_label_array_from_sources((T_img, Z_img, Y_img, X_img), patient_dir)

        original_nonzero, saved_nonzero = process_volume(
            image_path=image_path,
            label_array=label_array,
            label_source=label_source,
            label_files_used=label_files_used,
            out_root=preprocessed_root,
            split="val",
            crop_cfg=crop,
            patient_info=parse_info_cfg(patient_dir),
            mask_manifest=mask_manifest,
        )

        # Optional per-patient consistency check
        if require_manifest_consistency:
            if saved_nonzero > original_nonzero:
                raise RuntimeError(f"Consistency check failed for {patient_dir.name}: saved_nonzero ({saved_nonzero}) > original_nonzero ({original_nonzero})")

    # Save mask manifest
    try:
        manifest_path = preprocessed_root / "mask_manifest.json" if manifest_path_override is None else manifest_path_override
        with open(manifest_path, "w", encoding="utf-8") as mmf:
            json.dump(mask_manifest, mmf, indent=2)
    except Exception as e:
        print(f"Warning: failed to write mask_manifest.json: {e}")

    # Optionally write labeled-only split manifest for supervised experiments
    if write_labeled_split:
        try:
            labeled_split = build_labeled_split_from_manifest(mask_manifest, train_ratio, seed)
            with open(preprocessed_root / "split_manifest_labeled.json", "w", encoding="utf-8") as lf:
                json.dump(labeled_split, lf, indent=2)
            print("Wrote labeled-only split to:", preprocessed_root / "split_manifest_labeled.json")
        except Exception as e:
            print("Warning: failed to write labeled split manifest:", e)

    # Summary diagnostics
    try:
        total_saved = sum(info.get("saved_masks", 0) for info in mask_manifest.values())
        total_real = sum(info.get("real_masks", 0) for info in mask_manifest.values())
        frac = total_real / total_saved if total_saved > 0 else 0.0
        print(f"Mask manifest totals: saved_masks={total_saved}, real_masks={total_real}, fraction_labeled={frac:.4f}")
    except Exception:
        pass

    print("Preprocessing finished. Manifest saved to:", manifest_path)


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess cine-like 4D cardiac MRI into 2D slices (robust + info.cfg aware)")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Path to database root (contains training/)")
    parser.add_argument("--crop", type=int, default=DEFAULT_CROP, help="In-plane crop (pixels) applied to each side; pipeline will reduce if unsafe")
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Fraction of patients assigned to train (stratified by Group when available)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic splitting")
    parser.add_argument("--write_labeled_split", action="store_true", default=True, help="Write split_manifest_labeled.json using patients with real masks")
    parser.add_argument("--require_manifest_consistency", action="store_true", default=False, help="Run end-to-end mask consistency checks and raise on mismatch")
    parser.add_argument("--manifest_path", type=Path, default=None, help="Optional path to write mask_manifest.json (overrides default)")
    args = parser.parse_args()

    run(
        dataset_root=args.dataset_root,
        crop=args.crop,
        train_ratio=args.train_ratio,
        seed=args.seed,
        write_labeled_split=args.write_labeled_split,
        require_manifest_consistency=args.require_manifest_consistency,
        manifest_path_override=args.manifest_path,
    )
