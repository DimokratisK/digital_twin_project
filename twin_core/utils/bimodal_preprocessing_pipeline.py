#!/usr/bin/env python3
"""
Refactored preprocessing pipeline for cine-like 4D cardiac NIfTI datasets.

Improvements:
- Robust axis/orientation handling via nibabel.as_closest_canonical
- Streaming (per-slice) processing using nibabel.dataobj to avoid loading full 4D volumes
- Explicit out_root (--out_root); optional two-input pre-split mode (--train_input, --test_input)
- Configurable normalization: --norm {none,zscore,minmax,double}
- Configurable globbing for images and per-frame gt files
- Dry-run mode, logging, metadata with pixdim+affine
- Writes mask_manifest.json, split_manifest.json and split_manifest_labeled.json (if requested)
"""
from pathlib import Path
import argparse
import json
import random
import re
import logging
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List, Any

from .paths import dataset_paths, ensure_dir

# -------------------------
# Defaults
# -------------------------
DEFAULT_CROP = 8
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_EPS = 1e-8
DEFAULT_SEED = 42

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preproc")

EPS = DEFAULT_EPS

# -------------------------
# Normalization helpers
# -------------------------
def normalize_zscore(arr: np.ndarray) -> np.ndarray:
    mu = np.mean(arr)
    std = np.std(arr) + EPS
    return (arr - mu) / std


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    return (arr - vmin) / (vmax - vmin + EPS)


def apply_normalization(arr: np.ndarray, mode: str = "double") -> np.ndarray:
    """
    mode: 'none' | 'zscore' | 'minmax' | 'double'
    'double' = zscore then minmax (keeps previous behaviour by default)
    """
    if mode == "none":
        return arr.astype(np.float32)
    if mode == "zscore":
        return normalize_zscore(arr).astype(np.float32)
    if mode == "minmax":
        return normalize_minmax(arr).astype(np.float32)
    if mode == "double":
        return normalize_minmax(normalize_zscore(arr)).astype(np.float32)
    raise ValueError(f"Unknown norm mode: {mode}")


# -------------------------
# Helpers: file naming and parsing
# -------------------------
_FRAME_RE = re.compile(r"(?:[_\-]|^)(?:frame|f)0*([0-9]+)", flags=re.IGNORECASE)


def _parse_frame_index_from_name(fname: str) -> Optional[int]:
    m = _FRAME_RE.search(fname)
    if not m:
        return None
    try:
        idx = int(m.group(1))
        return max(0, idx - 1)
    except Exception:
        return None


# -------------------------
# nibabel / canonicalization helpers
# -------------------------
def canonicalize_nii(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Return a nibabel image in closest canonical orientation (RAS) to reduce orientation surprises.
    """
    try:
        return nib.as_closest_canonical(nii)
    except Exception:
        # best-effort fallback
        return nii


def canonical_shape_and_spacing(nii: nib.Nifti1Image) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """
    Return canonical shape and pixdim (zooms) of a nib image after canonicalization.
    """
    nii_c = canonicalize_nii(nii)
    shape = tuple(nii_c.shape)  # could be (X,Y,Z) or (X,Y,Z,T)
    pixdim = tuple(map(float, nii_c.header.get_zooms()))
    return shape, pixdim


def extract_slice_from_canonical(nii_c: nib.Nifti1Image, t_idx: int, z_idx: int) -> np.ndarray:
    """
    Given a canonicalized nib image (nii_c), extract a 2D slice for indices (t_idx, z_idx)
    and return a (Y, X) numpy array (type float).
    Assumes canonical spatial axes ordering (X,Y,Z) and time axis (if present) as the last axis.
    For a 4D image canonical shape is (X, Y, Z, T) so a single slice is nii_c.dataobj[:, :, z_idx, t_idx].
    For a 3D image canonical shape is (X, Y, Z) and t_idx is ignored (treated as 0).
    Returned array is transposed to (Y, X) to match earlier code expectations.
    """
    shape = nii_c.shape
    dataobj = nii_c.dataobj  # lazy access
    if len(shape) == 3:
        # (X, Y, Z)
        x_y = np.asarray(dataobj[:, :, int(z_idx)])  # (X, Y)
        return x_y.T  # (Y, X)
    elif len(shape) == 4:
        # (X, Y, Z, T)
        # clamp indices to valid ranges (caller should ensure valid indices)
        t = int(t_idx)
        z = int(z_idx)
        # indexing order: dataobj[:, :, z, t] -> (X, Y)
        x_y = np.asarray(dataobj[:, :, z, t])
        return x_y.T  # (Y, X)
    else:
        raise RuntimeError(f"Unsupported canonical image ndim: {len(shape)} for {nii_c}")


# -------------------------
# Parse info.cfg helpers (kept from original)
# -------------------------
def parse_info_cfg(patient_dir: Path) -> Dict[str, Optional[str]]:
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
        logger.warning("Failed to parse info.cfg for %s", patient_dir)
    return info


def info_values_as_ints(info: Dict[str, Optional[str]]) -> Dict[str, Optional[Any]]:
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
# Load per-frame GT map (uses nibabel canonicalization)
# -------------------------
def load_per_frame_gt_map(patient_dir: Path, mask_glob: str = "*_frame*_gt.nii*") -> Tuple[Dict[int, np.ndarray], List[str]]:
    frame_files = sorted(list(patient_dir.glob(mask_glob)))
    mapping: Dict[int, np.ndarray] = {}
    used_files: List[str] = []
    for p in frame_files:
        t_idx = _parse_frame_index_from_name(p.name)
        if t_idx is None:
            logger.debug("Could not parse frame index from %s; skipping", p.name)
            continue
        try:
            nii = nib.load(str(p))
            nii_c = canonicalize_nii(nii)
            arr = np.asarray(nii_c.get_fdata())  # load mask array (small)
            # Accept a few shapes
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr3 = arr[0]
            elif arr.ndim == 3:
                arr3 = arr
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
            mapping[int(t_idx)] = arr3.astype(np.uint8)
            used_files.append(p.name)
        except Exception as e:
            logger.warning("Failed to load per-frame GT %s: %s", p.name, e)
            continue
    return mapping, used_files


# -------------------------
# Build full label array aligned with canonical image shape (T,Z,Y,X)
# -------------------------
def build_label_array_from_sources(image_shape: Tuple[int, int, int, int], patient_dir: Path, mask_glob: str = "*_frame*_gt.nii*") -> Tuple[Optional[np.ndarray], str, List[str]]:
    """
    Tries to build a label array (T,Z,Y,X) aligned to the canonicalized image axes.
    Returns (label_array or None, label_source, list_of_files_used)
    """
    T_img, Z_img, Y_img, X_img = image_shape
    # 1) Try single 4D GT (look for *_4d_gt.nii*)
    gt4_path = next(patient_dir.glob("*_4d_gt.nii*"), None)
    if gt4_path is not None:
        try:
            nii = nib.load(str(gt4_path))
            nii_c = canonicalize_nii(nii)
            lbl = np.asarray(nii_c.get_fdata()).astype(np.uint8)
            # shape handling
            if lbl.ndim == 3:
                lbl = lbl[None]
            if lbl.ndim != 4:
                raise RuntimeError(f"Unexpected GT 4D shape {lbl.shape} for {gt4_path}")
            # Copy into aligned container
            res = np.zeros((T_img, Z_img, Y_img, X_img), dtype=np.uint8)
            t_copy = min(lbl.shape[0], T_img)
            z_copy = min(lbl.shape[1], Z_img)
            res[:t_copy, :z_copy, :Y_img, :X_img] = lbl[:t_copy, :z_copy, :Y_img, :X_img]
            return res, "4d", [gt4_path.name]
        except Exception as e:
            logger.warning("Failed to load 4D GT %s: %s; falling back to per-frame", gt4_path.name, e)

    # 2) per-frame
    mapping, used_files = load_per_frame_gt_map(patient_dir, mask_glob=mask_glob)
    if not mapping:
        return None, "none", []
    res = np.zeros((T_img, Z_img, Y_img, X_img), dtype=np.uint8)
    for t_idx, arr3 in mapping.items():
        if t_idx < 0 or t_idx >= T_img:
            logger.warning("Parsed frame index %d out of bounds for T=%d; skipping", t_idx, T_img)
            continue
        Z_lbl = arr3.shape[0]
        z_copy = min(Z_lbl, Z_img)
        res[t_idx, :z_copy, :Y_img, :X_img] = arr3[:z_copy, :Y_img, :X_img]
    return res, "per_frame", used_files


# -------------------------
# Core processing: process_volume (streaming-friendly)
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
    norm_mode: str = "double",
    image_glob: str = "*_4d.nii*",
    dry_run: bool = False,
    stream: bool = True,
) -> Tuple[int, int]:
    """
    Process a single image; save 2D slices (data) and masks (if label_array provided) into out_root/<split>/<patient_id>/{data,masks}
    Returns (original_nonzero, saved_nonzero)
    """
    patient_id = image_path.stem.split("_")[0]
    logger.info("Processing patient %s from %s (split=%s)", patient_id, image_path.name, split)

    nii = nib.load(str(image_path))
    nii_c = canonicalize_nii(nii)
    # canonical shape is typically (X, Y, Z) or (X, Y, Z, T)
    shape = nii_c.shape
    # infer T,Z,Y,X from canonical shape
    if len(shape) == 3:
        X, Y, Z = shape
        T = 1
    elif len(shape) == 4:
        X, Y, Z, T = shape
    else:
        raise RuntimeError(f"Unsupported canonical image shape {shape} for {image_path}")

    # create output dirs
    subject_dir = out_root / split / patient_id
    img_dir = ensure_dir(subject_dir / "data")
    msk_dir = ensure_dir(subject_dir / "masks") if label_array is not None else None

    # gather metadata (pixdim and affine if available)
    try:
        pixdim = tuple(map(float, nii_c.header.get_zooms()))
    except Exception:
        pixdim = ()
    try:
        affine = nii_c.affine.tolist() if hasattr(nii_c, "affine") else None
    except Exception:
        affine = None

    meta = {
        "patient_id": patient_id,
        "source_image": str(image_path.name),
        "label_source": label_source,
        "label_files_used": label_files_used,
        "info_cfg": patient_info,
        "canonical_shape": (T, Z, Y, X),
        "pixdim": pixdim,
        "affine": affine,
    }
    if not dry_run:
        with open(subject_dir / "metadata.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)

    # find frames that have mask (from label_array if provided)
    mask_frames_with_data = set()
    if label_array is not None:
        for t in range(min(label_array.shape[0], T)):
            if label_array[t].sum() > 0:
                mask_frames_with_data.add(t)

    # include ED/ES from info.cfg (convert to 0-index)
    info_ints = info_values_as_ints(patient_info)
    ed_frame = info_ints.get("ED")
    es_frame = info_ints.get("ES")
    if ed_frame is not None:
        mask_frames_with_data.add(ed_frame - 1 if ed_frame > 0 else ed_frame)
    if es_frame is not None:
        mask_frames_with_data.add(es_frame - 1 if es_frame > 0 else es_frame)

    saved_images = 0
    saved_masks = 0
    real_masks = 0

    # Streaming loops: iterate over t and z
    # note: canonical axes ordering implies slice extraction via extract_slice_from_canonical
    for t in range(T):
        for z in range(Z):
            try:
                vol = extract_slice_from_canonical(nii_c, t, z)  # (Y, X)
            except Exception as e:
                logger.warning("Failed to extract slice t=%d z=%d for %s: %s", t, z, e)
                continue

            h, w = vol.shape
            max_safe_crop = min((h - 1) // 2, (w - 1) // 2)
            crop = min(crop_cfg, max_safe_crop)
            if crop > 0:
                vol_cropped = vol[crop:-crop, crop:-crop]
            else:
                vol_cropped = vol

            if vol_cropped.size == 0:
                logger.debug("Empty crop for %s t%d z%d; skipping", patient_id, t, z)
                continue

            # normalize according to user choice
            try:
                vol_norm = apply_normalization(vol_cropped.astype(np.float32), norm_mode)
            except Exception as e:
                logger.warning("Normalization failed for %s t%d z%d: %s", patient_id, t, z, e)
                continue

            name = f"t{t:02d}_z{z:02d}.npy"
            if not dry_run:
                np.save(img_dir / name, vol_norm)
            saved_images += 1

            # mask handling
            if label_array is not None:
                if t < label_array.shape[0] and z < label_array.shape[1]:
                    mask = label_array[t, z]
                else:
                    mask = None

                if mask is None:
                    mask_cropped = np.zeros_like(vol_norm, dtype=np.uint8)
                else:
                    if crop > 0:
                        mask_cropped = mask[crop:-crop, crop:-crop]
                    else:
                        mask_cropped = mask

                # fix shape mismatches conservatively
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
                            logger.debug("Fixed mask shape for %s t%d z%d", patient_id, t, z)
                    except Exception:
                        mask_cropped = np.zeros_like(vol_norm, dtype=np.uint8)
                        logger.debug("Mask shape fix failed for %s t%d z%d -> using zeros", patient_id, t, z)

                if not dry_run:
                    np.save(msk_dir / name, mask_cropped.astype(np.uint8))
                saved_masks += 1
                if mask_cropped is not None and mask_cropped.sum() > 0:
                    real_masks += 1

    # compute labeled frame indices and per-frame nonzero counts
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

    # compute original_nonzero and saved_nonzero safely
    original_nonzero = int(np.count_nonzero(label_array)) if label_array is not None else 0
    saved_nonzero = 0
    if msk_dir is not None and msk_dir.exists() and not dry_run:
        for m in msk_dir.glob("*.npy"):
            try:
                arr = np.load(m)
                saved_nonzero += int(np.count_nonzero(arr))
            except Exception:
                continue

    logger.info("Patient %s done. saved_images=%d saved_masks=%d real_masks=%d", patient_id, saved_images, saved_masks, real_masks)
    return original_nonzero, saved_nonzero


# -------------------------
# Dataset traversal & splitting
# -------------------------
def collect_patients(training_root: Path, pattern: str = "*") -> List[Path]:
    return sorted([p for p in training_root.iterdir() if p.is_dir() and p.match(pattern)])


def stratified_split(patients: List[Path], train_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
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


def build_labeled_split_from_manifest(mask_manifest: Dict[str, Any], train_ratio: float, seed: int) -> Dict[str, Any]:
    labeled = [p for p, info in mask_manifest.items() if info.get("real_masks", 0) > 0]
    rng = random.Random(seed)
    rng.shuffle(labeled)
    n_train = int(len(labeled) * train_ratio)
    return {"train": labeled[:n_train], "val": labeled[n_train:], "train_ratio": train_ratio, "seed": seed}


# -------------------------
# Top-level run
# -------------------------
def run(
    dataset_root: Path,
    crop: int,
    train_ratio: float,
    seed: int,
    write_labeled_split: bool = True,
    require_manifest_consistency: bool = False,
    manifest_path_override: Optional[Path] = None,
    out_root: Optional[Path] = None,
    train_input: Optional[Path] = None,
    test_input: Optional[Path] = None,
    norm_mode: str = "double",
    image_glob: str = "*_4d.nii*",
    mask_glob: str = "*_frame*_gt.nii*",
    dry_run: bool = False,
    stream: bool = True,
):
    dataset_root = Path(dataset_root).resolve()
    if out_root is None:
        paths = dataset_paths(dataset_root)
        preprocessed_root = ensure_dir(paths["preprocessed"])
    else:
        preprocessed_root = ensure_dir(Path(out_root).resolve())

    logger.info("Preprocessed root: %s", preprocessed_root)

    # Determine mode: two-input (train_input+test_input) or single-pool (dataset_root/training)
    if train_input is not None and test_input is not None:
        # pre-split two-input mode
        train_root = Path(train_input).resolve()
        test_root = Path(test_input).resolve()
        if not train_root.exists() or not test_root.exists():
            raise RuntimeError("train_input or test_input path not found")
        train_patients = collect_patients(train_root)
        val_patients = collect_patients(test_root)
        logger.info("Two-input mode: train patients=%d val patients=%d", len(train_patients), len(val_patients))
        # save split manifest for reproducibility
        split_manifest = {
            "train": [p.name for p in train_patients],
            "val": [p.name for p in val_patients],
            "train_ratio": len(train_patients) / max(1, (len(train_patients) + len(val_patients))),
            "seed": seed,
            "mode": "two_input",
        }
        with open(preprocessed_root / "split_manifest.json", "w", encoding="utf-8") as sm:
            json.dump(split_manifest, sm, indent=2)
    else:
        # single-pool mode expects dataset_root/training
        training_root = dataset_root / "training"
        if not training_root.exists():
            raise RuntimeError(f"'training' folder not found under {dataset_root}. Provide --train_input/--test_input for pre-split mode or put data under dataset_root/training.")
        patients = collect_patients(training_root)
        if len(patients) == 0:
            raise RuntimeError("No patient folders found in training/")
        train_patients, val_patients = stratified_split(patients, train_ratio, seed)
        split_manifest = {
            "train": [p.name for p in train_patients],
            "val": [p.name for p in val_patients],
            "train_ratio": train_ratio,
            "seed": seed,
            "mode": "stratified_split",
        }
        with open(preprocessed_root / "split_manifest.json", "w", encoding="utf-8") as sm:
            json.dump(split_manifest, sm, indent=2)
        logger.info("Single-pool mode: total patients=%d -> train=%d val=%d", len(patients), len(train_patients), len(val_patients))

    # main loop: process train then val
    mask_manifest: Dict[str, Any] = {}

    # helper to process sets
    def _process_list(patient_dirs: List[Path], split_name: str):
        for patient_dir in tqdm(patient_dirs, desc=f"Preprocessing {split_name} patients"):
            image_path = next(patient_dir.glob(image_glob), None)
            if image_path is None:
                logger.warning("No 4D image found for %s; skipping", patient_dir.name)
                continue

            # determine canonical shape
            try:
                nii_img = nib.load(str(image_path))
                nii_c = canonicalize_nii(nii_img)
                shape = nii_c.shape
                # derive (T,Z,Y,X)
                if len(shape) == 3:
                    X, Y, Z = shape
                    T = 1
                elif len(shape) == 4:
                    X, Y, Z, T = shape
                else:
                    logger.warning("Unsupported canonical shape %s for %s; skipping", shape, image_path)
                    continue
            except Exception as e:
                logger.warning("Failed to inspect image %s: %s; skipping", image_path.name, e)
                continue

            # build label array aligned with canonicalized image shape
            label_array, label_source, label_files_used = build_label_array_from_sources((T, Z, Y, X), patient_dir, mask_glob=mask_glob)

            try:
                orig_nonzero, saved_nonzero = process_volume(
                    image_path=image_path,
                    label_array=label_array,
                    label_source=label_source,
                    label_files_used=label_files_used,
                    out_root=preprocessed_root,
                    split=split_name,
                    crop_cfg=crop,
                    patient_info=parse_info_cfg(patient_dir),
                    mask_manifest=mask_manifest,
                    norm_mode=norm_mode,
                    image_glob=image_glob,
                    dry_run=dry_run,
                    stream=stream,
                )
            except Exception as e:
                logger.exception("Failed to process %s: %s", patient_dir.name, e)
                continue

            if require_manifest_consistency and not dry_run:
                if saved_nonzero > orig_nonzero:
                    raise RuntimeError(f"Consistency check failed for {patient_dir.name}: saved_nonzero ({saved_nonzero}) > original_nonzero ({orig_nonzero})")

    # process train list
    _process_list(train_patients, "train")
    # process val list
    _process_list(val_patients, "val")

    # save mask manifest
    try:
        manifest_path = preprocessed_root / "mask_manifest.json" if manifest_path_override is None else manifest_path_override
        if not dry_run:
            with open(manifest_path, "w", encoding="utf-8") as mmf:
                json.dump(mask_manifest, mmf, indent=2)
        logger.info("Wrote mask_manifest to: %s", manifest_path)
    except Exception as e:
        logger.warning("Failed to write mask_manifest.json: %s", e)

    # write labeled-only split manifest if requested
    if write_labeled_split and not dry_run:
        try:
            labeled_split = build_labeled_split_from_manifest(mask_manifest, train_ratio, seed)
            with open(preprocessed_root / "split_manifest_labeled.json", "w", encoding="utf-8") as lf:
                json.dump(labeled_split, lf, indent=2)
            logger.info("Wrote labeled-only split to: %s", preprocessed_root / "split_manifest_labeled.json")
        except Exception as e:
            logger.warning("Failed to write labeled split manifest: %s", e)

    # summary diagnostics
    try:
        total_saved = sum(info.get("saved_masks", 0) for info in mask_manifest.values())
        total_real = sum(info.get("real_masks", 0) for info in mask_manifest.values())
        frac = total_real / total_saved if total_saved > 0 else 0.0
        logger.info("Mask manifest totals: saved_masks=%d real_masks=%d fraction_labeled=%.4f", total_saved, total_real, frac)
    except Exception:
        pass

    logger.info("Preprocessing finished. Manifest path: %s", manifest_path)


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess cine 4D cardiac MRI into 2D slices (robust + streaming)")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Path to dataset root (contains training/).")
    parser.add_argument("--out_root", type=Path, default=None, help="Explicit preprocessed output root (overrides inferred location).")
    parser.add_argument("--train_input", type=Path, default=None, help="(Optional) pre-split train input folder (each patient subfolder inside).")
    parser.add_argument("--test_input", type=Path, default=None, help="(Optional) pre-split test/val input folder (each patient subfolder inside).")
    parser.add_argument("--crop", type=int, default=DEFAULT_CROP, help="In-plane crop (pixels) applied to each side; pipeline will reduce if unsafe.")
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Fraction of patients for training (ignored in two-input mode).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic splitting.")
    parser.add_argument("--write_labeled_split", action="store_true", default=True, help="Write split_manifest_labeled.json using patients with real masks.")
    parser.add_argument("--require_manifest_consistency", action="store_true", default=False, help="Run end-to-end mask consistency checks and raise on mismatch.")
    parser.add_argument("--manifest_path", type=Path, default=None, help="Optional path to write mask_manifest.json (overrides default).")
    parser.add_argument("--norm", type=str, choices=("none", "zscore", "minmax", "double"), default="double", help="Normalization mode to apply to slices.")
    parser.add_argument("--image_glob", type=str, default="*_4d.nii*", help="Glob for discovering 4D input images inside patient folder.")
    parser.add_argument("--mask_glob", type=str, default="*_frame*_gt.nii*", help="Glob for discovering per-frame GT masks.")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Do not write output files; only display what would be processed.")
    parser.add_argument("--stream", action="store_true", default=True, help="Stream slices instead of loading full volume into memory (recommended).")
    args = parser.parse_args()

    run(
        dataset_root=args.dataset_root,
        crop=args.crop,
        train_ratio=args.train_ratio,
        seed=args.seed,
        write_labeled_split=args.write_labeled_split,
        require_manifest_consistency=args.require_manifest_consistency,
        manifest_path_override=args.manifest_path,
        out_root=args.out_root,
        train_input=args.train_input,
        test_input=args.test_input,
        norm_mode=args.norm,
        image_glob=args.image_glob,
        mask_glob=args.mask_glob,
        dry_run=args.dry_run,
        stream=args.stream,
    )
