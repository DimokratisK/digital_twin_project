#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from typing import List, Optional, Dict, Tuple, Any

# -------------------------
# Padding utilities (kept)
# -------------------------
def pad_to_multiple_2d(
    arr: np.ndarray,
    multiple: int = 16,
    mode: str = "constant",
    cval: float = 0.0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    if arr.ndim != 2:
        raise ValueError("pad_to_multiple_2d expects a 2D array")
    h, w = arr.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if mode == "constant":
        padded = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=cval,
        )
    else:
        padded = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode=mode,
        )
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def pad_to_target_2d(
    arr: np.ndarray,
    target_h: int,
    target_w: int,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("pad_to_target_2d expects a 2D array")
    h, w = arr.shape
    if h > target_h or w > target_w:
        raise ValueError(
            f"pad_to_target_2d: array shape {(h, w)} exceeds target {(target_h, target_w)}"
        )
    pad_h = target_h - h
    pad_w = target_w - w
    pad_top = 0
    pad_bottom = pad_h
    pad_left = 0
    pad_right = pad_w
    if mode == "constant":
        padded = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=cval,
        )
    else:
        padded = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode=mode,
        )
    return padded


# -------------------------
# CardiacDataset
# -------------------------
class CardiacDataset(Dataset):
    """
    Dataset for 2D cardiac MRI slice segmentation.

    Expected directory structure (preprocessed output):
        root/
            patient001/
                data/
                    t00_z00.npy
                masks/
                    t00_z00.npy
                metadata.json

    Key behaviors:
    - Ensures masks are integer tensors (torch.long) for CrossEntropyLoss.
    - Returns images as float32 tensors with channel-first shape (1,H,W).
    - If exclude_missing_masks=True, samples without mask files are omitted.
    - Otherwise missing masks are returned as explicit zero masks.
    - Exposes helpers for samplers and metadata inspection.
    """

    def __init__(
        self,
        root: Path,
        augment: Optional[A.Compose] = None,
        prefer_ed_es: bool = False,
        metadata_index: Optional[Dict[str, dict]] = None,
        one_hot: bool = False,
        n_classes: int = 4,
        exclude_missing_masks: bool = False,
        pad_multiple: int = 16,
    ):
        self.root = Path(root)
        self.augment = augment
        self.metadata_index = metadata_index or {}
        self.prefer_ed_es = prefer_ed_es
        self.one_hot = one_hot
        self.n_classes = int(n_classes)
        self.exclude_missing_masks = exclude_missing_masks
        self.pad_multiple = int(pad_multiple)

        # Build sample list (the old method that populates self.samples)
        self.samples: List[Tuple[str, int, int, Path, Optional[Path]]] = []
        self._build_samples()  # <-- existing method that currently fills self.samples

        # --- Ensure we only keep labeled samples when requested ---
        # At this point self.samples should be a list of tuples:
        # (patient_id, t_idx, z_idx, image_path, mask_path_or_None)
        
        if self.exclude_missing_masks:
            original_len = len(self.samples)

            # Option A (robust, correct): load each mask once and keep only those with any foreground
            filtered: List[Tuple[str, int, int, Path, Optional[Path]]] = []
            for s in self.samples:
                pid, t_idx, z_idx, img_p, mask_p = s
                if mask_p is None:
                    continue
                mask_path = Path(mask_p)
                if not mask_path.exists():
                    continue
                try:
                    arr = np.load(mask_path)
                    # keep only masks that contain at least one nonzero (foreground) pixel
                    if np.any(arr != 0):
                        filtered.append(s)
                except Exception:
                    # if loading fails, skip this sample (avoid silent zero-mask fallbacks)
                    continue

            self.samples = filtered
            filtered_len = len(self.samples)
            print(f"[dataset] filtered unlabeled/empty masks: {original_len} → {filtered_len}")

        else:
            # do nothing; keep the originally discovered list
            pass

        # rebuild sample->patient mapping used by samplers
        self._sample_to_patient = [s[0] for s in self.samples]



        if len(self.samples) == 0:
            raise RuntimeError(
                "No samples available after filtering. "
                "If you expected labeled samples, check preprocessed/ and mask_manifest.json"
            )

        # Compute global target size (remainder of __init__ unchanged)
        self.target_h, self.target_w = self._compute_global_target_size()

    # -------------------------
    # Sample collection
    # -------------------------
    def _build_samples(self) -> None:
        base_patients = sorted([p for p in self.root.glob("*") if p.is_dir()])

        for patient_dir in base_patients:
            pid = patient_dir.name
            data_dir = patient_dir / "data"
            masks_dir = patient_dir / "masks"

            if not data_dir.exists():
                continue

            image_files = sorted(data_dir.glob("*.npy"))
            for img_path in image_files:
                t_idx, z_idx = self._parse_tz_from_name(img_path.stem)
                mask_path = masks_dir / img_path.name
                if not mask_path.exists():
                    mask_path = None
                    if self.exclude_missing_masks:
                        continue
                self.samples.append((pid, t_idx, z_idx, img_path, mask_path))

        # Optionally upsample ED/ES frames (duplicates appended to samples)
        if self.prefer_ed_es and self.metadata_index:
            extra_samples: List[Tuple[str, int, int, Path, Optional[Path]]] = []
            samples_by_patient: Dict[str, List[Tuple[str, int, int, Path, Optional[Path]]]] = {}
            for s in self.samples:
                samples_by_patient.setdefault(s[0], []).append(s)

            for pid, meta in self.metadata_index.items():
                info = meta.get("info_cfg", {}) if isinstance(meta, dict) else {}
                ed = info.get("ED")
                es = info.get("ES")
                try:
                    ed_idx = int(ed) - 1 if ed is not None else None
                except Exception:
                    ed_idx = None
                try:
                    es_idx = int(es) - 1 if es is not None else None
                except Exception:
                    es_idx = None

                patient_samples = samples_by_patient.get(pid, [])
                for s in patient_samples:
                    _, t, _, _, _ = s
                    if ed_idx is not None and t == ed_idx:
                        extra_samples.append(s)
                    if es_idx is not None and t == es_idx:
                        extra_samples.append(s)

            if extra_samples:
                self.samples.extend(extra_samples)
                # update sample->patient map
                self._sample_to_patient = [s[0] for s in self.samples]

    def _compute_global_target_size(self) -> Tuple[int, int]:
        """
        Compute global target height and width using per-patient metadata.json files.

        Strategy:
        - For each unique patient present in self.samples, look for patient_dir/metadata.json.
        - If metadata.json exists and contains "nifti_shape_after_reorder", extract (T,Z,Y,X)
          and use Y,X as the per-patient image H,W.
        - If metadata.json is missing or malformed for a patient, fall back to loading a single
          representative .npy image for that patient (only one load per patient).
        - Aggregate max H and W across patients, then round up to the configured pad_multiple.
        """
        max_h = 0
        max_w = 0
        seen_patients = set()
        # iterate samples but only inspect each patient once
        for pid, _, _, img_path, _ in self.samples:
            if pid in seen_patients:
                continue
            seen_patients.add(pid)
            patient_dir = img_path.parent.parent  # data/<file> -> patient_dir
            meta_path = patient_dir / "metadata.json"
            h = None
            w = None
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    # Expect "nifti_shape_after_reorder": [T, Z, Y, X]
                    nifti_shape = meta.get("nifti_shape_after_reorder")
                    if isinstance(nifti_shape, (list, tuple)) and len(nifti_shape) >= 4:
                        # Y is index 2, X is index 3
                        h = int(nifti_shape[2])
                        w = int(nifti_shape[3])
                except Exception:
                    # malformed metadata.json -> fallback to single .npy load below
                    h = None
                    w = None

            if h is None or w is None:
                # Fallback: load a single representative .npy for this patient (first file)
                try:
                    data_dir = patient_dir / "data"
                    first_img = next(data_dir.glob("*.npy"), None)
                    if first_img is not None:
                        arr = np.load(first_img)
                        if arr.ndim == 2:
                            h, w = arr.shape
                        else:
                            # if somehow not 2D, try to squeeze or take last two dims
                            arr = np.asarray(arr)
                            if arr.ndim >= 2:
                                h, w = arr.shape[-2], arr.shape[-1]
                            else:
                                h, w = 0, 0
                    else:
                        h, w = 0, 0
                except Exception:
                    h, w = 0, 0

            if h is None or w is None:
                h, w = 0, 0

            if h > max_h:
                max_h = h
            if w > max_w:
                max_w = w

        # Round up to pad_multiple
        m = self.pad_multiple
        target_h = ((max_h + m - 1) // m) * m if max_h > 0 else m
        target_w = ((max_w + m - 1) // m) * m if max_w > 0 else m
        return target_h, target_w

    @staticmethod
    def _parse_tz_from_name(stem: str) -> Tuple[int, int]:
        t_idx = 0
        z_idx = 0
        try:
            parts = stem.split("_")
            for p in parts:
                if p.startswith("t"):
                    t_idx = int(p[1:])
                elif p.startswith("z"):
                    z_idx = int(p[1:])
        except Exception:
            t_idx, z_idx = 0, 0
        return t_idx, z_idx

    # -------------------------
    # Sampler / introspection helpers
    # -------------------------
    def sample_to_patient(self, idx: int) -> str:
        return self._sample_to_patient[idx]

    def patient_ids(self) -> List[str]:
        return sorted(list({s[0] for s in self.samples}))

    def samples_for_patient(self, patient_id: str) -> List[int]:
        return [i for i, s in enumerate(self.samples) if s[0] == patient_id]

    def has_mask(self, idx: int, check_nonzero: bool = False) -> bool:
        mask_path = self.samples[idx][4]
        if mask_path is None:
            return False

        if not check_nonzero:
            return True

    # check_nonzero=True → verify mask actually contains labeled pixels
        try:
            import numpy as np
            arr = np.load(mask_path)
            return np.any(arr != 0)
        except Exception:
            return False
    
    

    def get_sample_meta(self, idx: int) -> Dict[str, Any]:
        pid, t_idx, z_idx, img_path, mask_path = self.samples[idx]
        meta = self.metadata_index.get(pid, {})
        return {
            "patient_id": pid,
            "t": t_idx,
            "z": z_idx,
            "image_path": str(img_path),
            "mask_path": str(mask_path) if mask_path is not None else None,
            "metadata": meta,
        }

    # -------------------------
    # Dataset API
    # -------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pid, t_idx, z_idx, image_path, mask_path = self.samples[idx]

        # Load image (preprocessed floats expected)
        image = np.load(image_path).astype(np.float32)

        # Mask must exist because dataset was constructed with exclude_missing_masks=True
        if mask_path is None:
            raise RuntimeError(f"Missing mask for sample {image_path}. Dataset should have filtered unlabeled samples.")
        mask = np.load(mask_path).astype(np.uint8)


        # Apply augmentations (albumentations expects HxW arrays)
        if self.augment is not None:
            augmented = self.augment(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Pad image and mask to global target size (target_h, target_w)
        image_padded = pad_to_target_2d(
            image,
            target_h=self.target_h,
            target_w=self.target_w,
            mode="constant",
            cval=0.0,
        )
        mask_padded = pad_to_target_2d(
            mask,
            target_h=self.target_h,
            target_w=self.target_w,
            mode="constant",
            cval=0,
        )

        # Mask shape check
        if image_padded.shape != mask_padded.shape:
            raise RuntimeError(f"Image/mask shape mismatch after padding for sample {image_path}: image {image_padded.shape}, mask {mask_padded.shape}")


        # Convert mask to desired format
        if self.one_hot:
            mask_tensor = self._to_one_hot(mask_padded, self.n_classes)  # (C, H, W) float
            mask_tensor = mask_tensor.float()
        else:
            # keep integer labels as single channel and ensure long dtype for CrossEntropy
            mask_tensor = torch.from_numpy(mask_padded.astype(np.int64)).long().unsqueeze(0)  # (1, H, W)

        # Convert image to tensor with channel-first and float32
        image_tensor = torch.from_numpy(np.expand_dims(image_padded.astype(np.float32), axis=0)).float()  # (1, H, W)

        # Final sanity assertion 
        if mask_tensor.shape[1:] != image_tensor.shape[1:]:
            raise RuntimeError(f"Final image/mask shape mismatch for sample {image_path}: image {image_tensor.shape}, mask {mask_tensor.shape}")


        return image_tensor, mask_tensor

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _to_one_hot(mask: np.ndarray, n_classes: int) -> torch.Tensor:
        h, w = mask.shape
        one_hot = np.zeros((n_classes, h, w), dtype=np.uint8)
        mask_clipped = np.clip(mask, 0, n_classes - 1)
        for c in range(n_classes):
            one_hot[c] = (mask_clipped == c).astype(np.uint8)
        return torch.from_numpy(one_hot)
