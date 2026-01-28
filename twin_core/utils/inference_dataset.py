#!/usr/bin/env python3
"""
inference_dataset.py

Dataset for running inference on 2D slices extracted from either:
 - a 4D cine NIfTI (T, Z, Y, X) file
 - a preprocessed folder following the repo's layout (patient/data/*.npy)

The output of __getitem__ is:

    image_tensor, meta

where:
 - image_tensor: torch.FloatTensor shape (1, H_p, W_p) dtype float32 (channel-first),
                 padded to the same global target size used in training.
 - meta: dict with keys:
     - "source": "nifti" or "preprocessed"
     - "patient_id"
     - "t": time/frame index
     - "z": slice index
     - "image_path": str or None
     - "orig_shape": (H, W)
     - "pad": (top, bottom, left, right)

This dataset intentionally mirrors CardiacDataset.__getitem__ processing
(image dtype, channel, padding) so you can directly compare inferred masks
to ground-truth masks produced by the training dataset.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import warnings
import numpy as np
import torch
from torch.utils.data import Dataset

# Reuse padding helper from dataset.py to keep behaviour identical.
# dataset.pad_to_target_2d was defined at module top-level.
try:
    from twin_core.data_ingestion.dataset import pad_to_target_2d, pad_to_multiple_2d
except Exception:
    # Fallback: local minimal implementation if import fails
    def pad_to_target_2d(arr: np.ndarray, target_h: int, target_w: int, mode: str = "constant", cval: float = 0.0) -> np.ndarray:
        h, w = arr.shape
        if h > target_h or w > target_w:
            raise ValueError(f"pad_to_target_2d: array {(h,w)} exceeds target {(target_h,target_w)}")
        pad_h = target_h - h
        pad_w = target_w - w
        pad_top = 0
        pad_bottom = pad_h
        pad_left = 0
        pad_right = pad_w
        if mode == "constant":
            padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=cval)
        else:
            padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)
        return padded

    def pad_to_multiple_2d(arr: np.ndarray, multiple: int = 16, mode: str = "constant", cval: float = 0.0) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
        h, w = arr.shape
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if mode == "constant":
            padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=cval)
        else:
            padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)
        return padded, (pad_top, pad_bottom, pad_left, pad_right)


# Prefer SimpleITK for robust NIfTI handling; fallback to nibabel if needed.
try:
    import SimpleITK as sitk  # type: ignore
    _HAS_SITK = True
except Exception:
    _HAS_SITK = False
    try:
        import nibabel as nib  # type: ignore
        _HAS_NIB = True
    except Exception:
        _HAS_NIB = False


class InferenceDataset(Dataset):
    """
    Create inference dataset that yields individual 2D slices prepared exactly
    like the training dataset.

    Modes:
      - nifti_path: path to a 4D NIfTI (T, Z, Y, X) -> flattens into (T,Z) slices
      - preprocessed_root: root patient folder containing patient/<patient_id>/data/*.npy

    Exactly one of nifti_path or preprocessed_root must be provided.
    """

    def __init__(
        self,
        nifti_path: Optional[Path] = None,
        preprocessed_root: Optional[Path] = None,
        patient_id: Optional[str] = None,
        target_h: Optional[int] = None,
        target_w: Optional[int] = None,
        pad_multiple: int = 16,
        augment = None,
    ):
        if (nifti_path is None) == (preprocessed_root is None):
            raise ValueError("Provide exactly one of nifti_path or preprocessed_root")

        self.nifti_path = Path(nifti_path) if nifti_path is not None else None
        self.preprocessed_root = Path(preprocessed_root) if preprocessed_root is not None else None
        self.patient_id = patient_id
        self.augment = augment
        self.pad_multiple = int(pad_multiple)

        # Storage for slices: for memory economy we store references + indices
        # samples: list of dicts with keys depending on mode
        self.samples: List[Dict[str, Any]] = []

        # Data loaded only when needed for nifti (keeps memory flexible)
        self._nifti_arr: Optional[np.ndarray] = None
        self._spacing: Optional[Tuple[float,float,float]] = None

        if self.nifti_path is not None:
            self._build_from_nifti(self.nifti_path)
        else:
            assert self.preprocessed_root is not None
            self._build_from_preprocessed(self.preprocessed_root, patient_id=self.patient_id)

        # Determine global target_h/target_w if not provided
        if target_h is None or target_w is None:
            # compute from first sample's shape (common in preprocessed repo)
            # fallback: compute max H,W across few samples
            max_h = 0
            max_w = 0
            inspect_n = min(200, len(self.samples))
            for s in self.samples[:inspect_n]:
                h,w = s["shape"]
                if h > max_h:
                    max_h = h
                if w > max_w:
                    max_w = w
            if max_h == 0 or max_w == 0:
                # worst case: set to pad_multiple
                max_h = self.pad_multiple
                max_w = self.pad_multiple
            # round up to pad_multiple
            m = self.pad_multiple
            target_h = ((max_h + m - 1) // m) * m
            target_w = ((max_w + m - 1) // m) * m

        self.target_h = int(target_h)
        self.target_w = int(target_w)

    # ---------------------------------------------
    # Builders
    # ---------------------------------------------
    def _build_from_nifti(self, nifti_path: Path):
        if not nifti_path.exists():
            raise FileNotFoundError(f"NIfTI not found: {nifti_path}")
        # load array (prefer SimpleITK)
        if _HAS_SITK:
            img = sitk.ReadImage(str(nifti_path))
            arr = sitk.GetArrayFromImage(img)  # expected (T, Z, Y, X)
            spacing = img.GetSpacing()  # (X, Y, Z) or similar
            # keep spacing for mesh extraction (caller may wish to use)
            self._spacing = (spacing[2], spacing[1], spacing[0]) if len(spacing) >= 3 else (1.0,1.0,1.0)
        elif _HAS_NIB:
            # nibabel returns array in (X,Y,Z,T) or similar; attempt to reorder
            nb = nib.load(str(nifti_path))
            arr_raw = nb.get_fdata()
            # try to get (T, Z, Y, X)
            if arr_raw.ndim == 4:
                # try common ordering -> (X,Y,Z,T) -> transpose
                arr = np.transpose(arr_raw, (3, 2, 1, 0))
            elif arr_raw.ndim == 3:
                # single volume: treat T=1
                arr = arr_raw[np.newaxis, ...]  # (1, Z, Y, X) if raw=(Z,Y,X)
            else:
                raise ValueError("Unsupported nibabel array shape: " + str(arr_raw.shape))
            self._spacing = tuple(getattr(nb.header, 'get_zooms', lambda: (1.0,1.0,1.0))()[:3])
        else:
            raise RuntimeError("Neither SimpleITK nor nibabel available to read NIfTI. Install SimpleITK.")

        # Basic validation
        if arr.ndim != 4:
            # try to coerce (T,Z,Y,X)
            raise ValueError(f"Expected 4D array (T,Z,Y,X) from nifti; got shape {arr.shape}")

        self._nifti_arr = arr.astype(np.float32)
        T, Z, Y, X = self._nifti_arr.shape

        # create sample index entries (no file path)
        for t in range(T):
            for z in range(Z):
                self.samples.append({
                    "mode": "nifti",
                    "patient_id": self.patient_id or self.nifti_path.stem.split("_")[0],
                    "t": int(t),
                    "z": int(z),
                    "shape": (int(Y), int(X)),
                    "image_path": None
                })

    def _build_from_preprocessed(self, root: Path, patient_id: Optional[str] = None):
        # Expect root to be either a patient folder or a root containing patient folders
        root = Path(root)
        patient_dirs = []
        if (root / "data").exists():
            # single patient folder
            patient_dirs = [root]
        else:
            # iterate patient subfolders
            patient_dirs = [p for p in root.iterdir() if p.is_dir()]

        for p in sorted(patient_dirs):
            pid = p.name
            data_dir = p / "data"
            if not data_dir.exists():
                continue
            image_files = sorted(data_dir.glob("*.npy"))
            for img_path in image_files:
                # assume filenames like t00_z00.npy or similar; we won't parse indices here
                arr = np.load(img_path)
                if arr.ndim == 2:
                    h,w = arr.shape
                else:
                    # try to take last two dims if shape has extra dims
                    h,w = int(arr.shape[-2]), int(arr.shape[-1])
                self.samples.append({
                    "mode": "preprocessed",
                    "patient_id": pid,
                    "t": None,
                    "z": None,
                    "shape": (h,w),
                    "image_path": img_path
                })

    # ---------------------------------------------
    # Dataset protocol
    # ---------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def _load_slice_array(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        s = self.samples[idx]
        if s["mode"] == "nifti":
            assert self._nifti_arr is not None
            t = s["t"]
            z = s["z"]
            arr = self._nifti_arr[t, z].astype(np.float32)  # (Y, X)
            meta = {
                "source": "nifti",
                "patient_id": s["patient_id"],
                "t": t,
                "z": z,
                "image_path": None,
                "orig_shape": s["shape"],
            }
            return arr, meta
        else:
            # preprocessed single-slice .npy
            img_path = s["image_path"]
            arr = np.load(img_path).astype(np.float32)
            meta = {
                "source": "preprocessed",
                "patient_id": s["patient_id"],
                "t": s.get("t"),
                "z": s.get("z"),
                "image_path": str(img_path),
                "orig_shape": s["shape"],
            }
            return arr, meta

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Return image_tensor (1,H_p,W_p) float32 and meta dict.
        """
        image, meta = self._load_slice_array(idx)

        # apply augmentations if provided (must accept image as HxW)
        if self.augment is not None:
            augmented = self.augment(image=image)
            image = augmented["image"]

        # pad to target
        padded = pad_to_target_2d(image, target_h=self.target_h, target_w=self.target_w, mode="constant", cval=0.0)
        # compute pad info (top,bottom,left,right)
        h, w = image.shape
        pad_top = 0
        pad_bottom = self.target_h - h
        pad_left = 0
        pad_right = self.target_w - w
        meta["pad"] = (pad_top, pad_bottom, pad_left, pad_right)

        # convert to channel-first tensor
        image_tensor = torch.from_numpy(np.expand_dims(padded.astype(np.float32), axis=0)).float()  # (1,H_p,W_p)

        return image_tensor, meta

    # helper: return original slice (unpadded) useful for direct visual comparison
    def get_original_slice(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Return (image_2d_numpy, meta). Useful for comparing to GT mask loaded from
        the training dataset manifest (before padding).
        """
        arr, meta = self._load_slice_array(idx)
        return arr, meta


# -------------------------
# small CLI example
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick test of InferenceDataset")
    parser.add_argument("--nifti", type=str, default=None, help="Path to 4D nifti (T,Z,Y,X)")
    parser.add_argument("--preprocessed", type=str, default=None, help="Path to preprocessed root or patient folder")
    parser.add_argument("--idx", type=int, default=0, help="sample index to print")
    args = parser.parse_args()

    ds = InferenceDataset(nifti_path=Path(args.nifti) if args.nifti else None,
                          preprocessed_root=Path(args.preprocessed) if args.preprocessed else None)
    img_t, meta = ds[args.idx]
    print("sample", args.idx, "tensor shape", img_t.shape, "meta", meta)
