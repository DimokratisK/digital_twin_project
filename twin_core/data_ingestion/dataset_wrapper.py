# twin_core/data_ingestion/dataset_wrapper.py
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from twin_core.data_ingestion.dataset import CardiacDataset  # adapt import if dataset.py moves
import torch

class CardiacDatasetWithMeta(CardiacDataset):
    """
    Thin adapter around CardiacDataset that exposes metadata and robust mask checks.

    Inherits all behavior from CardiacDataset. Adds:
      - sample_meta(idx) -> dict
      - has_mask(idx, check_nonzero=False) -> bool
      - indices_for_patient(patient_id) -> List[int]
      - patient_from_index(idx) -> str
    """

    def sample_meta(self, idx: int) -> Dict[str, Any]:
        """
        Return metadata for a sample index.
        Uses the internal `self.samples` structure from CardiacDataset:
          self.samples[idx] == (patient_id, t_idx, z_idx, image_path, mask_path_or_None)
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.samples)}")
        pid, t_idx, z_idx, img_path, mask_path = self.samples[idx]
        meta = self.metadata_index.get(pid, {}) if isinstance(self.metadata_index, dict) else {}
        return {
            "patient_id": pid,
            "t": int(t_idx),
            "z": int(z_idx),
            "image_path": str(img_path),
            "mask_path": str(mask_path) if mask_path is not None else None,
            "metadata": meta,
        }

    def has_mask(self, idx: int, check_nonzero: bool = False) -> bool:
        """
        Return True if a mask file exists for this sample.
        If check_nonzero=True, load the .npy mask and return True only if it contains any nonzero voxels.
        """
        if idx < 0 or idx >= len(self.samples):
            return False
        mask_path = self.samples[idx][4]
        if mask_path is None:
            return False
        if not check_nonzero:
            return True
        try:
            arr = np.load(mask_path)
            return int(np.count_nonzero(arr)) > 0
        except Exception:
            # treat unreadable files as no mask
            return False

    def indices_for_patient(self, patient_id: str) -> List[int]:
        """Return dataset indices that belong to the given patient_id."""
        return [i for i, s in enumerate(self.samples) if s[0] == patient_id]

    def patient_from_index(self, idx: int) -> str:
        """Return patient id for a given sample index."""
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")
        return self.samples[idx][0]

    # Optional convenience: count labeled samples quickly using manifest-like check
    def count_labeled_samples(self, check_nonzero: bool = False) -> int:
        """Return number of samples with masks (or nonzero masks if check_nonzero=True)."""
        if not check_nonzero:
            return sum(1 for s in self.samples if s[4] is not None)
        cnt = 0
        for _, _, _, _, mask_path in self.samples:
            if mask_path is None:
                continue
            try:
                if np.count_nonzero(np.load(mask_path)) > 0:
                    cnt += 1
            except Exception:
                continue
        return cnt
