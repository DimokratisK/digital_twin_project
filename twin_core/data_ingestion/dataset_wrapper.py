# twin_core/data_ingestion/dataset_wrapper.py
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from twin_core.data_ingestion.dataset import CardiacDataset  # adapt import if dataset.py moves
import torch
import albumentations as A

class CardiacDatasetWithMeta(CardiacDataset):
    """
    Thin adapter around CardiacDataset that exposes metadata and robust mask checks.
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
        # Forward core dataset args to the base implementation
        super().__init__(
            root=root,
            augment=augment,
            prefer_ed_es=prefer_ed_es,
            one_hot=one_hot,
            n_classes=n_classes,
            exclude_missing_masks=exclude_missing_masks,
            pad_multiple=pad_multiple,
        )

        # Adapter-specific state
        self.metadata_index = metadata_index or {}

        # Ensure helper caches exist (base class should have created these)
        # If base class already sets these, these lines are harmless.
        self._sample_to_patient = [s[0] for s in self.samples]
        self.target_h, self.target_w = self._compute_global_target_size()
