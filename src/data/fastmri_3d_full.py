"""FastMRI-breast 3D data module for full dataset (including negatives).

Subclasses FastMRIBreast3DKSpaceDataModule to add:
- Loading of patient_labels.csv for per-patient metadata (has_lesion flag).
- Patient-level oversampling of positive patients to address the double
  imbalance (few positive patients AND few positive patches per positive patient).
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from .fastmri_3d import (
    FastMRIBreast3DKSpaceDataModule,
    FastMRIKspaceMaskAugmentDataset,
    FastMRIVolumePatchDataset2T,
)


class FastMRIBreast3DKSpaceFullDataModule(FastMRIBreast3DKSpaceDataModule):
    """Full-dataset variant with patient-level oversampling for positives."""

    def __init__(
        self,
        batch_size: int,
        dataset_dir: str,
        patch_depth: int = 16,
        patch_stride: int = 8,
        train_val_ratio: float = 0.8,
        oversample_positives_factor: int = 1,
        train_pos_fraction: Optional[float] = None,
        predefined_patient_ids: Optional[Dict[str, List[str]]] = None,
        augment_fastmri_mask_prob: float = 0.0,
        augment_accel_specs: Optional[List[Tuple[float, ...]]] = None,
        augment_mask_type: str = "random",
        patient_oversample_factor: int = 3,
        use_extra_timepoints: bool = True,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            dataset_dir=dataset_dir,
            patch_depth=patch_depth,
            patch_stride=patch_stride,
            train_val_ratio=train_val_ratio,
            oversample_positives_factor=oversample_positives_factor,
            train_pos_fraction=train_pos_fraction,
            predefined_patient_ids=predefined_patient_ids,
            augment_fastmri_mask_prob=augment_fastmri_mask_prob,
            augment_accel_specs=augment_accel_specs,
            augment_mask_type=augment_mask_type,
        )
        self.patient_oversample_factor = int(patient_oversample_factor)
        self.use_extra_timepoints = use_extra_timepoints
        self.patient_has_lesion: Dict[str, bool] = {}

    @property
    def name(self) -> str:
        return "FastMRIBreast3DFull"

    def _build_subject_list(self) -> None:
        super()._build_subject_list()
        # Load patient labels if available.
        csv_path = os.path.join(self.dataset_dir, "patient_labels.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.patient_has_lesion[row["patient_id"]] = int(row["has_lesion"]) > 0

    def _create_dataset(
        self,
        subjects_split: List[Dict[str, Any]],
        is_train: bool = False,
    ) -> Tuple[Dataset, Optional[List[float]]]:
        patch_samples: List[Dict[str, Any]] = []
        need_weights = is_train and (self.oversample_positives_factor > 1 or self.train_pos_fraction is not None)
        sample_weights: Optional[List[float]] = [] if need_weights else None
        is_pos_flags: Optional[List[bool]] = [] if need_weights else None
        patient_paths: Dict[str, Dict[str, str]] = {}
        temp_mmap_arrays: Dict[str, np.memmap] = {}

        # Determine which patients are positive for patient-level oversampling.
        positive_pids = set()
        for s in subjects_split:
            pid = s["patient_id"]
            if self.patient_has_lesion.get(pid, False):
                positive_pids.add(pid)

        for s in subjects_split:
            pid = s["patient_id"]

            patient_paths[pid] = {
                "input": s["input_path"],
                "target": s["target_path"],
                "label": s["label_path"],
            }

            if pid not in temp_mmap_arrays:
                temp_mmap_arrays[pid] = np.load(s["label_path"], mmap_mode="r")

            label_stack = temp_mmap_arrays[pid]
            S = label_stack.shape[0]
            z_positions = list(range(0, max(S - self.patch_depth + 1, 1), self.patch_stride))
            if not z_positions:
                z_positions = [0]
            if S > self.patch_depth:
                last = int(S - self.patch_depth)
                if z_positions[-1] != last:
                    z_positions.append(last)

            # How many times to repeat this patient's patches during training.
            # Positive patients are repeated patient_oversample_factor times.
            n_repeats = 1
            if is_train and pid in positive_pids and self.patient_oversample_factor > 1:
                n_repeats = self.patient_oversample_factor

            for _rep in range(n_repeats):
                for z0 in z_positions:
                    patch_samples.append({"patient_id": pid, "z_start": z0})
                    if is_train and sample_weights is not None:
                        zend = min(z0 + self.patch_depth, S)
                        patch_sum = float(np.sum(label_stack[z0:zend, ...]))
                        is_pos = patch_sum > 0.0
                        if is_pos_flags is not None:
                            is_pos_flags.append(bool(is_pos))
                        if self.train_pos_fraction is not None:
                            sample_weights.append(1.0)
                        else:
                            sample_weights.append(float(self.oversample_positives_factor) if is_pos else 1.0)

            # Training-only alternative timepoint pairs.
            if is_train and self.use_extra_timepoints:
                pdir = os.path.dirname(s["input_path"])
                extra_variants: List[Tuple[str, str]] = []

                path_t0t1 = os.path.join(pdir, "input_kspace_stack_t0t1.npy")
                if os.path.exists(path_t0t1):
                    extra_variants.append(("_t0t1", path_t0t1))

                path_t0t3 = os.path.join(pdir, "input_kspace_stack_t0t3.npy")
                if os.path.exists(path_t0t3):
                    extra_variants.append(("_t0t3", path_t0t3))

                for suffix, ipath in extra_variants:
                    alt_pid = f"{pid}{suffix}"
                    patient_paths[alt_pid] = {
                        "input": ipath,
                        "target": s["target_path"],
                        "label": s["label_path"],
                    }
                    for _rep in range(n_repeats):
                        for z0 in z_positions:
                            patch_samples.append({"patient_id": alt_pid, "z_start": z0})
                            if is_train and sample_weights is not None:
                                zend = min(z0 + self.patch_depth, S)
                                patch_sum = float(np.sum(label_stack[z0:zend, ...]))
                                is_pos = patch_sum > 0.0
                                if is_pos_flags is not None:
                                    is_pos_flags.append(bool(is_pos))
                                if self.train_pos_fraction is not None:
                                    sample_weights.append(1.0)
                                else:
                                    sample_weights.append(float(self.oversample_positives_factor) if is_pos else 1.0)

        # Set weights to achieve target positive patch fraction.
        if is_train and sample_weights is not None and self.train_pos_fraction is not None and is_pos_flags is not None:
            pos = int(sum(is_pos_flags))
            neg = int(len(is_pos_flags) - pos)
            if pos > 0 and neg > 0:
                p = float(self.train_pos_fraction)
                w_pos = p / pos
                w_neg = (1.0 - p) / neg
                sample_weights = [w_pos if flag else w_neg for flag in is_pos_flags]

        dataset = FastMRIVolumePatchDataset2T(
            patch_samples,
            patient_paths,
            self.patient_id_to_index,
            self.patch_depth,
        )
        if is_train and self.augment_fastmri_mask_prob > 0.0:
            dataset = FastMRIKspaceMaskAugmentDataset(  # type: ignore[assignment]
                dataset,
                prob=self.augment_fastmri_mask_prob,
                accel_specs=self.augment_accel_specs,
                mask_type=self.augment_mask_type,
            )
        return dataset, sample_weights
