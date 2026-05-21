from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl


class KspaceVolumePatchDatasetBase(Dataset):
    """Base dataset for 3D k-space patches.

    Expects patient-wise stacks on disk:
      - input: (S, C_in_flat, H, W)
      - target: (S, C_tgt_flat, H, W)
      - label: (S, H, W)

    Subclasses implement `_build_example` to map per-patch numpy arrays to
    PyTorch tensors with the desired channel layout.
    """

    def __init__(
        self,
        patch_samples: List[Dict[str, Any]],
        patient_paths: Dict[str, Dict[str, str]],
        patient_id_to_index: Dict[str, int],
        patch_depth: int,
    ) -> None:
        self.patch_samples = patch_samples
        self.patient_paths = patient_paths
        self.patient_id_to_index = patient_id_to_index
        self.patch_depth = int(patch_depth)
        self._worker_mmap_cache: Dict[str, Tuple[np.memmap, np.memmap, np.memmap]] = {}

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.patch_samples)

    def _get_mmap_arrays(self, patient_id: str) -> Tuple[np.memmap, np.memmap, np.memmap]:
        """Get memory-mapped (input, target, label) stacks for a patient.

        Each array has shapes:
          - input_stack:  (S, C_in_flat, H, W)
          - target_stack: (S, C_tgt_flat, H, W)
          - label_stack:  (S, H, W)
        """
        if patient_id not in self._worker_mmap_cache:
            paths = self.patient_paths[patient_id]
            self._worker_mmap_cache[patient_id] = (
                np.load(paths["input"], mmap_mode="r"),
                np.load(paths["target"], mmap_mode="r"),
                np.load(paths["label"], mmap_mode="r"),
            )
        return self._worker_mmap_cache[patient_id]

    def _pad_patches(
        self,
        in_patch: np.ndarray,
        tgt_patch_flat: np.ndarray,
        lbl_patch: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pad along the slice dimension to fixed `patch_depth` by replication."""
        d = in_patch.shape[0]
        if d >= self.patch_depth:
            return in_patch, tgt_patch_flat, lbl_patch

        pad_d = self.patch_depth - d
        in_pad = np.repeat(in_patch[-1:, ...], pad_d, axis=0)
        tgt_pad = np.repeat(tgt_patch_flat[-1:, ...], pad_d, axis=0)
        lbl_pad = np.repeat(lbl_patch[-1:, ...], pad_d, axis=0)
        in_patch = np.concatenate([in_patch, in_pad], axis=0)
        tgt_patch_flat = np.concatenate([tgt_patch_flat, tgt_pad], axis=0)
        lbl_patch = np.concatenate([lbl_patch, lbl_pad], axis=0)
        return in_patch, tgt_patch_flat, lbl_patch

    def _build_example(
        self,
        in_patch: np.ndarray,
        tgt_patch_flat: np.ndarray,
        lbl_patch: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert numpy patches to tensors.

        Subclasses must implement this to return `(input_k, target_k, label_mask)`.
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:  # type: ignore[override]
        info = self.patch_samples[idx]
        patient_id: str = info["patient_id"]
        z_start: int = int(info["z_start"])
        patient_index = int(self.patient_id_to_index.get(patient_id, -1))

        input_stack, target_stack_flat, label_stack = self._get_mmap_arrays(patient_id)
        s_total = input_stack.shape[0]
        z_end = min(z_start + self.patch_depth, s_total)

        in_patch = np.array(input_stack[z_start:z_end], copy=True)
        tgt_patch_flat = np.array(target_stack_flat[z_start:z_end], copy=True)
        lbl_patch = np.array(label_stack[z_start:z_end], copy=True)

        in_patch, tgt_patch_flat, lbl_patch = self._pad_patches(in_patch, tgt_patch_flat, lbl_patch)

        input_k, target_k, label_mask = self._build_example(in_patch, tgt_patch_flat, lbl_patch)

        return {
            "input": {"data": input_k},
            "label": {"data": target_k},
            "label_mask": {"data": label_mask},
            "meta": {
                "patient_index": torch.tensor(patient_index, dtype=torch.int32),
                "z_start": torch.tensor(z_start, dtype=torch.int32),
                "s_total": torch.tensor(int(s_total), dtype=torch.int32),
            },
        }


class Kspace3DPatchDataModuleBase(pl.LightningDataModule):
    """Base LightningDataModule for 3D k-space patch datasets.

    Subclasses implement:
      - `_build_subject_list` to populate `self.subject_list` and
        `self.patient_id_to_index`.
      - `_create_dataset` to turn a subject split into `(dataset, sample_weights)`.

    This base handles patient-level splitting, optional predefined splits,
    oversampling of positive patches, and DataLoader construction.
    """

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
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        self.dataset_dir = dataset_dir
        self.patch_depth = int(patch_depth)
        self.patch_stride = int(patch_stride)
        self.train_val_ratio = float(train_val_ratio)
        self.oversample_positives_factor = int(oversample_positives_factor)
        self.train_pos_fraction = train_pos_fraction
        self.predefined_patient_ids = predefined_patient_ids

        self.subject_list: List[Dict[str, Any]] = []
        self.patient_id_to_index: Dict[str, int] = {}
        self.train_sampler: Optional[WeightedRandomSampler] = None

    # --- Properties that subclasses must define ---

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def input_shape(self) -> Tuple[int, int, int, int, int]:
        raise NotImplementedError

    @property
    def label_shape(self) -> Tuple[int, int, int, int, int]:
        raise NotImplementedError

    @property
    def input_domain(self) -> str:
        return "kspace"

    @property
    def label_domain(self) -> str:
        return "kspace"

    # --- Hooks for subclasses ---

    def _build_subject_list(self) -> None:
        """Populate `self.subject_list` and `self.patient_id_to_index`.

        Each subject entry should be a dict with keys:
          - 'patient_id'
          - 'input_path'
          - 'target_path'
          - 'label_path'
        """
        raise NotImplementedError

    def _create_dataset(
        self,
        subjects_split: List[Dict[str, Any]],
        is_train: bool = False,
    ) -> Tuple[Dataset, Optional[List[float]]]:
        """Create dataset and optional per-sample weights from a subject split."""
        raise NotImplementedError

    # --- Lightning hooks ---

    def prepare_data(self) -> None:  # type: ignore[override]
        # Keep empty to avoid duplicating work across ranks in DDP.
        pass

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        # Build subject list once so all ranks see the same patients.
        if not self.subject_list:
            self._build_subject_list()

        if self.predefined_patient_ids is not None and isinstance(self.predefined_patient_ids, dict):
            pid_to_entry = {s["patient_id"]: s for s in self.subject_list}
            train_pids = self.predefined_patient_ids.get("train", [])
            val_pids = self.predefined_patient_ids.get("val", [])
            test_pids = self.predefined_patient_ids.get("test", val_pids)

            train_list = [pid_to_entry[pid] for pid in train_pids if pid in pid_to_entry]
            val_list = [pid_to_entry[pid] for pid in val_pids if pid in pid_to_entry]
            test_list = [pid_to_entry[pid] for pid in test_pids if pid in pid_to_entry]
        else:
            num_subjects = len(self.subject_list)
            if num_subjects == 0:
                self.train_set = []  # type: ignore[attr-defined]
                self.val_set = []  # type: ignore[attr-defined]
                self.test_set = []  # type: ignore[attr-defined]
                return

            num_train = int(num_subjects * self.train_val_ratio)
            num_val = max(1, (num_subjects - num_train) // 2)
            num_test = max(1, num_subjects - num_train - num_val)
            num_train = num_subjects - num_val - num_test

            train_subjects, val_subjects, test_subjects = torch.utils.data.random_split(  # type: ignore[name-defined]
                self.subject_list,
                [num_train, num_val, num_test],
                torch.Generator().manual_seed(123),  # type: ignore[name-defined]
            )
            train_list = [self.subject_list[i] for i in train_subjects.indices]
            val_list = [self.subject_list[i] for i in val_subjects.indices]
            test_list = [self.subject_list[i] for i in test_subjects.indices]

        self.train_set, train_weights = self._create_dataset(train_list, is_train=True)
        self.val_set, _ = self._create_dataset(val_list, is_train=False)
        self.test_set, _ = self._create_dataset(test_list, is_train=False)

        if (self.oversample_positives_factor > 1 or self.train_pos_fraction is not None) and train_weights:
            self.train_sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(train_weights),  # type: ignore[arg-type]
                num_samples=len(train_weights),
                replacement=True,
            )

    # --- Dataloaders ---

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        shuffle = self.train_sampler is None
        return DataLoader(
            self.train_set,  # type: ignore[attr-defined]
            batch_size=self.batch_size,
            num_workers=8,
            sampler=self.train_sampler,
            shuffle=shuffle,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=6,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        return DataLoader(
            self.val_set,  # type: ignore[attr-defined]
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=6,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        return DataLoader(
            self.test_set,  # type: ignore[attr-defined]
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=6,
        )
