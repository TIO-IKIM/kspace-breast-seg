from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .kspace3d import KspaceVolumePatchDatasetBase, Kspace3DPatchDataModuleBase


class MamaMIAKspaceMaskAugmentDataset(Dataset):
    """Training-only fastMRI-style 1D undersampling on k-space input.

    Applies RandomMaskFunc + apply_mask with probability `prob`.
    Uses (accel, center_fraction) in {(2,0.04), (4,0.08)}.
    """

    def __init__(self, base: Dataset, prob: float = 0.0) -> None:
        self.base = base
        self.prob = float(prob)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        if self.prob <= 0.0:
            return sample
        if float(torch.rand(()).item()) >= self.prob:
            return sample

        from fastmri.data.subsample import RandomMaskFunc
        from fastmri.data.transforms import apply_mask

        x = sample["input"]["data"]  # (C=1, V=2, D, H, W)
        accel = 2 if float(torch.rand(()).item()) < 0.5 else 4
        center_frac = 0.04 if accel == 2 else 0.08
        mask_func = RandomMaskFunc([float(center_frac)], [int(accel)])
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        # fastMRI apply_mask expects complex in last dim: (..., H, W, 2)
        x5 = x.permute(0, 2, 3, 4, 1).contiguous()  # (C, D, H, W, 2)
        x5_masked, _, _ = apply_mask(x5, mask_func, seed=seed)
        sample["input"]["data"] = x5_masked.permute(0, 4, 1, 2, 3).contiguous()
        return sample


class MamaMIAVolumePatchDataset(KspaceVolumePatchDatasetBase):
    """3D MAMA-MIA k-space volume patches.

    Expects per-patient stacks on disk with shapes:
      - input_kspace_stack.npy:  (S, 2, H, W)
      - target_kspace_stack.npy: (S, 2, H, W)  foreground (real, imag)
      - label_mask_stack.npy:    (S, H, W)     ordinal mask

    For each patch of depth D this yields:
      - input: (1, 2, D, H, W)
      - label: (1, 2, D, H, W)  fg-only (real, imag); loss expands to bg+fg on GPU
      - label_mask: (D, H, W)
    """

    def _build_example(
        self,
        in_patch: np.ndarray,
        tgt_patch_flat: np.ndarray,
        lbl_patch: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_k = torch.from_numpy(in_patch).permute(1, 0, 2, 3).contiguous()  # (2, D, H, W)
        input_k = input_k.unsqueeze(0)  # (1, 2, D, H, W)

        # Return fg-only k-space target (1, 2, D, H, W).
        # The background channel is synthesized by the loss function on GPU.
        fr = torch.from_numpy(tgt_patch_flat[:, 0, ...])
        fi = torch.from_numpy(tgt_patch_flat[:, 1, ...])
        target_k = torch.stack([fr, fi], dim=0).unsqueeze(0)  # (1, 2, D, H, W)

        label_mask = torch.from_numpy(lbl_patch)  # (D, H, W)
        return input_k, target_k, label_mask


class MamaMIA3DKSpaceDataModule(Kspace3DPatchDataModuleBase):
    """3D k-space patch DataModule for MAMA-MIA.

    Uses precomputed stacks from `datasets/mamamia/prepare_syn.py`:
      - input_kspace_stack.npy
      - target_kspace_stack.npy
      - label_mask_stack.npy
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
        augment_fastmri_mask_prob: float = 0.0,
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
        )
        self._spatial_hw: Optional[Tuple[int, int]] = None
        self.augment_fastmri_mask_prob = float(augment_fastmri_mask_prob)

    # --- Properties ---

    @property
    def name(self) -> str:
        return "MamaMIA3D"

    @property
    def input_shape(self) -> Tuple[int, int, int, int, int]:
        # (C_in=1, V=2, D, H, W)
        h, w = self._spatial_hw if self._spatial_hw is not None else (384, 384)
        return (1, 2, self.patch_depth, h, w)

    @property
    def label_shape(self) -> Tuple[int, int, int, int, int]:
        # fg-only k-space target: (C_out=1, V=2, D, H, W).
        # Loss functions synthesize the background channel on GPU when needed.
        h, w = self._spatial_hw if self._spatial_hw is not None else (384, 384)
        return (1, 2, self.patch_depth, h, w)

    # --- Subject list and dataset creation ---

    def _build_subject_list(self) -> None:
        if self.subject_list:
            return

        patient_ids = sorted(
            [
                d
                for d in os.listdir(self.dataset_dir)
                if os.path.isdir(os.path.join(self.dataset_dir, d))
            ]
        )
        self.patient_id_to_index = {pid: idx for idx, pid in enumerate(patient_ids)}

        for patient_id in patient_ids:
            pdir = os.path.join(self.dataset_dir, patient_id)
            ipath = os.path.join(pdir, "input_kspace_stack.npy")
            tpath = os.path.join(pdir, "target_kspace_stack.npy")
            lpath = os.path.join(pdir, "label_mask_stack.npy")
            if os.path.exists(ipath) and os.path.exists(tpath) and os.path.exists(lpath):
                self.subject_list.append(
                    {
                        "patient_id": patient_id,
                        "input_path": ipath,
                        "target_path": tpath,
                        "label_path": lpath,
                    }
                )

        # Derive H, W once
        if self.subject_list:
            arr = np.load(self.subject_list[0]["input_path"], mmap_mode="r")
            self._spatial_hw = (int(arr.shape[-2]), int(arr.shape[-1]))

    def _create_dataset(
        self,
        subjects_split: List[Dict[str, Any]],
        is_train: bool = False,
    ) -> Tuple[MamaMIAVolumePatchDataset, Optional[List[float]]]:
        patch_samples: List[Dict[str, Any]] = []
        need_weights = is_train and (self.oversample_positives_factor > 1 or self.train_pos_fraction is not None)
        sample_weights: Optional[List[float]] = ([] if need_weights else None)
        is_pos_flags: Optional[List[bool]] = ([] if need_weights else None)
        patient_paths: Dict[str, Dict[str, str]] = {}
        temp_mmap_arrays: Dict[str, np.memmap] = {}

        for s in subjects_split:
            pid = s["patient_id"]

            patient_paths[pid] = {
                "input": s["input_path"],
                "target": s["target_path"],
                "label": s["label_path"],
            }

            if pid not in temp_mmap_arrays:
                temp_mmap_arrays[pid] = np.load(s["label_path"], mmap_mode="r")

            label_stack = temp_mmap_arrays[pid]  # (S, H, W)
            S = label_stack.shape[0]
            z_positions = list(range(0, max(S - self.patch_depth + 1, 1), self.patch_stride))
            if not z_positions:
                z_positions = [0]
            # Ensure tail coverage (include the last valid start) so eval can cover the full volume.
            if S > self.patch_depth:
                last = int(S - self.patch_depth)
                if z_positions[-1] != last:
                    z_positions.append(last)

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

        # If requested, set weights to achieve a target positive patch fraction.
        if is_train and sample_weights is not None and self.train_pos_fraction is not None and is_pos_flags is not None:
            pos = int(sum(is_pos_flags))
            neg = int(len(is_pos_flags) - pos)
            if pos > 0 and neg > 0:
                p = float(self.train_pos_fraction)
                w_pos = p / pos
                w_neg = (1.0 - p) / neg
                sample_weights = [w_pos if flag else w_neg for flag in is_pos_flags]

        dataset = MamaMIAVolumePatchDataset(
            patch_samples,
            patient_paths,
            self.patient_id_to_index,
            self.patch_depth,
        )
        if is_train and self.augment_fastmri_mask_prob > 0.0:
            dataset = MamaMIAKspaceMaskAugmentDataset(dataset, prob=self.augment_fastmri_mask_prob)
        return dataset, sample_weights
