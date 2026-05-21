from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .kspace3d import KspaceVolumePatchDatasetBase, Kspace3DPatchDataModuleBase


class FastMRIKspaceMaskAugmentDataset(Dataset):
    """Training-only fastMRI-style 1D undersampling on k-space input.

    Applies `RandomMaskFunc` or `EquiSpacedMaskFunc` + `apply_mask` with
    probability `prob`. Acceleration is sampled from `accel_specs`, which is a
    list of either (accel, center_fraction) or (accel, center_fraction, weight)
    tuples. Missing weights default to uniform. Defaults to 50/50 {2x, 4x} with
    RandomMaskFunc for backward compatibility.
    """

    DEFAULT_ACCEL_SPECS = [
        (2, 0.04, 0.5),
        (4, 0.08, 0.5),
    ]

    def __init__(
        self,
        base: Dataset,
        prob: float = 0.0,
        accel_specs: Optional[List[Tuple[float, ...]]] = None,
        mask_type: str = "random",
    ) -> None:
        self.base = base
        self.prob = float(prob)
        specs = accel_specs if accel_specs is not None else self.DEFAULT_ACCEL_SPECS
        self._accels = np.array([s[0] for s in specs], dtype=np.int64)
        self._center_fracs = np.array([s[1] for s in specs], dtype=np.float64)
        weights = np.array(
            [s[2] if len(s) > 2 else 1.0 for s in specs], dtype=np.float64
        )
        self._weights = weights / weights.sum()
        self._mask_type = mask_type

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        sample = self.base[idx]
        if self.prob <= 0.0:
            return sample
        if float(torch.rand(()).item()) >= self.prob:
            return sample

        # Lazy import to keep baseline usage light.
        from fastmri.data.subsample import EquiSpacedMaskFunc, RandomMaskFunc
        from fastmri.data.transforms import apply_mask

        x = sample["input"]["data"]  # (C=1, V=2, D, H, W)
        i = np.random.choice(len(self._accels), p=self._weights)
        accel = int(self._accels[i])
        center_frac = float(self._center_fracs[i])
        if self._mask_type == "equispaced":
            mask_func = EquiSpacedMaskFunc([center_frac], [accel])
        else:
            mask_func = RandomMaskFunc([center_frac], [accel])
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        # fastMRI apply_mask expects complex in last dim: (..., H, W, 2)
        x5 = x.permute(0, 2, 3, 4, 1).contiguous()  # (C, D, H, W, 2)
        x5_masked, _, _ = apply_mask(x5, mask_func, seed=seed)
        sample["input"]["data"] = x5_masked.permute(0, 4, 1, 2, 3).contiguous()
        return sample


class FastMRIVolumePatchDataset2T(KspaceVolumePatchDatasetBase):
    """FastMRI-breast 3D k-space volume dataset using a single complex subtraction.

    Expects per-patient stacks on disk with shapes:
      - input_kspace_stack.npy:  (S, 2, H, W)
          [Re(t_other - t_ref), Im(t_other - t_ref)]
      - target_kspace_stack.npy: (S, 2, H, W) foreground k-space (real, imag)
      - label_mask_stack.npy:    (S, H, W) ordinal mask

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
        d, c_flat, h, w = in_patch.shape
        if c_flat != 2:
            raise ValueError(f"Expected 2-channel input (Re,Im) subtraction, got shape {in_patch.shape}")

        in_reshaped = in_patch.reshape(d, 1, 2, h, w)
        input_k = torch.from_numpy(in_reshaped).permute(1, 2, 0, 3, 4).contiguous()  # (1, 2, D, H, W)

        # Return fg-only k-space target (1, 2, D, H, W).
        # The background channel is synthesized by the loss function on GPU.
        fr = torch.from_numpy(tgt_patch_flat[:, 0, ...])
        fi = torch.from_numpy(tgt_patch_flat[:, 1, ...])
        target_k = torch.stack([fr, fi], dim=0).unsqueeze(0)  # (1, 2, D, H, W)

        label_mask = torch.from_numpy(lbl_patch)  # (D, H, W)
        return input_k, target_k, label_mask


class FastMRIBreast3DKSpaceDataModule(Kspace3DPatchDataModuleBase):
    """FastMRI-breast 3D k-space patch DataModule.

    Uses precomputed stacks from `datasets/fastMRI/prepare_kspace_slices.py`:
      - input_kspace_stack.npy
      - target_kspace_stack.npy
      - label_mask_stack.npy

    During training, additional pseudo-volumes based on alternative timepoint
    pairs (e.g. t0t1, t0t3) are added when the corresponding stacks exist.
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
        augment_accel_specs: Optional[List[Tuple[float, ...]]] = None,
        augment_mask_type: str = "random",
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
        self.augment_accel_specs = augment_accel_specs
        self.augment_mask_type = augment_mask_type

    # --- Properties ---

    @property
    def name(self) -> str:
        return "FastMRIBreast3D"

    @property
    def input_shape(self) -> Tuple[int, int, int, int, int]:
        # Single complex subtraction volume (e.g. t2 - t0): C_in=1, V=2
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

        if self.subject_list:
            arr = np.load(self.subject_list[0]["input_path"], mmap_mode="r")
            self._spatial_hw = (int(arr.shape[-2]), int(arr.shape[-1]))

    def _create_dataset(
        self,
        subjects_split: List[Dict[str, Any]],
        is_train: bool = False,
    ) -> Tuple[FastMRIVolumePatchDataset2T, Optional[List[float]]]:
        patch_samples: List[Dict[str, Any]] = []
        need_weights = is_train and (self.oversample_positives_factor > 1 or self.train_pos_fraction is not None)
        sample_weights: Optional[List[float]] = ([] if need_weights else None)
        is_pos_flags: Optional[List[bool]] = ([] if need_weights else None)
        patient_paths: Dict[str, Dict[str, str]] = {}
        temp_mmap_arrays: Dict[str, np.memmap] = {}

        for s in subjects_split:
            pid = s["patient_id"]

            # Base paths (t0+t2) so each worker can open its own memmaps
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

            # Canonical (t0,t2) volume, used also for val/test
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

            # During training only, add extra pseudo-volumes based on alternative
            # timepoint pairs (t0,t1), (t0,t3) if the corresponding stacks exist.
            if is_train:
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

        # If requested, set weights to achieve a target positive patch fraction.
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
