#!/usr/bin/env python
"""Prepare synthetic (image-space) k-space training data for ALL fastMRI-breast patients.

Mirrors `prepare_kspace_full.py` byte-for-byte (per-patient t=0 RMS scaling,
extra-timepoint pairs, zero-mask fallback, output layout) — the ONLY
difference is the k-space source: instead of reading complex k-space from
`*_kspace.h5`, the baseline and difference k-spaces are obtained by per-slice
2D FFT of the image-space `temptv` reconstructions in `*_2.h5`.
"""
import os
import sys

sys.path.insert(0, "/home/l721f/code/kspace-pred-net")

import csv
import glob
import h5py
import SimpleITK as sitk
import numpy as np
import numpy.fft as fft
import pandas as pd
from tqdm import tqdm


RAW_DIR = "/home/l721f/data/fastmri-breast/rawdata"
MASK_DIR = "/home/l721f/data/fastmri-breast/masks"

OUT_BASE = "/home/l721f/data/fastmri-breast/training_data_syn_full"

LABELS_XLSX = "/home/l721f/data/fastmri-breast/fastMRI_breast_labels_short.xlsx"
STATUS_KEEP = {0, 1, 2}

RAW_PATTERN = "*_2.h5"

labels_df = pd.read_excel(LABELS_XLSX)
lesion_map = dict(zip(labels_df["patient_name"], labels_df["lesion_status"]))

EXTRA_TIMEPOINT_KEYS = ("t0t1", "t0t3")
WRITE_EXTRA_TIMEPOINTS = True


def _already_done(patient_id: str) -> bool:
    out_dir = os.path.join(OUT_BASE, patient_id)
    if not os.path.isdir(out_dir):
        return False
    req = [
        "input_kspace_stack.npy",
        "target_kspace_stack.npy",
        "label_mask_stack.npy",
        "kspace_rms_scale.npy",
    ]
    if WRITE_EXTRA_TIMEPOINTS:
        req.extend([f"input_kspace_stack_{k}.npy" for k in EXTRA_TIMEPOINT_KEYS])
    return all(os.path.exists(os.path.join(out_dir, f)) for f in req)


def _centered_fft2(img_hw: np.ndarray) -> np.ndarray:
    """Centered 2D FFT of a real-valued (H, W) slice."""
    img0 = fft.ifftshift(img_hw, axes=(-2, -1))
    k = fft.fftshift(fft.fft2(img0, norm="ortho"), axes=(-2, -1))
    return k.astype(np.complex64, copy=False)


def _image_volume_to_kspace(img_sthw: np.ndarray) -> np.ndarray:
    """Per-slice 2D FFT of a (S, T, H, W) real image volume → (S, T, H, W) complex."""
    s, t, h, w = img_sthw.shape
    out = np.empty((s, t, h, w), dtype=np.complex64)
    for i in range(s):
        for j in range(t):
            out[i, j] = _centered_fft2(img_sthw[i, j].astype(np.float32, copy=False))
    return out


def save_case(
    patient_id: str,
    kspace_all_t: np.ndarray,
    mask_dhw: np.ndarray,
) -> None:
    """Process and save k-space data with per-patient baseline (t0) RMS scaling.

    Mirrors `prepare_kspace_full.py::save_case`. Does NOT skip cases with empty masks.
    """
    s, t, h, w = kspace_all_t.shape

    def make_pair_input(kspace_st: np.ndarray, t_ref: int, t_other: int) -> np.ndarray:
        k_diff = kspace_st[:, t_other] - kspace_st[:, t_ref]
        return np.stack([np.real(k_diff), np.imag(k_diff)], axis=1).astype(np.float32, copy=False)

    scale_factor = float(np.sqrt(np.mean(np.abs(kspace_all_t[:, 0]) ** 2, dtype=np.float64)))
    scale_factor = scale_factor if scale_factor > 0.0 else 1.0
    kspace_all_t = kspace_all_t / scale_factor

    input_stack_t0t2 = make_pair_input(kspace_all_t, 0, 2)

    extra_inputs: dict[str, np.ndarray] = {}
    if WRITE_EXTRA_TIMEPOINTS:
        if t > 1:
            extra_inputs["t0t1"] = make_pair_input(kspace_all_t, 0, 1)
        if t > 3:
            extra_inputs["t0t3"] = make_pair_input(kspace_all_t, 0, 3)

    mask_bin = (mask_dhw > 0).astype(np.uint8, copy=False)

    # Build foreground k-space target (zeros if mask is all-zero).
    onehot = np.zeros((s, 2, h, w), dtype=np.float32)
    onehot[:, 0] = (mask_bin == 0)
    onehot[:, 1] = (mask_bin == 1)
    onehot0 = fft.ifftshift(onehot, axes=(-2, -1))
    k_target = fft.fftshift(fft.fft2(onehot0, axes=(-2, -1), norm="ortho"), axes=(-2, -1))
    k_fg = np.stack([np.real(k_target[:, 1]), np.imag(k_target[:, 1])], axis=1).astype(np.float32, copy=False)

    out_dir = os.path.join(OUT_BASE, patient_id)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "input_kspace_stack.npy"), input_stack_t0t2)
    for key, arr in extra_inputs.items():
        np.save(os.path.join(out_dir, f"input_kspace_stack_{key}.npy"), arr)
    np.save(os.path.join(out_dir, "target_kspace_stack.npy"), k_fg)
    np.save(os.path.join(out_dir, "label_mask_stack.npy"), mask_bin.astype(np.uint8, copy=False))
    np.save(os.path.join(out_dir, "kspace_rms_scale.npy"), np.array([scale_factor], dtype=np.float32))


def read_mask_nii(mask_path: str) -> np.ndarray:
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)
    return arr.astype(np.uint8, copy=False)


def main() -> None:
    os.makedirs(OUT_BASE, exist_ok=True)
    h5_files = sorted(glob.glob(os.path.join(RAW_DIR, RAW_PATTERN)))

    label_rows: list[tuple[str, int, int]] = []

    for h5_file in tqdm(h5_files, desc="fastMRI-imgspace-preproc-full"):
        base = os.path.splitext(os.path.basename(h5_file))[0]
        patient_id = "_".join(base.split("_")[:-1])

        if patient_id not in lesion_map:
            continue
        lesion_status = int(lesion_map[patient_id])
        if lesion_status not in STATUS_KEEP:
            continue

        with h5py.File(h5_file, "r", swmr=True) as f:
            img = f["temptv"][:]  # (S, T, H, W) real
        img = img.astype(np.float32, copy=False)

        # Synthesize complex k-space from the image volume (per-slice 2D FFT).
        k = _image_volume_to_kspace(img)

        # Load mask; substitute all-zeros on mismatch or absence.
        mask_path = os.path.join(MASK_DIR, f"{patient_id}_mask.nii")
        if os.path.exists(mask_path):
            mask_dhw = read_mask_nii(mask_path)
            if mask_dhw.shape[0] != k.shape[0]:
                mask_dhw = np.zeros((k.shape[0], k.shape[2], k.shape[3]), dtype=np.uint8)
        else:
            mask_dhw = np.zeros((k.shape[0], k.shape[2], k.shape[3]), dtype=np.uint8)

        has_lesion = int(np.count_nonzero(mask_dhw) > 0)
        label_rows.append((patient_id, lesion_status, has_lesion))

        if _already_done(patient_id):
            continue

        save_case(patient_id, k, mask_dhw)

    csv_path = os.path.join(OUT_BASE, "patient_labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "lesion_status", "has_lesion"])
        for row in sorted(label_rows):
            writer.writerow(row)
    print(f"Wrote {len(label_rows)} patient labels to {csv_path}")


if __name__ == "__main__":
    main()
