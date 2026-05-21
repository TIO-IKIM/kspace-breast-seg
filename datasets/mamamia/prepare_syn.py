#!/usr/bin/env python
"""
Preprocessing pipeline for DUKE_* patients (2D Slice K-space Target version).

This script iterates over all DUKE_* directories in the source folders.
For each patient:
1. Loads the DCE-MRI phases: pre-contrast (phase 0), first post-contrast (phase 1), and second post-contrast (phase 2), and the segmentation mask.
2. Resamples all phases and the segmentation to 1x1x1 mm isotropic spacing.
3. Computes per-patient DCE-wide z-score statistics (mean and std) across all available phases (pre → last post), then z-scores each phase with those same statistics.
4. Forms a subtraction volume as: last post-contrast (phase 2) z-scored minus pre-contrast (phase 0) z-scored.
5. Iterates through axial slices (Depth dimension).
6. For each slice:
    a. Pads/crops to the target spatial size.
    b. Calculates the 2D FFT of the subtraction slice -> Input K-space.
    c. Calculates the 2D FFT of the (padded/cropped) binary segmentation mask slice (and builds one-hot target in k-space).
7. Stacks the results for all slices.
8. Saves the following for each patient:
    - /path/to/output/DUKE_XXX/input_kspace_stack.npy  (S, 2, H, W) float16
    - /path/to/output/DUKE_XXX/target_kspace_stack.npy (S, 2, H, W) float16 where channels are (fg_real, fg_imag)
    - /path/to/output/DUKE_XXX/label_mask_stack.npy   (S, H, W) uint8

K-space stacks are stored as float16 to halve disk size + IO and let the full
dataset fit in OS page cache (~330 GB instead of ~660 GB). Empirically the
RMS-normalized z-scored k-space has |x| <= ~130, well within fp16 range, with
<0.05% of bins underflowing to subnormals — far below the bf16-mixed precision
floor that training already imposes.

Input subtraction slices come from z-scored phases (global μ/σ over all phases for the patient),
in line with the MAMA-MIA recommendation. The mask is not z-scored.

FFT convention:
  - Write centered k-space (DC at [H//2, W//2]) via:
      K = fftshift( fft2( ifftshift(x) ) )
  - Training/visualization assumes the matching centered iFFT:
      x = fftshift( ifft2( ifftshift(K) ) )

If you have an older `syn_slices` produced with `K = fftshift(fft2(x))`, delete it and rerun.
"""

import os
import sys
sys.path.insert(0, '/home/l721f/code/kspace-pred-net')
import SimpleITK as sitk
import numpy as np
import numpy.fft as fft
from tqdm import tqdm
import concurrent.futures
import functools

# Import processing functions from preprocessing module
from preprocessing import (
    read_mri_phase_from_patient_id,
    read_segmentation_from_patient_id,
    resample_sitk # Keep resampling for consistency
)

# --- Define Target Spatial Size for Slices ---
# Choose based on data analysis (e.g., 95th percentile size after resampling)
# Needs to be large enough for most slices, and ideally power of 2 or divisible by 16/32
TARGET_SIZE = (384, 384) # Example: (Height, Width)

def pad_or_crop_slice(slice_2d, target_shape, mode='constant', apply_window=False, window_alpha=0.25):
    """
    Pads or crops a 2D numpy array to a target shape. Optionally applies a
    cosine-taper window (similar to a Tukey/Hanning window) within any padded
    region to reduce FFT ringing artefacts introduced by hard zero padding.

    Parameters
    ----------
    slice_2d : np.ndarray
        The input 2-D array with shape (H, W).
    target_shape : tuple
        Desired output shape (target_H, target_W).
    mode : str, default 'constant'
        Padding mode forwarded to ``np.pad``.
    apply_window : bool, default False
        If True, a smooth window is multiplied with the padded slice so that
        values taper from 1 inside the original FOV to 0 at the new borders.
    window_alpha : float, default 0.25
        Fraction of the full dimension that is tapered (0<alpha<1). Ignored
        if *apply_window* is False.

    Returns
    -------
    np.ndarray
        The padded/cropped (and optionally windowed) 2-D array.
    """
    assert len(target_shape) == 2, "Target shape must be (H, W)"
    target_h, target_w = target_shape
    current_h, current_w = slice_2d.shape

    # Calculate padding/cropping for height
    if current_h < target_h:
        h_pad_total = target_h - current_h
        h_pad_before = h_pad_total // 2
        h_pad_after = h_pad_total - h_pad_before
        h_crop_start, h_crop_end = 0, current_h
    else:
        h_pad_before, h_pad_after = 0, 0
        h_crop_total = current_h - target_h
        h_crop_start = h_crop_total // 2
        h_crop_end = h_crop_start + target_h

    # Calculate padding/cropping for width
    if current_w < target_w:
        w_pad_total = target_w - current_w
        w_pad_before = w_pad_total // 2
        w_pad_after = w_pad_total - w_pad_before
        w_crop_start, w_crop_end = 0, current_w
    else:
        w_pad_before, w_pad_after = 0, 0
        w_crop_total = current_w - target_w
        w_crop_start = w_crop_total // 2
        w_crop_end = w_crop_start + target_w

    # Crop (if necessary)
    cropped_slice = slice_2d[h_crop_start:h_crop_end, w_crop_start:w_crop_end]

    # Pad (if necessary)
    padded_slice = np.pad(
        cropped_slice,
        ((h_pad_before, h_pad_after), (w_pad_before, w_pad_after)),
        mode=mode
    )

    # Optionally apply a cosine-taper window in the padded region to smooth
    # the transition to zeros and reduce ringing artefacts.
    if apply_window and (h_pad_before + h_pad_after + w_pad_before + w_pad_after) > 0:
        # Build 1-D windows for both dimensions
        def _cosine_taper(length, pad_before, pad_after):
            """Create a 1-D symmetric cosine taper window."""
            win = np.ones(length, dtype=np.float32)

            # Top/left taper
            if pad_before > 0:
                idx = np.arange(pad_before)
                win[idx] = 0.5 * (1 - np.cos(np.pi * (idx + 1) / (pad_before + 1)))

            # Bottom/right taper
            if pad_after > 0:
                idx = np.arange(pad_after)
                # Reverse index so that it decreases towards the edge
                win[-pad_after:] = 0.5 * (1 - np.cos(np.pi * (idx[::-1] + 1) / (pad_after + 1)))
            return win

        win_y = _cosine_taper(target_h, h_pad_before, h_pad_after)
        win_x = _cosine_taper(target_w, w_pad_before, w_pad_after)
        window_2d = np.outer(win_y, win_x).astype(padded_slice.dtype)

        padded_slice = padded_slice * window_2d

    return padded_slice


WRITE_EXTRA_TIMEPOINTS = False


def _phase_file_exists(images_folder, patient_id, phase):
    return os.path.exists(
        f"{images_folder}/{patient_id}/{patient_id}_000{phase}.nii.gz"
    )


def _centered_fft2(img_hw):
    """Centered 2D FFT of a (H, W) real slice: fftshift(fft2(ifftshift(x)))."""
    img0 = fft.ifftshift(img_hw, axes=(-2, -1))
    k = fft.fftshift(fft.fft2(img0, norm="ortho"), axes=(-2, -1))
    return k.astype(np.complex64, copy=False)


def process_patient_kspace_slices(patient_id, images_folder, seg_folder, output_base):
    """
    Process one patient for 2D slice k-space target training:
      - Load phases 0 (pre), 2 (post2); optionally 1 (post1) and 3 (post3).
      - Resample each phase (BSpline) and the segmentation (NearestNeighbor)
        to 1x1x1 mm.
      - For each axial slice, pad/crop each phase image to TARGET_SIZE with the
        cosine-taper window, and pad/crop the mask without window. 2D-FFT every
        phase slice individually -> per-slice complex k-space per phase.
      - Normalize all phase k-spaces by the per-patient RMS of the t=0 k-space
        magnitude (mirrors prepare_kspace_full.py).
      - Compute input pairs as k-space subtractions (K[post_t] - K[pre]):
        t0t2 primary, plus t0t1 and t0t3 extras when WRITE_EXTRA_TIMEPOINTS.
      - Save input_kspace_stack[_t0tX].npy (S, 2, H, W), target_kspace_stack.npy
        (S, 2, H, W), label_mask_stack.npy (S, H, W), kspace_rms_scale.npy.
    """
    patient_output_dir = os.path.join(output_base, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)

    required_phases = [0, 2]
    optional_phases = [1, 3] if WRITE_EXTRA_TIMEPOINTS else []

    phases_arr = {}
    for p in required_phases:
        img_sitk = read_mri_phase_from_patient_id(images_folder, patient_id, phase=p)
        img_sitk = sitk.Cast(img_sitk, sitk.sitkFloat32)
        img_rs = resample_sitk(img_sitk, new_spacing=[1, 1, 1], interpolator=sitk.sitkBSpline)
        phases_arr[p] = sitk.GetArrayFromImage(img_rs).astype(np.float32)

    for p in optional_phases:
        if not _phase_file_exists(images_folder, patient_id, p):
            continue
        img_sitk = read_mri_phase_from_patient_id(images_folder, patient_id, phase=p)
        img_sitk = sitk.Cast(img_sitk, sitk.sitkFloat32)
        img_rs = resample_sitk(img_sitk, new_spacing=[1, 1, 1], interpolator=sitk.sitkBSpline)
        phases_arr[p] = sitk.GetArrayFromImage(img_rs).astype(np.float32)

    seg_sitk = read_segmentation_from_patient_id(seg_folder, patient_id)
    seg_resampled = resample_sitk(seg_sitk, new_spacing=[1, 1, 1], interpolator=sitk.sitkNearestNeighbor)
    mask_array = sitk.GetArrayFromImage(seg_resampled).astype(np.uint8)

    num_slices = phases_arr[0].shape[0]
    H, W = TARGET_SIZE

    phases_k = {p: np.empty((num_slices, H, W), dtype=np.complex64) for p in phases_arr}
    label_mask_slices = np.empty((num_slices, H, W), dtype=np.uint8)
    target_kspace_slices = np.empty((num_slices, 2, H, W), dtype=np.float16)

    for i in range(num_slices):
        for p, arr in phases_arr.items():
            img_slice = pad_or_crop_slice(arr[i], TARGET_SIZE, mode='constant', apply_window=True)
            phases_k[p][i] = _centered_fft2(img_slice)

        mask_slice = pad_or_crop_slice(
            mask_array[i], TARGET_SIZE, mode='constant', apply_window=False
        ).astype(np.uint8)
        label_mask_slices[i] = mask_slice

        pixel_space_one_hot = np.zeros((2, H, W), dtype=np.float32)
        pixel_space_one_hot[0] = (mask_slice == 0)
        pixel_space_one_hot[1] = (mask_slice == 1)
        oh0 = fft.ifftshift(pixel_space_one_hot, axes=(-2, -1))
        k_oh = fft.fftshift(fft.fft2(oh0, axes=(-2, -1), norm="ortho"), axes=(-2, -1))
        target_kspace_slices[i, 0] = np.real(k_oh[1])
        target_kspace_slices[i, 1] = np.imag(k_oh[1])

    k0 = phases_k[0]
    scale_factor = float(np.sqrt(np.mean(np.abs(k0) ** 2, dtype=np.float64)))
    scale_factor = scale_factor if scale_factor > 0.0 else 1.0
    for p in phases_k:
        phases_k[p] = phases_k[p] / scale_factor

    def _pair(t_ref, t_other):
        k_diff = phases_k[t_other] - phases_k[t_ref]
        return np.stack([np.real(k_diff), np.imag(k_diff)], axis=1).astype(np.float16, copy=False)

    input_stack_t0t2 = _pair(0, 2)

    extra_inputs = {}
    if WRITE_EXTRA_TIMEPOINTS:
        if 1 in phases_k:
            extra_inputs["t0t1"] = _pair(0, 1)
        if 3 in phases_k:
            extra_inputs["t0t3"] = _pair(0, 3)

    np.save(os.path.join(patient_output_dir, "input_kspace_stack.npy"), input_stack_t0t2)
    for key, arr in extra_inputs.items():
        np.save(os.path.join(patient_output_dir, f"input_kspace_stack_{key}.npy"), arr)
    np.save(os.path.join(patient_output_dir, "target_kspace_stack.npy"), target_kspace_slices)
    np.save(os.path.join(patient_output_dir, "label_mask_stack.npy"), label_mask_slices)
    np.save(os.path.join(patient_output_dir, "kspace_rms_scale.npy"),
            np.array([scale_factor], dtype=np.float32))

    return f"Processed {patient_id}"

def main():
    # Define paths
    images_folder = "/home/l721f/data/mama-mia/images"
    seg_folder = "/home/l721f/data/mama-mia/segmentations/expert"
    # Define output base directory for SLICE-BASED k-space preprocessed data
    output_base = "/home/l721f/data/mama-mia/syn_slices" # New output folder
    os.makedirs(output_base, exist_ok=True)
    num_workers = 32 # Number of parallel processes

    # List only DUKE_* patient directories in the images folder
    patient_ids = sorted([
        d for d in os.listdir(images_folder)
        if os.path.isdir(os.path.join(images_folder, d)) #and d.startswith("DUKE_")
    ])

    # --- Parallel Processing ---
    print(f"Starting parallel processing for {len(patient_ids)} patients using {num_workers} workers...")

    # Use functools.partial to fix arguments for the worker function
    worker_func = functools.partial(process_patient_kspace_slices,
                                    images_folder=images_folder,
                                    seg_folder=seg_folder,
                                    output_base=output_base)

    futures = []
    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        for patient_id in patient_ids:
            futures.append(executor.submit(worker_func, patient_id))

        # Use tqdm to track completion progress
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Patients (2D Slices)"):
            try:
                result = future.result() # Get result (or exception) from the future
                results.append(result)
                # Optionally log success/failure based on result string
                if result is not None and "Error" in result:
                    print(f"Worker reported issue: {result}") # Print errors clearly
            except Exception as exc:
                # This catches exceptions raised *before* the try/except within process_patient_kspace_slices
                # (e.g., during argument passing or process startup) or if future.result() raises one directly
                # Find which patient caused the error if possible (tricky with as_completed)
                print(f'Processing generated an exception: {exc}')
                # Attempt to find the patient ID associated with the failed future (may not always work)
                # for i, f in enumerate(futures):
                #     if f == future:
                #         print(f"Exception likely occurred for patient: {patient_ids[i]}")
                #         break

    print(f"\nParallel processing finished. Processed {len(results)} jobs.")
    print("\nPreprocessing complete. Image slices are now Z-score normalized before FFT.")


if __name__ == '__main__':
    # Consider setting a start method if issues arise, though 'fork' (default on Linux) is usually fine
    # import multiprocessing
    # try:
    #     multiprocessing.set_start_method('fork', force=True) # Or 'spawn'
    # except RuntimeError:
    #     pass # Ignore if already set
    main()