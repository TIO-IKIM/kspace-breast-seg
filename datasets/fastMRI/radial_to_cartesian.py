"""Radial k-space -> Cartesian single-coil pipeline with spoke / sample masks.

Mirrors the working preprocessing in `reconstruction.py` (3-bin temporal
binning, Pipe-Menon DCF, sigpy Kaiser-Bessel gridding, ESPIRiT-combine, t0-t2
difference, RMS normalization), with two coil paths:

  - 16-coil RAW path (default, used by the old-model comparison): read
    `/rawdata/{pid}_2.h5`, partition zero-fill + z-FFT, ESPIRiT-calibrate
    on time-mean of all 16 coils. Matches the old project's training
    distribution exactly (modulo the spoke/sample mask).
  - 4-coil GCC path: read the new project's GCC H5 and reuse its
    pre-cached ESPIRiT maps. Cheaper but a different ESPIRiT phase
    convention than the old model trained on, so it produces in-magnitude
    but out-of-phase k-space relative to the old preprocessing.

In both paths the per-patient RMS scale is loaded from the old
preprocessing's `training_data_full/{pid}/kspace_rms_scale.npy` so the
input distribution matches the old model's training distribution.

The result is a `(S=192, 2, 320, 320) float32` tensor matching the layout of
`input_kspace_stack.npy` consumed by `FastMRIBreast3DKSpaceFullDataModule`.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import sigpy as sp
from sigpy import Device, backend, interp
from sigpy.mri import app
from sigpy.mri import dcf as mri_dcf

BASE_RES = 320
N_SAMPLES = BASE_RES * 2
EXPECTED_N_SPOKES_TOTAL = 288
EXPECTED_N_VIRTUAL_COILS = 4
EXPECTED_N_RAW_COILS = 16
IMAGES_PER_SLAB = 192
N_TIME = 3
SPOKES_PER_FRAME = EXPECTED_N_SPOKES_TOTAL // N_TIME

# OLD pipeline ESPIRiT settings — sigpy defaults via positional calib_width=32.
ESPIRIT_CALIB_WIDTH = 32

GRID_KERNEL = "kaiser_bessel"
GRID_WIDTH = 4
GRID_BETA = 8
PIPE_MENON_MAX_ITER = 15

# DCF cache key schema:
#   (mask_id, t_idx) -> (traj_kept_dev, dcf_kept_dev) staged on the resolved device.
# `mask_id` should be a hashable value uniquely identifying the (spoke, sample)
# mask combination per timepoint (e.g. an int derived from a content hash). The
# caller is responsible for invalidating the cache when the device changes.
_TRAJ_DCF_CACHE: Dict[Tuple, Tuple] = {}
_TRAJ_FULL_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def _golden_angle_traj_full(
    n_spokes_total: int = EXPECTED_N_SPOKES_TOTAL,
    n_samples: int = N_SAMPLES,
    base_res: int = BASE_RES,
    gind: int = 1,
) -> np.ndarray:
    """Return (n_spokes_total, n_samples, 2) float32 golden-angle trajectory.

    Identical convention to `reconstruction.py:get_traj` after `traj /= 2`
    (kmax = base_res / 2, in sigpy gridding "FFT index" units).
    """
    key = (n_spokes_total, n_samples, base_res)
    cached = _TRAJ_FULL_CACHE.get(key)
    if cached is not None:
        return cached
    base_lin = np.arange(n_samples, dtype=np.float32).reshape(1, -1) - base_res
    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)
    angles = np.arange(n_spokes_total, dtype=np.float32).reshape(-1, 1) * base_rad
    traj = np.zeros((n_spokes_total, n_samples, 2), dtype=np.float32)
    traj[..., 0] = (np.cos(angles) @ base_lin).astype(np.float32, copy=False)
    traj[..., 1] = (np.sin(angles) @ base_lin).astype(np.float32, copy=False)
    traj /= 2.0
    _TRAJ_FULL_CACHE[key] = traj
    return traj


def load_gcc_radial_kspace(h5_path: str) -> np.ndarray:
    """Load (S=192, K=4, n_spokes_total=288, n_samples=640) complex64 k-space.

    The upstream H5 has already had the partition zero-fill and z-FFT applied
    (see `reconstruct_native_radial_full.py`).
    """
    with h5py.File(h5_path, "r") as f:
        ksp = f["kspace_full"][:]
        n_spokes_total = int(f.attrs["n_spokes_total"])
        ncc = int(f.attrs["n_virtual_coils"])
        base_res = int(f.attrs["base_res"])
    if n_spokes_total != EXPECTED_N_SPOKES_TOTAL:
        raise RuntimeError(
            f"{h5_path}: n_spokes_total={n_spokes_total} != "
            f"EXPECTED {EXPECTED_N_SPOKES_TOTAL}"
        )
    if ncc != EXPECTED_N_VIRTUAL_COILS:
        raise RuntimeError(
            f"{h5_path}: n_virtual_coils={ncc} != EXPECTED "
            f"{EXPECTED_N_VIRTUAL_COILS}"
        )
    if base_res != BASE_RES:
        raise RuntimeError(f"{h5_path}: base_res={base_res} != {BASE_RES}")
    return ksp.astype(np.complex64, copy=False)


def load_full_sens_maps(sens_path: str) -> np.ndarray:
    """Load (S=192, K=4, 320, 320) complex64 ESPIRiT maps from the new project.

    The on-disk format is float16 with a leading real/imag axis-2; combine into
    complex on load.
    """
    arr = np.load(sens_path, mmap_mode="r")
    if arr.ndim != 5 or arr.shape[2] != 2:
        raise RuntimeError(
            f"{sens_path}: expected shape (S, K, 2, H, W); got {arr.shape}"
        )
    real = arr[:, :, 0].astype(np.float32, copy=False)
    imag = arr[:, :, 1].astype(np.float32, copy=False)
    return (real + 1j * imag).astype(np.complex64, copy=False)


# --- 16-coil raw path (matches reconstruction.process_h5_file_fast bit-for-bit) ---


def load_raw_kspace_16coil(h5_path: str) -> np.ndarray:
    """Read raw 16-coil radial H5 -> (S=192, C=16, 288, 640) complex64.

    Mirrors `reconstruction.py:process_h5_file_fast` lines 86-103: read raw
    real/imag k-space, transpose to (partitions, coils, spokes, samples),
    zero-fill partitions to IMAGES_PER_SLAB, and z-FFT along the partition
    axis.
    """
    with h5py.File(h5_path, "r", swmr=True) as f:
        ksp_f = f["kspace"][:].T
    # `.T` followed by transpose([4,3,2,1,0]) is a no-op on the axis layout but
    # is preserved here to match `reconstruction.py` exactly.
    ksp_f = np.transpose(ksp_f, (4, 3, 2, 1, 0))
    ksp = (ksp_f[0] + 1j * ksp_f[1]).astype(np.complex64, copy=False)
    # (spokes, samples, coils, partitions) -> (partitions, coils, spokes, samples)
    ksp = np.transpose(ksp, (3, 2, 0, 1))

    if ksp.shape[1] != EXPECTED_N_RAW_COILS:
        raise RuntimeError(
            f"{h5_path}: expected {EXPECTED_N_RAW_COILS} raw coils, got {ksp.shape[1]}"
        )
    if ksp.shape[2] != EXPECTED_N_SPOKES_TOTAL:
        raise RuntimeError(
            f"{h5_path}: expected {EXPECTED_N_SPOKES_TOTAL} spokes, got {ksp.shape[2]}"
        )
    if ksp.shape[3] != N_SAMPLES:
        raise RuntimeError(
            f"{h5_path}: expected {N_SAMPLES} samples, got {ksp.shape[3]}"
        )

    partitions = ksp.shape[0]
    center = partitions // 2
    shift = IMAGES_PER_SLAB // 2 - center
    ksp_zf = np.zeros(
        [IMAGES_PER_SLAB] + list(ksp.shape[1:]), dtype=np.complex64,
    )
    ksp_zf[shift: shift + partitions] = ksp
    ksp_zf = sp.fft(ksp_zf, axes=(0,))
    return ksp_zf  # (S=192, C=16, 288, 640)


def compute_full_espirit_maps_16coil(
    ksp: np.ndarray,
    device_id: int = 0,
    base_res: int = BASE_RES,
) -> np.ndarray:
    """Per-slice ESPIRiT maps from full-spoke time-mean Cartesian k-space.

    Mirrors the calibration in `reconstruction.py:process_h5_file_fast` lines
    197-206: bin into N_TIME=3 timepoints, grid each timepoint with
    Pipe-Menon-DCF-weighted Kaiser-Bessel, mean across timepoints per slice,
    `app.EspiritCalib(centered_kspace, calib_width=32)`. R-invariant — this
    is the calibration the old model trained against.

    Args:
        ksp: (S, C=16, n_spokes_total, n_samples) complex64 (post-zero-fill,
            post-z-FFT — i.e. output of `load_raw_kspace_16coil`).
        device_id: GPU device id (-1 -> CPU).

    Returns:
        (S, C=16, base_res, base_res) complex64 sens maps.
    """
    S, C, n_spokes_total, n_samples = ksp.shape
    if n_spokes_total != EXPECTED_N_SPOKES_TOTAL:
        raise ValueError(
            f"expected {EXPECTED_N_SPOKES_TOTAL} spokes; got {n_spokes_total}"
        )

    full_mask = np.ones(n_spokes_total, dtype=np.float32)
    ksp_binned, kept_per_t = apply_spoke_mask_and_bin(ksp, full_mask, n_time=N_TIME)
    # ksp_binned: (S, T, C, spokes_per_frame, n_samples)

    device = Device(device_id) if device_id >= 0 else sp.cpu_device
    traj_per_t = _per_timepoint_traj_per_t(
        n_time=N_TIME, spokes_per_frame=ksp_binned.shape[3],
        n_samples=n_samples, base_res=base_res,
    )

    mps_out = np.zeros((S, C, base_res, base_res), dtype=np.complex64)

    with device:
        for t in range(N_TIME):
            ksp_t = ksp_binned[:, t]
            ksp_flat, traj_flat = _flatten_with_sample_mask(
                ksp_t, traj_per_t[t], None, kept_per_t[t],
            )
            cache_key = ("espirit_calib_full", t, n_samples, ksp_flat.shape[-1])
            traj_dev, dcf_dev = _stage_traj_dcf(cache_key, traj_flat, device)
            ksp_dev = backend.to_device(ksp_flat, device)
            ksp_dcf = ksp_dev * dcf_dev[None, None, :]
            k_cart_t = interp.gridding(
                ksp_dcf, traj_dev,
                shape=(S, C, base_res, base_res),
                kernel=GRID_KERNEL, width=GRID_WIDTH, param=GRID_BETA,
            )
            if t == 0:
                k_cart_sum = k_cart_t
            else:
                k_cart_sum = k_cart_sum + k_cart_t
        k_cart_mean = k_cart_sum / float(N_TIME)
        xp = backend.get_array_module(k_cart_mean)

        # Per-slice ESPIRiT calib (sigpy default thresh / kernel_width / max_iter
        # / crop, matching the old reconstruction pipeline).
        for s_idx in range(S):
            kspace_shifted = xp.fft.fftshift(k_cart_mean[s_idx], axes=(-2, -1))
            mps = app.EspiritCalib(
                kspace_shifted,
                ESPIRIT_CALIB_WIDTH,
                show_pbar=False,
                device=device,
            ).run()
            mps_out[s_idx] = backend.to_device(mps, sp.cpu_device).astype(
                np.complex64, copy=False,
            )
    return mps_out


def cache_or_compute_sens_maps_16coil(
    pid: str,
    raw_dir: str,
    cache_dir: str,
    device_id: int = 0,
) -> np.ndarray:
    """Load cached 16-coil sens maps for `pid`, or compute + persist them.

    Cache path: `{cache_dir}/{pid}_mps_16coil.npy`. Atomic via .tmp + rename.
    Returns (S=192, C=16, 320, 320) complex64.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{pid}_mps_16coil.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r").astype(np.complex64, copy=False)

    raw_path = os.path.join(raw_dir, f"{pid}_2.h5")
    ksp = load_raw_kspace_16coil(raw_path)
    mps = compute_full_espirit_maps_16coil(ksp, device_id=device_id)
    tmp = cache_path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, mps)
    os.replace(tmp, cache_path)
    return mps


def apply_spoke_mask_and_bin(
    ksp: np.ndarray,
    spoke_mask: np.ndarray,
    n_time: int = N_TIME,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-mask spokes, then reshape into (S, T, K, spokes_per_frame, samples).

    Args:
        ksp: (S, K, n_spokes_total, n_samples) complex64.
        spoke_mask: (n_spokes_total,) {0,1} float32.
        n_time: number of temporal bins (default 3).

    Returns:
        (ksp_binned, kept_per_t) where:
          ksp_binned: (S, T, K, spokes_per_frame, n_samples) complex64.
          kept_per_t: (T, spokes_per_frame) bool — per-timepoint per-bin spoke
            keep flags after binning. Used by the gridding step to skip
            timepoints with zero kept spokes (degenerate at extreme R+contig).
    """
    S, K, n_spokes_total, n_samples = ksp.shape
    if spoke_mask.shape[0] != n_spokes_total:
        raise ValueError(
            f"spoke_mask has {spoke_mask.shape[0]} entries; expected "
            f"{n_spokes_total}"
        )
    spokes_per_frame = n_spokes_total // n_time
    used = n_time * spokes_per_frame
    mask = spoke_mask[:used].astype(np.float32, copy=False)
    # Apply mask on the spoke axis before reshape.
    ksp_redu = ksp[:, :, :used, :] * mask[None, None, :, None]
    ksp_binned = ksp_redu.reshape(S, K, n_time, spokes_per_frame, n_samples)
    ksp_binned = np.transpose(ksp_binned, (0, 2, 1, 3, 4)).copy()
    kept_per_t = mask.reshape(n_time, spokes_per_frame) > 0.5
    return ksp_binned, kept_per_t


def _per_timepoint_traj_per_t(
    n_time: int = N_TIME,
    spokes_per_frame: int = SPOKES_PER_FRAME,
    n_samples: int = N_SAMPLES,
    base_res: int = BASE_RES,
) -> np.ndarray:
    """Return (T, spokes_per_frame, n_samples, 2) float32 per-timepoint traj."""
    full = _golden_angle_traj_full(
        n_spokes_total=n_time * spokes_per_frame,
        n_samples=n_samples,
        base_res=base_res,
    )
    return full.reshape(n_time, spokes_per_frame, n_samples, 2)


def _flatten_with_sample_mask(
    ksp_t: np.ndarray,
    traj_t: np.ndarray,
    sample_mask_t: Optional[np.ndarray],
    kept_t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten one timepoint to (S, K, npts) k-space and (npts, 2) trajectory.

    Drops:
      - spokes whose `kept_t` flag is False (so DCF on the dropped spokes
        does not waste iterations on identically-zero rows),
      - samples whose `sample_mask_t` is False (per-spoke sample mask).

    Args:
        ksp_t: (S, K, spokes_per_frame, n_samples) complex64.
        traj_t: (spokes_per_frame, n_samples, 2) float32.
        sample_mask_t: optional (spokes_per_frame, n_samples) {0,1} float32 or
            None to keep all samples on every spoke.
        kept_t: (spokes_per_frame,) bool — which spokes survived the spoke mask.

    Returns:
        (ksp_flat, traj_flat) where ksp_flat is (S, K, npts) and traj_flat is
        (npts, 2). `npts` reflects the union of kept (spoke, sample) pairs.
    """
    S, K, spokes_per_frame, n_samples = ksp_t.shape

    # Spoke-keep selection.
    if not kept_t.any():
        return (
            np.zeros((S, K, 0), dtype=np.complex64),
            np.zeros((0, 2), dtype=np.float32),
        )

    keep_idx = np.flatnonzero(kept_t)
    ksp_kept = ksp_t[:, :, keep_idx, :]              # (S, K, n_kept, n_samp)
    traj_kept = traj_t[keep_idx]                     # (n_kept, n_samp, 2)

    if sample_mask_t is None:
        return (
            ksp_kept.reshape(S, K, -1).astype(np.complex64, copy=False),
            traj_kept.reshape(-1, 2).astype(np.float32, copy=False),
        )

    sm_kept = sample_mask_t[keep_idx].astype(bool)   # (n_kept, n_samp)
    flat_keep = sm_kept.reshape(-1)                  # (n_kept * n_samp,)
    if not flat_keep.any():
        return (
            np.zeros((S, K, 0), dtype=np.complex64),
            np.zeros((0, 2), dtype=np.float32),
        )

    ksp_flat = ksp_kept.reshape(S, K, -1)[:, :, flat_keep]
    traj_flat = traj_kept.reshape(-1, 2)[flat_keep]
    return (
        ksp_flat.astype(np.complex64, copy=False),
        traj_flat.astype(np.float32, copy=False),
    )


def _pipe_menon_dcf(traj_flat: np.ndarray, base_res: int = BASE_RES) -> np.ndarray:
    """Pipe-Menon DCF on CPU (matches reconstruction.py)."""
    if traj_flat.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    w = mri_dcf.pipe_menon_dcf(
        traj_flat,
        img_shape=[base_res, base_res],
        device=sp.cpu_device,
        max_iter=PIPE_MENON_MAX_ITER,
        beta=GRID_BETA,
        width=GRID_WIDTH,
        show_pbar=False,
    )
    return np.asarray(w, dtype=np.float32)


def _stage_traj_dcf(
    cache_key: Tuple,
    traj_flat: np.ndarray,
    device,
) -> Tuple:
    """Compute DCF + stage (traj, dcf) on `device`. Cached by `cache_key`."""
    cached = _TRAJ_DCF_CACHE.get(cache_key)
    if cached is not None:
        return cached
    dcf_cpu = _pipe_menon_dcf(traj_flat)
    with device:
        traj_dev = backend.to_device(traj_flat, device)
        dcf_dev = backend.to_device(dcf_cpu, device)
    _TRAJ_DCF_CACHE[cache_key] = (traj_dev, dcf_dev)
    return traj_dev, dcf_dev


def grid_and_combine(
    ksp_binned: np.ndarray,
    kept_per_t: np.ndarray,
    sens_maps: np.ndarray,
    sample_mask: Optional[np.ndarray] = None,
    cache_id: Optional[Tuple] = None,
    base_res: int = BASE_RES,
    device_id: int = 0,
) -> np.ndarray:
    """Grid radial spokes per timepoint and ESPIRiT-combine to single coil.

    Args:
        ksp_binned: (S, T, K, spokes_per_frame, n_samples) complex64. The
            spoke mask must already be applied.
        kept_per_t: (T, spokes_per_frame) bool.
        sens_maps: (S, K, base_res, base_res) complex64 — full-spoke ESPIRiT
            maps shared across all R values.
        sample_mask: optional (n_spokes_total, n_samples) {0,1} float32. When
            provided, dropped sample/trajectory entries are removed from
            gridding and DCF is recomputed accordingly.
        cache_id: hashable that uniquely identifies (spoke_mask, sample_mask).
            Used as part of the DCF cache key. If None, the cache is bypassed.
        base_res: Cartesian image side length.
        device_id: GPU device id for sigpy gridding (-1 -> CPU).

    Returns:
        (S, T, base_res, base_res) complex64 — single-coil k-space (DC at
        center, fftshift convention).
    """
    S, T, K, spokes_per_frame, n_samples = ksp_binned.shape
    if sens_maps.shape != (S, K, base_res, base_res):
        raise ValueError(
            f"sens_maps shape {sens_maps.shape} != ({S}, {K}, {base_res}, {base_res})"
        )

    device = Device(device_id) if device_id >= 0 else sp.cpu_device
    traj_per_t = _per_timepoint_traj_per_t(
        n_time=T, spokes_per_frame=spokes_per_frame,
        n_samples=n_samples, base_res=base_res,
    )

    # Reshape sample mask onto per-timepoint per-spoke layout.
    if sample_mask is not None:
        if sample_mask.shape != (T * spokes_per_frame, n_samples):
            raise ValueError(
                f"sample_mask shape {sample_mask.shape} != "
                f"({T * spokes_per_frame}, {n_samples})"
            )
        sm_per_t = sample_mask.reshape(T, spokes_per_frame, n_samples)
    else:
        sm_per_t = None

    out = np.zeros((S, T, base_res, base_res), dtype=np.complex64)

    with device:
        # Stage sens maps once on the device to amortize host->device transfers.
        sens_dev = backend.to_device(sens_maps, device)

        for t in range(T):
            kept_t = kept_per_t[t]
            traj_t = traj_per_t[t]
            sm_t = sm_per_t[t] if sm_per_t is not None else None
            ksp_t = ksp_binned[:, t]   # (S, K, spokes_per_frame, n_samples)

            ksp_flat, traj_flat = _flatten_with_sample_mask(
                ksp_t, traj_t, sm_t, kept_t,
            )
            if ksp_flat.shape[-1] == 0:
                # Fully-masked timepoint -> contribute zeros (the t0-t2 diff
                # will then equal the surviving timepoint's k-space).
                continue

            key = (cache_id, t) if cache_id is not None else None
            if key is not None:
                traj_dev, dcf_dev = _stage_traj_dcf(key, traj_flat, device)
            else:
                dcf_cpu = _pipe_menon_dcf(traj_flat)
                traj_dev = backend.to_device(traj_flat, device)
                dcf_dev = backend.to_device(dcf_cpu, device)

            ksp_dev = backend.to_device(ksp_flat, device)        # (S, K, npts)
            ksp_dcf = ksp_dev * dcf_dev[None, None, :]
            k_cart = interp.gridding(
                ksp_dcf, traj_dev,
                shape=(S, K, base_res, base_res),
                kernel=GRID_KERNEL, width=GRID_WIDTH, param=GRID_BETA,
            )                                                    # (S, K, H, W)

            xp = backend.get_array_module(k_cart)
            # sigpy gridding output has DC at [0, 0]. Centered iFFT2 from there
            # is `fftshift(ifft2(k_cart))`; fftshift+ifftshift would be a no-op.
            img = xp.fft.fftshift(
                xp.fft.ifft2(k_cart, axes=(-2, -1), norm="ortho"),
                axes=(-2, -1),
            )                                                     # (S, K, H, W)
            combined_img = xp.sum(xp.conj(sens_dev) * img, axis=1)  # (S, H, W)
            # Centered FFT2 back to k-space (DC at center).
            combined_k = xp.fft.fftshift(
                xp.fft.fft2(
                    xp.fft.ifftshift(combined_img, axes=(-2, -1)),
                    axes=(-2, -1), norm="ortho",
                ),
                axes=(-2, -1),
            )

            out[:, t] = backend.to_device(combined_k, sp.cpu_device).astype(
                np.complex64, copy=False,
            )

    return out


def kspace_difference_input(
    combined_kspace: np.ndarray,
    rms_scale: float,
    t_ref: int = 0,
    t_other: int = 2,
) -> np.ndarray:
    """Build (S, 2, H, W) float32 t_other-t_ref difference, RMS-normalized.

    Mirrors `prepare_kspace_full.save_case` exactly, except `rms_scale` is
    consumed externally so the comparison reuses the per-patient scale that
    the old preprocessing already wrote. Both timepoints are scaled by the
    same factor before subtraction; this is algebraically equivalent to
    scaling the difference once.
    """
    if combined_kspace.ndim != 4:
        raise ValueError(
            f"combined_kspace must be (S, T, H, W); got {combined_kspace.shape}"
        )
    s = float(rms_scale) if rms_scale > 0.0 else 1.0
    scaled = combined_kspace / s
    diff = scaled[:, t_other] - scaled[:, t_ref]
    return np.stack([diff.real, diff.imag], axis=1).astype(np.float32, copy=False)


def build_spoke_mask(
    n_spokes: int,
    accel: int,
    scheme: str,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Spoke-axis keep mask matching the kspace-radial project's helper.

    Inlined so this module can be imported without putting the kspace-radial
    `src/` package on sys.path (it would collide with kspace-pred-net's own
    `src/` package).

    - accel<=1 -> all-ones.
    - For accel=R, keeps `n_spokes // R` spokes.
    - `uniform_equispaced`: indices [0, R, 2R, ...].
    - `contiguous`: first `n_spokes // R` indices.
    - `random`: uniform-random subset (requires `rng`).
    """
    mask = np.zeros(n_spokes, dtype=np.float32)
    if accel <= 1:
        mask[:] = 1.0
        return mask
    n_keep = max(1, n_spokes // int(accel))
    if scheme == "uniform_equispaced":
        keep = np.arange(n_keep, dtype=np.int64) * int(accel)
    elif scheme == "contiguous":
        keep = np.arange(n_keep, dtype=np.int64)
    elif scheme == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        keep = rng.choice(n_spokes, size=n_keep, replace=False)
    else:
        raise ValueError(f"Unknown spoke-mask scheme: {scheme}")
    mask[keep] = 1.0
    return mask


def build_sample_mask(
    n_spokes: int,
    n_samples: int,
    accel: int,
    scheme: str = "random",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Build a (n_spokes, n_samples) {0,1} float32 mask along the readout axis.

    - `random`: independent uniform-random subset of size `n_samples // R` per
      spoke. Different spokes get different subsets; representative of a
      detector readout that drops random samples per acquisition.
    - `uniform_equispaced`: indices [0, R, 2R, ...], same on every spoke.
    - `contiguous`: first `n_samples // R` samples on every spoke (truncated
      readout).
    - `accel<=1`: all-ones.
    """
    mask = np.zeros((n_spokes, n_samples), dtype=np.float32)
    if accel <= 1:
        mask[:] = 1.0
        return mask
    n_keep = max(1, n_samples // int(accel))
    if scheme == "uniform_equispaced":
        keep = np.arange(n_keep, dtype=np.int64) * int(accel)
        mask[:, keep] = 1.0
    elif scheme == "contiguous":
        mask[:, :n_keep] = 1.0
    elif scheme == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        for s in range(n_spokes):
            idx = rng.choice(n_samples, size=n_keep, replace=False)
            mask[s, idx] = 1.0
    else:
        raise ValueError(f"Unknown sample-mask scheme: {scheme}")
    return mask


def reset_caches() -> None:
    """Clear the trajectory / DCF caches. Call between fold transitions if
    holding GPU buffers becomes inconvenient."""
    _TRAJ_DCF_CACHE.clear()
    _TRAJ_FULL_CACHE.clear()
