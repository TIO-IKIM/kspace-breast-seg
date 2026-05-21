import os
import sys
import glob
import h5py
import numpy as np

import sigpy as sp
from sigpy import Device, backend, interp
from sigpy.mri import app
from sigpy.mri import dcf as mri_dcf

from tqdm import tqdm

sys.path.append("/home/l721f/code/kspace-pred-net/")
from recon_algos.tools.tools import espirit_combine


# IO
IN_DIR = "/home/l721f/data/fastmri-breast/rawdata/"
OUT_DIR_KS = "/home/l721f/data/fastmri-breast/kspace"
OUT_DIR_IMG = "/home/l721f/data/fastmri-breast/images"
os.makedirs(OUT_DIR_KS, exist_ok=True)
os.makedirs(OUT_DIR_IMG, exist_ok=True)


# Gridding / DCF
GRID_KERNEL = "kaiser_bessel"
GRID_WIDTH = 4
GRID_BETA = 8
PIPE_MENON_MAX_ITER = 15
_DCF_CACHE = {}


def get_traj(N_spokes: int, N_time: int, base_res: int, gind: int = 1) -> np.ndarray:
    """
    Golden-angle radial trajectory in "FFT index units" (SigPy gridding convention).
    Returns: (N_time, N_spokes, N_samples, 2) float32
    """
    N_tot_spokes = N_spokes * N_time
    N_samples = base_res * 2
    base_lin = np.arange(N_samples).reshape(1, -1) - base_res
    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (gind + tau - 1)
    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad
    traj = np.zeros((N_tot_spokes, N_samples, 2), dtype=np.float32)
    traj[..., 0] = (np.cos(base_rot) @ base_lin).astype(np.float32, copy=False)
    traj[..., 1] = (np.sin(base_rot) @ base_lin).astype(np.float32, copy=False)
    traj = traj / 2.0
    traj = traj.reshape(N_time, N_spokes, N_samples, 2)
    return traj


def compute_pipe_menon_dcf(traj_flat: np.ndarray, base_res: int) -> np.ndarray:
    w = mri_dcf.pipe_menon_dcf(
        traj_flat.astype(np.float32, copy=False),
        img_shape=[base_res, base_res],
        device=sp.cpu_device,
        max_iter=PIPE_MENON_MAX_ITER,
        beta=GRID_BETA,
        width=GRID_WIDTH,
        show_pbar=False,
    )
    return np.asarray(w, dtype=np.float32)


def get_cached_pipe_menon_dcf(traj_flat: np.ndarray, base_res: int, cache_key):
    if cache_key in _DCF_CACHE:
        return _DCF_CACHE[cache_key]
    w = compute_pipe_menon_dcf(traj_flat, base_res)
    _DCF_CACHE[cache_key] = w
    return w


def process_h5_file_fast(
    h5_file: str,
    n_spacing: int = 1,
    N_time: int = 3,
    base_res: int = 320,
    device_id: int = 0,
) -> None:
    base_name = os.path.splitext(os.path.basename(h5_file))[0]
    patient_parts = base_name.split("_")[:-1]
    patient_name = "_".join(patient_parts).strip()
    print(f"Processing {h5_file}")

    with h5py.File(h5_file, "r", swmr=True) as f:
        ksp_f = f["kspace"][:].T
        ref_img_dtype = f["temptv"].dtype

    # ksp_f: (2, partitions, samples, coils, spokes_total) after transpose below
    ksp_f = np.transpose(ksp_f, (4, 3, 2, 1, 0))
    ksp = (ksp_f[0] + 1j * ksp_f[1]).astype(np.complex64, copy=False)
    ksp = np.transpose(ksp, (3, 2, 0, 1))  # (partitions, coils, spokes, samples)

    # Zero-fill partitions and FFT along z (matches dataset `temptv` slice order).
    partitions = ksp.shape[0]
    images_per_slab = 192
    center_partition = partitions // 2
    shift = int(images_per_slab / 2 - center_partition)
    ksp_zf = np.zeros([images_per_slab] + list(ksp.shape[1:]), dtype=np.complex64)
    ksp_zf[shift : shift + partitions, ...] = ksp
    ksp_zf = sp.fft(ksp_zf, axes=(0,))
    N_slices = images_per_slab

    slice_indices = np.arange(0, N_slices, n_spacing)
    ksp_selected = ksp_zf[slice_indices]  # (S, coils, spokes, samples)
    S = int(ksp_selected.shape[0])
    C, N_spokes_total, N_samples = ksp_selected.shape[1:]

    spokes_per_frame = N_spokes_total // N_time
    N_spokes_prep = N_time * spokes_per_frame

    ksp_redu = ksp_selected[:, :, :N_spokes_prep, :]
    ksp_prep = np.reshape(ksp_redu, (S, C, N_time, spokes_per_frame, N_samples))
    ksp_prep = np.transpose(ksp_prep, (0, 2, 1, 3, 4))  # (S, T, C, spokes, samples)

    traj = get_traj(N_spokes=spokes_per_frame, N_time=N_time, base_res=base_res, gind=1)

    # DCF per timepoint (CPU, cached).
    dcf_flat_per_t = []
    for t in range(N_time):
        traj_flat_t = traj[t].reshape(-1, 2)
        cache_key = (
            "pipe_menon",
            int(base_res),
            int(N_time),
            int(spokes_per_frame),
            int(N_samples),
            int(t),
            int(GRID_WIDTH),
            int(GRID_BETA),
            int(PIPE_MENON_MAX_ITER),
        )
        dcf_flat_per_t.append(get_cached_pipe_menon_dcf(traj_flat_t, base_res, cache_key))

    # GPU: batch gridding over (S, C) for each timepoint.
    gpu = Device(device_id)
    with gpu:
        ksp_prep_dev = backend.to_device(ksp_prep, gpu)  # (S, T, C, spokes, samples)
        traj_dev = [
            backend.to_device(traj[t].reshape(-1, 2).astype(np.float32, copy=False), gpu)
            for t in range(N_time)
        ]
        dcf_dev = [
            backend.to_device(dcf_flat_per_t[t].astype(np.float32, copy=False), gpu)
            for t in range(N_time)
        ]

        k_cart_all_t = []
        for t in range(N_time):
            ksp_st = ksp_prep_dev[:, t]  # (S, C, spokes, samples)
            ksp_flat = ksp_st.reshape(S, C, -1)  # (S, C, npts)
            ksp_dcf = ksp_flat * dcf_dev[t][None, None, :]
            k_cart_t = interp.gridding(
                ksp_dcf,
                traj_dev[t],
                shape=(S, C, base_res, base_res),
                kernel=GRID_KERNEL,
                width=GRID_WIDTH,
                param=GRID_BETA,
            )
            k_cart_all_t.append(k_cart_t)

        xp = backend.get_array_module(k_cart_all_t[0])

        # Stream outputs to disk slice-by-slice.
        out_kspace = os.path.join(OUT_DIR_KS, f"{patient_name}_kspace.h5")
        out_image = os.path.join(OUT_DIR_IMG, f"{patient_name}_image.h5")

        kspace_shape = (S, N_time, base_res, base_res)

        if np.issubdtype(ref_img_dtype, np.floating):
            img_dtype = ref_img_dtype
            save_magnitude = True
        elif np.issubdtype(ref_img_dtype, np.complexfloating):
            img_dtype = np.complex64
            save_magnitude = False
        else:
            img_dtype = np.float32
            save_magnitude = True

        with h5py.File(out_kspace, "w") as f_k, h5py.File(out_image, "w") as f_i:
            ds_k = f_k.create_dataset(
                "kspace",
                shape=kspace_shape,
                dtype=np.complex64,
                chunks=(1, 1, base_res, base_res),
            )
            ds_i = f_i.create_dataset(
                "image",
                shape=kspace_shape,
                dtype=img_dtype,
                chunks=(1, 1, base_res, base_res),
            )

            print("  ESPIRiT combine on GPU, streaming slices...")
            for s in tqdm(range(S), desc="Slices"):
                # ESPIRiT calibration on mean k-space over time (better SNR).
                kspace_calib = xp.zeros((C, base_res, base_res), dtype=xp.complex64)
                for t in range(N_time):
                    kspace_calib += k_cart_all_t[t][s]
                kspace_calib /= float(N_time)

                # Gridding output is unshifted (DC at [0,0]); ESPIRiT expects centered.
                kspace_shifted = xp.fft.fftshift(kspace_calib, axes=(-2, -1))
                mps = app.EspiritCalib(kspace_shifted, 32, show_pbar=False, device=gpu).run()

                combined_s = xp.zeros((N_time, base_res, base_res), dtype=xp.complex64)
                images_s = xp.zeros_like(combined_s)
                for t in range(N_time):
                    kspace_shifted_t = xp.fft.fftshift(k_cart_all_t[t][s], axes=(-2, -1))
                    combined_s[t] = espirit_combine(kspace_shifted_t, mps)
                    images_s[t] = sp.ifft(combined_s[t], axes=(-2, -1))

                combined_np = backend.to_device(combined_s, sp.cpu_device)
                ds_k[s] = combined_np.astype(np.complex64, copy=False)

                images_np = backend.to_device(images_s, sp.cpu_device)
                if save_magnitude:
                    ds_i[s] = np.abs(images_np).astype(img_dtype, copy=False)
                else:
                    ds_i[s] = images_np.astype(img_dtype, copy=False)

    print("  Done.")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default=IN_DIR)
    p.add_argument("--pattern", type=str, default="*_2.h5")
    p.add_argument("--n_spacing", type=int, default=1)
    p.add_argument("--n_time", type=int, default=3)
    p.add_argument("--base_res", type=int, default=320)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    print(f"Found {len(files)} files in {args.in_dir} matching '{args.pattern}'")
    for f in files:
        process_h5_file_fast(
            f,
            n_spacing=args.n_spacing,
            N_time=args.n_time,
            base_res=args.base_res,
            device_id=args.device,
        )


if __name__ == "__main__":
    main()


