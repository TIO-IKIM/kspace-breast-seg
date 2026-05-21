import os
import argparse
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt


def find_positive_slice(label_stack: np.ndarray, min_positive_voxels: int = 20) -> int:
    """
    Return index of a positive slice (max positive voxels among those > threshold).

    label_stack: (S, H, W) uint8
    """
    pos_counts = label_stack.reshape(label_stack.shape[0], -1).sum(axis=1)
    pos_indices = np.where(pos_counts > min_positive_voxels)[0]
    if pos_indices.size == 0:
        return -1
    # Pick the slice with the most positive voxels
    best_local = pos_indices[np.argmax(pos_counts[pos_indices])]
    return int(best_local)


def ifft_mag_from_shifted_kspace(k_real: np.ndarray, k_imag: np.ndarray) -> np.ndarray:
    """
    Given fftshifted full-complex k-space (real/imag), compute magnitude image.
    Inputs are (H, W) arrays.
    """
    k_complex = k_real.astype(np.float32) + 1j * k_imag.astype(np.float32)
    k_unshift = fft.ifftshift(k_complex, axes=(-2, -1))
    img_c = fft.ifft2(k_unshift, norm="ortho")
    return np.abs(img_c).astype(np.float32)


def percentile_symmetric_limits(arr: np.ndarray, pct: float = 99.0) -> float:
    """
    Symmetric visualization limits around zero using percentile of absolute values.
    """
    a = np.abs(arr).astype(np.float32)
    if a.size == 0:
        return 1.0
    return float(np.percentile(a, pct)) + 1e-6


def normalize01(img: np.ndarray) -> np.ndarray:
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - vmin) / (vmax - vmin)).astype(np.float32)


def plot_kspace_magnitude(ax,
                          real: np.ndarray,
                          imag: np.ndarray,
                          title: str,
                          mag_pct: float = 99.0,
                          use_log_value: bool = True) -> None:
    mag = np.sqrt(real.astype(np.float32) ** 2 + imag.astype(np.float32) ** 2)
    vmax = float(np.percentile(mag, mag_pct)) + 1e-6
    if use_log_value:
        img = np.log1p(mag) / np.log1p(vmax)
    else:
        img = np.clip(mag / vmax, 0.0, 1.0)
    img = np.clip(img, 0.0, 1.0)
    ax.imshow(img, cmap='gray', vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis('off')


def make_figure(dataset_dir: str,
                output_path: str,
                patient_id: str | None = None,
                slice_index: int | None = None,
                min_positive_voxels: int = 20,
                mag_pct: float = 99.0,
                use_log_value: bool = True,
                ) -> str:
    """
    Create a 2x3 panel figure for one positive slice:
    [A] DCE subtraction image (reconstructed from input k-space)
    [B] True pixel-space mask
    [C] Input k-space real
    [D] Input k-space imag
    [E] True k-space mask real (foreground)
    [F] True k-space mask imag (foreground)
    """
    # Determine patient directory
    if patient_id is None:
        candidates = sorted([d for d in os.listdir(dataset_dir)
                             if os.path.isdir(os.path.join(dataset_dir, d))])
    else:
        candidates = [patient_id]

    chosen = None
    chosen_slice = None

    for pid in candidates:
        pdir = os.path.join(dataset_dir, pid)
        lbl_path = os.path.join(pdir, "label_mask_stack.npy")
        inp_path = os.path.join(pdir, "input_kspace_stack.npy")
        tgt_path = os.path.join(pdir, "target_kspace_stack.npy")
        if not (os.path.isfile(lbl_path) and os.path.isfile(inp_path) and os.path.isfile(tgt_path)):
            continue

        label_stack = np.load(lbl_path).astype(np.uint8)  # (S, H, W)
        if slice_index is None:
            sidx = find_positive_slice(label_stack, min_positive_voxels)
        else:
            sidx = int(slice_index)
        if sidx < 0 or sidx >= label_stack.shape[0]:
            continue

        chosen = pid
        chosen_slice = sidx
        # Load corresponding k-space stacks
        input_kspace = np.load(inp_path).astype(np.float32)   # (S, 2, H, W)
        target_kspace = np.load(tgt_path).astype(np.float32)  # (S, 2, H, W) foreground-only
        gt_mask = label_stack[chosen_slice]                    # (H, W)
        k_in_real = input_kspace[chosen_slice, 0]
        k_in_imag = input_kspace[chosen_slice, 1]
        k_fg_real = target_kspace[chosen_slice, 0]
        k_fg_imag = target_kspace[chosen_slice, 1]

        # Reconstruct DCE subtraction image magnitude
        dce_mag = ifft_mag_from_shifted_kspace(k_in_real, k_in_imag)
        dce_mag_n = normalize01(dce_mag)

        # Build 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

        axes[0, 0].imshow(dce_mag_n, cmap='gray', vmin=0.0, vmax=1.0)
        axes[0, 0].set_title("A. DCE subtraction (|IFFT|)")
        axes[0, 0].axis('off')

        plot_kspace_magnitude(axes[0, 1], k_in_real, k_in_imag,
                              title="B. Input k-space magnitude",
                              mag_pct=mag_pct,
                              use_log_value=use_log_value)

        axes[1, 0].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title("C. Ground-truth mask (pixel space)")
        axes[1, 0].axis('off')

        plot_kspace_magnitude(axes[1, 1], k_fg_real, k_fg_imag,
                              title="D. True k-space mask FG magnitude",
                              mag_pct=mag_pct,
                              use_log_value=use_log_value)

        sup_title = f"Patient {chosen} – slice {chosen_slice}"
        fig.suptitle(sup_title, fontsize=12)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved figure: {output_path}")
        return output_path

    raise RuntimeError("No suitable positive slice found. Consider lowering --min-positive-voxels or specifying --patient-id/--slice.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a demonstration figure for one positive slice.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/l721f/data/mama-mia/images_preproc_kspace_slices",
        help="Root directory containing per-patient preprocessed slice stacks.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/l721f/code/kspace-pred-net/results/figures/positive_slice_demo.png",
        help="Path to save the output figure.",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Specific patient ID to use (must exist under dataset-dir).",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="Specific slice index to use (bypasses positive-slice search).",
    )
    parser.add_argument(
        "--min-positive-voxels",
        type=int,
        default=20,
        help="Minimum positive voxels to consider a slice positive.",
    )
    parser.add_argument(
        "--mag-pct",
        type=float,
        default=99.0,
        help="Percentile for magnitude clipping (improves visibility).",
    )
    parser.add_argument(
        "--no-log-value",
        action="store_true",
        help="Disable log scaling of magnitude for HSV value channel.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        patient_id=args.patient_id,
        slice_index=args.slice,
        min_positive_voxels=args.min_positive_voxels,
        mag_pct=args.mag_pct,
        use_log_value=(not args.no_log_value),
    )


