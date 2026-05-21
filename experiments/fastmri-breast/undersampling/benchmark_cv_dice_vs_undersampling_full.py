"""Evaluate full-dataset CV models under various undersampling rates.

Uses the best-Dice checkpoint for Dice evaluation and the best-AUROC
checkpoint for AUROC evaluation, so each metric is reported from its
optimal model.

Reports:
  - Dice on positive patients only (from best-Dice checkpoint)
  - AUROC for lesion detection (from best-AUROC checkpoint)
"""
import sys
sys.path.append("/home/l721f/code/kspace-pred-net")

import csv
import glob
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from fastmri.data.subsample import RandomMaskFunc
from fastmri.data.transforms import apply_mask

from src.data.fastmri_3d_full import FastMRIBreast3DKSpaceFullDataModule
from src.model.lightning_full import LitModelFullDataset
from src.utils.kspace_ops import kspace_to_pixel_probs, mask_to_onehot_like

# ---------- CONFIG ----------
DATASET_DIR = "/home/l721f/data/fastmri-breast/training_data_full"
RESULTS_BASE = "/home/l721f/code/kspace-pred-net/results/fastmri-breast"
OUT_DIR = f"{RESULTS_BASE}/undersampling_benchmark_cv_full"
BATCH_SIZE = 16
PATCH_DEPTH = 24
PATCH_STRIDE = 16
NUM_WORKERS = 2
N_FOLDS = 5
SEED = 123
GLOBAL_SEED = 123

SPECS = [
    (1, 1.0),  # no undersampling
    (2, 0.04), (4, 0.08), (6, 0.05), (8, 0.04),
    (10, 0.03), (12, 0.02), (16, 0.015), (24, 0.008),
    (32, 0.004), (48, 0.002), (64, 0.001),
]

NOISE_SNRS = [20.0, 10.0, 5.0, 0.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]

# model_name -> (cv_dir_suffix, model_class_name)
MODELS = {
    "UNet3D": ("UNet3D", "UNet3D"),
    "UNet3D_K2Img_noHWdown": ("UNet3D_K2Img_noHWdown", "UNet3D_K2Img"),
    "UNet3D_iFFT_complex": ("UNet3D_iFFT_complex", "UNet3D_iFFT"),
    "UNet3D_iFFT_mag": ("UNet3D_iFFT_mag", "UNet3D_iFFT"),
}


def get_patient_splits(dataset_dir: str, n_folds: int = 5, seed: int = 123):
    """Stratified splits matching training (by has_lesion)."""
    csv_path = os.path.join(dataset_dir, "patient_labels.csv")
    patient_ids = []
    labels = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_ids.append(row["patient_id"])
            labels.append(int(row["has_lesion"]))

    existing = set(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    filtered = [(pid, lab) for pid, lab in zip(patient_ids, labels) if pid in existing]
    patient_ids = [x[0] for x in filtered]
    labels = [x[1] for x in filtered]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in skf.split(patient_ids, labels):
        splits.append({
            "train": [patient_ids[i] for i in train_idx],
            "val": [patient_ids[i] for i in val_idx],
        })
    return splits, dict(zip(patient_ids, labels))


def load_patient_labels(dataset_dir: str):
    """Load has_lesion flags from patient_labels.csv."""
    csv_path = os.path.join(dataset_dir, "patient_labels.csv")
    has_lesion = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_lesion[row["patient_id"]] = int(row["has_lesion"]) > 0
    return has_lesion


def mix_seed(accel, patient_index, z_start):
    s = GLOBAL_SEED & 0xFFFFFFFF
    s = (s + 1000003 * accel) & 0xFFFFFFFF
    s = (s + 10007 * patient_index) & 0xFFFFFFFF
    s = (s + 101 * z_start) & 0xFFFFFFFF
    return s


class UndersampleDataset(Dataset):
    def __init__(self, base, accel, center_frac):
        self.base = base
        self.accel = int(accel)
        self.center_frac = float(center_frac)
        self.mask_func = RandomMaskFunc([float(center_frac)], [int(accel)]) if accel > 1 else None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if self.accel == 1:
            return sample
        x = sample["input"]["data"]
        meta = sample["meta"]
        seed = mix_seed(self.accel, meta["patient_index"].item(), meta["z_start"].item())
        x5 = x.permute(0, 2, 3, 4, 1).contiguous()
        x5_masked, _, _ = apply_mask(x5, self.mask_func, seed=seed)
        sample["input"]["data"] = x5_masked.permute(0, 4, 1, 2, 3).contiguous()
        return sample


class NoisyKspaceDataset(Dataset):
    """Adds complex Gaussian k-space noise at a given SNR (dB). No undersampling."""

    def __init__(self, base, snr_db):
        self.base = base
        self.snr_db = float(snr_db)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        x = sample["input"]["data"]
        meta = sample["meta"]
        seed = mix_seed(
            int(self.snr_db * 100) & 0xFFFFFFFF,
            meta["patient_index"].item(),
            meta["z_start"].item(),
        )
        rng = torch.Generator()
        rng.manual_seed(seed)
        signal_power = x.pow(2).mean()
        noise_power = signal_power / (10.0 ** (self.snr_db / 10.0))
        noise = torch.randn(x.shape, generator=rng) * noise_power.sqrt()
        sample["input"]["data"] = x + noise
        return sample


def find_ckpt(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def evaluate_dice(model, loader, device, idx_to_pid, has_lesion_map, skip_ifft=False):
    """Per-patient volume Dice (positive-only)."""
    model.eval()
    cur_pid = None
    cur_sum = None
    cur_count = None
    cur_gt = None
    case_dice = []

    def finalize():
        nonlocal cur_pid, cur_sum, cur_count, cur_gt
        if cur_pid is None:
            return
        count_b = cur_count.reshape(-1, 1, 1).astype(np.uint16)
        pred = (cur_sum.astype(np.uint16) * 2 > count_b)
        g = cur_gt.astype(bool)
        tp = int(np.logical_and(pred, g).sum())
        fp = int(np.logical_and(pred, ~g).sum())
        fn = int(np.logical_and(~pred, g).sum())
        denom = 2 * tp + fp + fn
        d = float("nan") if denom == 0 else (2.0 * tp) / denom
        case_dice.append((cur_pid, d))
        cur_pid = None

    with torch.no_grad():
        for batch in loader:
            x = batch["input"]["data"].to(device).float()
            y_mask = batch["label_mask"]["data"].to(device)
            meta = batch.get("meta", {})

            y_hat, _ = model.infer_batch({
                "input": {"data": x},
                "label": {"data": batch["label"]["data"].to(device).float()},
                "label_mask": {"data": y_mask},
                "skip_ifft": skip_ifft,
            })
            if model.output_image_logits:
                yhat_pix = torch.softmax(y_hat.float(), dim=1)
            else:
                yhat_pix = kspace_to_pixel_probs(y_hat)

            y_oh = mask_to_onehot_like(y_mask, yhat_pix, num_classes=2)
            yhat_oh = model.logits_to_mask(yhat_pix, y_oh.shape[1])

            y2 = y_oh.squeeze(2) if y_oh.shape[2] == 1 else y_oh
            yhat2 = yhat_oh.squeeze(2) if yhat_oh.shape[2] == 1 else yhat_oh

            pred_fg = (torch.argmax(yhat2, dim=1) == 1).to(torch.uint8).cpu().numpy()
            gt_fg = (torch.argmax(y2, dim=1) == 1).to(torch.uint8).cpu().numpy()

            pids = meta["patient_index"].detach().cpu().tolist()
            z_starts = meta["z_start"].detach().cpu().tolist()
            s_totals = meta["s_total"].detach().cpu().tolist()

            d_patch = int(pred_fg.shape[1])
            h = int(pred_fg.shape[2])
            w = int(pred_fg.shape[3])

            for i in range(int(pred_fg.shape[0])):
                pid = int(pids[i])
                z0 = int(z_starts[i])
                s_total = int(s_totals[i])

                if cur_pid is None:
                    cur_pid = pid
                    cur_sum = np.zeros((s_total, h, w), dtype=np.uint16)
                    cur_count = np.zeros((s_total,), dtype=np.uint16)
                    cur_gt = np.zeros((s_total, h, w), dtype=np.uint8)
                elif pid != cur_pid:
                    finalize()
                    cur_pid = pid
                    cur_sum = np.zeros((s_total, h, w), dtype=np.uint16)
                    cur_count = np.zeros((s_total,), dtype=np.uint16)
                    cur_gt = np.zeros((s_total, h, w), dtype=np.uint8)

                z_end = min(z0 + d_patch, s_total)
                if z_end <= z0:
                    continue
                dv = int(z_end - z0)
                cur_sum[z0:z_end] += pred_fg[i, :dv].astype(np.uint16, copy=False)
                cur_count[z0:z_end] += 1
                cur_gt[z0:z_end] = gt_fg[i, :dv]

    finalize()

    pos_dices = [d for pid, d in case_dice
                 if has_lesion_map.get(idx_to_pid.get(pid, str(pid)), False)]
    dice_mean = float(np.nanmean(pos_dices)) if pos_dices else float("nan")
    return dice_mean, case_dice


def evaluate_auroc(model, loader, device, idx_to_pid, has_lesion_map, skip_ifft=False):
    """Per-patient AUROC for lesion detection."""
    model.eval()
    cur_pid = None
    cur_sum = None
    cur_count = None
    cur_gt = None
    detection_rows = []

    def finalize():
        nonlocal cur_pid, cur_sum, cur_count, cur_gt
        if cur_pid is None:
            return
        count_b = cur_count.reshape(-1, 1, 1).astype(np.uint16)
        pred = (cur_sum.astype(np.uint16) * 2 > count_b)
        g = cur_gt.astype(bool)
        tp = int(np.logical_and(pred, g).sum())
        fp = int(np.logical_and(pred, ~g).sum())
        detect_score = float(tp + fp)
        pid_str = idx_to_pid.get(cur_pid, str(cur_pid))
        gt_label = 1 if has_lesion_map.get(pid_str, False) else 0
        detection_rows.append((cur_pid, gt_label, detect_score))
        cur_pid = None

    with torch.no_grad():
        for batch in loader:
            x = batch["input"]["data"].to(device).float()
            y_mask = batch["label_mask"]["data"].to(device)
            meta = batch.get("meta", {})

            y_hat, _ = model.infer_batch({
                "input": {"data": x},
                "label": {"data": batch["label"]["data"].to(device).float()},
                "label_mask": {"data": y_mask},
                "skip_ifft": skip_ifft,
            })
            if model.output_image_logits:
                yhat_pix = torch.softmax(y_hat.float(), dim=1)
            else:
                yhat_pix = kspace_to_pixel_probs(y_hat)

            y_oh = mask_to_onehot_like(y_mask, yhat_pix, num_classes=2)
            yhat_oh = model.logits_to_mask(yhat_pix, y_oh.shape[1])

            y2 = y_oh.squeeze(2) if y_oh.shape[2] == 1 else y_oh
            yhat2 = yhat_oh.squeeze(2) if yhat_oh.shape[2] == 1 else yhat_oh

            pred_fg = (torch.argmax(yhat2, dim=1) == 1).to(torch.uint8).cpu().numpy()
            gt_fg = (torch.argmax(y2, dim=1) == 1).to(torch.uint8).cpu().numpy()

            pids = meta["patient_index"].detach().cpu().tolist()
            z_starts = meta["z_start"].detach().cpu().tolist()
            s_totals = meta["s_total"].detach().cpu().tolist()

            d_patch = int(pred_fg.shape[1])
            h = int(pred_fg.shape[2])
            w = int(pred_fg.shape[3])

            for i in range(int(pred_fg.shape[0])):
                pid = int(pids[i])
                z0 = int(z_starts[i])
                s_total = int(s_totals[i])

                if cur_pid is None:
                    cur_pid = pid
                    cur_sum = np.zeros((s_total, h, w), dtype=np.uint16)
                    cur_count = np.zeros((s_total,), dtype=np.uint16)
                    cur_gt = np.zeros((s_total, h, w), dtype=np.uint8)
                elif pid != cur_pid:
                    finalize()
                    cur_pid = pid
                    cur_sum = np.zeros((s_total, h, w), dtype=np.uint16)
                    cur_count = np.zeros((s_total,), dtype=np.uint16)
                    cur_gt = np.zeros((s_total, h, w), dtype=np.uint8)

                z_end = min(z0 + d_patch, s_total)
                if z_end <= z0:
                    continue
                dv = int(z_end - z0)
                cur_sum[z0:z_end] += pred_fg[i, :dv].astype(np.uint16, copy=False)
                cur_count[z0:z_end] += 1
                cur_gt[z0:z_end] = gt_fg[i, :dv]

    finalize()

    gt_labels = [r[1] for r in detection_rows]
    det_scores = [r[2] for r in detection_rows]
    if len(set(gt_labels)) >= 2:
        try:
            auroc = float(roc_auc_score(gt_labels, det_scores))
        except Exception:
            auroc = float("nan")
    else:
        auroc = float("nan")

    return auroc, detection_rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits, lesion_status_map = get_patient_splits(DATASET_DIR, N_FOLDS, SEED)
    has_lesion_map = load_patient_labels(DATASET_DIR)
    accels = [s[0] for s in SPECS]

    all_dice = {name: {a: [] for a, _ in SPECS} for name in MODELS}
    # all_auroc = {name: {a: [] for a, _ in SPECS} for name in MODELS}
    per_patient_rows = []
    # detection_rows_all = []

    all_dice_noise = {name: {snr: [] for snr in NOISE_SNRS} for name in MODELS}
    per_patient_rows_noise = []

    for fold_idx, split in enumerate(splits, start=1):
        print(f"\n=== Fold {fold_idx}/{N_FOLDS} ===")
        dm = FastMRIBreast3DKSpaceFullDataModule(
            batch_size=BATCH_SIZE,
            dataset_dir=DATASET_DIR,
            patch_depth=PATCH_DEPTH,
            patch_stride=PATCH_STRIDE,
            predefined_patient_ids=split,
        )
        dm.setup()
        base_val = dm.val_set
        idx_to_pid = {v: k for k, v in dm.patient_id_to_index.items()}

        for model_name, (cv_suffix, model_class) in MODELS.items():
            ckpt_dir = f"{RESULTS_BASE}/3d_cv_{cv_suffix}_full/fold_{fold_idx}/{model_class}/checkpoints"

            # Best-Dice checkpoint (primary: best_model*)
            dice_ckpt = find_ckpt(os.path.join(ckpt_dir, "best_model*.ckpt"))
            # # Best-AUROC checkpoint
            # auroc_ckpt = find_ckpt(os.path.join(ckpt_dir, "best_auroc_model*.ckpt"))

            if not dice_ckpt:
                print(f"  {model_name}: no checkpoints found")
                continue

            for accel, frac in SPECS:
                ds = UndersampleDataset(base_val, accel, frac)
                loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

                # --- Dice from best-Dice checkpoint ---
                dice_mean = float("nan")
                case_dice = []
                if dice_ckpt:
                    model_dice = LitModelFullDataset.load_from_checkpoint(dice_ckpt, map_location="cpu", weights_only=False)
                    model_dice.to(device)
                    dice_mean, case_dice = evaluate_dice(model_dice, loader, device, idx_to_pid, has_lesion_map)
                    del model_dice
                    torch.cuda.empty_cache()

                # # --- AUROC from best-AUROC checkpoint ---
                # auroc = float("nan")
                # det_rows = []
                # ckpt_for_auroc = auroc_ckpt or dice_ckpt
                # if ckpt_for_auroc:
                #     model_auroc = LitModelFullDataset.load_from_checkpoint(ckpt_for_auroc, map_location="cpu", weights_only=False)
                #     model_auroc.to(device)
                #     auroc, det_rows = evaluate_auroc(model_auroc, loader, device, idx_to_pid, has_lesion_map)
                #     del model_auroc
                #     torch.cuda.empty_cache()

                all_dice[model_name][accel].append(dice_mean)
                # all_auroc[model_name][accel].append(auroc)
                print(f"    {model_name} {accel}x: dice={dice_mean:.4f}")

                for pidx, dice_val in case_dice:
                    pid_str = idx_to_pid.get(pidx, str(pidx))
                    per_patient_rows.append({
                        "model": model_name,
                        "acceleration": accel,
                        "center_fraction": frac,
                        "fold": fold_idx,
                        "patient_index": pidx,
                        "patient_id": pid_str,
                        "has_lesion": int(has_lesion_map.get(pid_str, False)),
                        "dice": dice_val,
                    })

                # for pidx, gt_label, max_fg in det_rows:
                #     pid_str = idx_to_pid.get(pidx, str(pidx))
                #     detection_rows_all.append({
                #         "model": model_name,
                #         "acceleration": accel,
                #         "center_fraction": frac,
                #         "fold": fold_idx,
                #         "patient_index": pidx,
                #         "patient_id": pid_str,
                #         "has_lesion": gt_label,
                #         "max_fg_prob": max_fg,
                #     })

            # --- K-space noise sweep (no undersampling, best-Dice ckpt) ---
            if dice_ckpt:
                for snr_db in NOISE_SNRS:
                    ds_noise = NoisyKspaceDataset(base_val, snr_db)
                    loader_noise = DataLoader(ds_noise, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

                    model_noise = LitModelFullDataset.load_from_checkpoint(dice_ckpt, map_location="cpu", weights_only=False)
                    model_noise.to(device)
                    dice_noise_mean, case_dice_noise = evaluate_dice(model_noise, loader_noise, device, idx_to_pid, has_lesion_map)
                    del model_noise
                    torch.cuda.empty_cache()

                    all_dice_noise[model_name][snr_db].append(dice_noise_mean)
                    print(f"    {model_name} snr={snr_db:>5.1f}dB: dice={dice_noise_mean:.4f}")

                    for pidx, dice_val in case_dice_noise:
                        pid_str = idx_to_pid.get(pidx, str(pidx))
                        per_patient_rows_noise.append({
                            "model": model_name,
                            "snr_db": float(snr_db),
                            "fold": fold_idx,
                            "patient_index": pidx,
                            "patient_id": pid_str,
                            "has_lesion": int(has_lesion_map.get(pid_str, False)),
                            "dice": dice_val,
                        })

    # --- Aggregate and save ---
    rows = []
    for model_name in MODELS:
        for accel, frac in SPECS:
            fold_dice = all_dice[model_name][accel]
            # fold_auroc = all_auroc[model_name][accel]
            dm = float(np.nanmean(fold_dice)) if fold_dice else float("nan")
            ds = float(np.nanstd(fold_dice, ddof=1)) if len(fold_dice) > 1 else 0.0
            # am = float(np.nanmean(fold_auroc)) if fold_auroc else float("nan")
            # astd = float(np.nanstd(fold_auroc, ddof=1)) if len(fold_auroc) > 1 else 0.0
            rows.append({
                "model": model_name,
                "acceleration": accel,
                "center_fraction": frac,
                "dice_mean": dm,
                "dice_std": ds,
                # "auroc_mean": am,
                # "auroc_std": astd,
                "fold_dice": ";".join(f"{x:.6f}" for x in fold_dice),
                # "fold_auroc": ";".join(f"{x:.6f}" for x in fold_auroc),
            })

    csv_path = os.path.join(OUT_DIR, "dice_auroc_vs_accel.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "acceleration", "center_fraction",
            "dice_mean", "dice_std",
            # "auroc_mean", "auroc_std",
            "fold_dice",
            # "fold_auroc",
        ])
        w.writeheader()
        w.writerows(rows)

    # Per-patient Dice CSV
    patient_csv_path = os.path.join(OUT_DIR, "dice_per_patient.csv")
    with open(patient_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "acceleration", "center_fraction", "fold",
            "patient_index", "patient_id", "has_lesion", "dice",
        ])
        w.writeheader()
        w.writerows(per_patient_rows)

    # # Detection CSV
    # det_csv_path = os.path.join(OUT_DIR, "detection_per_patient.csv")
    # with open(det_csv_path, "w", newline="") as f:
    #     w = csv.DictWriter(f, fieldnames=[
    #         "model", "acceleration", "center_fraction", "fold",
    #         "patient_index", "patient_id", "has_lesion", "max_fg_prob",
    #     ])
    #     w.writeheader()
    #     w.writerows(detection_rows_all)

    # --- Noise sweep CSVs ---
    noise_rows = []
    for model_name in MODELS:
        for snr_db in NOISE_SNRS:
            fold_dice = all_dice_noise[model_name][snr_db]
            dm_val = float(np.nanmean(fold_dice)) if fold_dice else float("nan")
            ds_val = float(np.nanstd(fold_dice, ddof=1)) if len(fold_dice) > 1 else 0.0
            noise_rows.append({
                "model": model_name,
                "snr_db": snr_db,
                "dice_mean": dm_val,
                "dice_std": ds_val,
                "fold_dice": ";".join(f"{x:.6f}" for x in fold_dice),
            })

    noise_csv_path = os.path.join(OUT_DIR, "noise_dice_vs_snr.csv")
    with open(noise_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "snr_db", "dice_mean", "dice_std", "fold_dice",
        ])
        w.writeheader()
        w.writerows(noise_rows)

    noise_patient_csv_path = os.path.join(OUT_DIR, "noise_dice_per_patient.csv")
    with open(noise_patient_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "snr_db", "fold",
            "patient_index", "patient_id", "has_lesion", "dice",
        ])
        w.writeheader()
        w.writerows(per_patient_rows_noise)

    # --- Plots ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for name in MODELS:
        dice_means = [float(np.nanmean(all_dice[name][a])) if all_dice[name][a] else float("nan") for a, _ in SPECS]
        dice_stds = [float(np.nanstd(all_dice[name][a], ddof=1)) if len(all_dice[name][a]) > 1 else 0.0 for a, _ in SPECS]
        if any(not np.isnan(v) for v in dice_means):
            ax.errorbar(accels, dice_means, yerr=dice_stds, marker="o", capsize=3, label=name)

    ax.set_xlabel("Acceleration factor")
    ax.set_ylabel("Dice (positive patients, best-Dice ckpt)")
    ax.set_title("Dice vs Undersampling")
    ax.set_xticks(accels)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    # for name in MODELS:
    #     auroc_means = [float(np.nanmean(all_auroc[name][a])) if all_auroc[name][a] else float("nan") for a, _ in SPECS]
    #     auroc_stds = [float(np.nanstd(all_auroc[name][a], ddof=1)) if len(all_auroc[name][a]) > 1 else 0.0 for a, _ in SPECS]
    #     if any(not np.isnan(v) for v in auroc_means):
    #         axes[1].errorbar(accels, auroc_means, yerr=auroc_stds, marker="o", capsize=3, label=name)
    #
    # axes[1].set_xlabel("Acceleration factor")
    # axes[1].set_ylabel("AUROC (best-AUROC ckpt)")
    # axes[1].set_title("AUROC vs Undersampling")
    # axes[1].set_xticks(accels)
    # axes[1].tick_params(axis="x", rotation=45)
    # axes[1].grid(True, alpha=0.3)
    # axes[1].legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "dice_auroc_vs_accel.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    # --- Noise plot ---
    fig_n, ax_n = plt.subplots(1, 1, figsize=(7, 5))
    for name in MODELS:
        dice_means = [float(np.nanmean(all_dice_noise[name][s])) if all_dice_noise[name][s] else float("nan") for s in NOISE_SNRS]
        dice_stds = [float(np.nanstd(all_dice_noise[name][s], ddof=1)) if len(all_dice_noise[name][s]) > 1 else 0.0 for s in NOISE_SNRS]
        if any(not np.isnan(v) for v in dice_means):
            ax_n.errorbar(NOISE_SNRS, dice_means, yerr=dice_stds, marker="o", capsize=3, label=name)
    ax_n.set_xlabel("SNR (dB)")
    ax_n.set_ylabel("Dice (positive patients, best-Dice ckpt)")
    ax_n.set_title("Dice vs K-space Noise")
    ax_n.invert_xaxis()
    ax_n.grid(True, alpha=0.3)
    ax_n.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    noise_png_path = os.path.join(OUT_DIR, "noise_dice_vs_snr.png")
    plt.savefig(noise_png_path, dpi=150)
    plt.close()

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {patient_csv_path}")
    # print(f"Saved: {det_csv_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {noise_csv_path}")
    print(f"Saved: {noise_patient_csv_path}")
    print(f"Saved: {noise_png_path}")


if __name__ == "__main__":
    main()
