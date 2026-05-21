import os
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger


class PositiveSliceVisualizer(pl.Callback):
    def __init__(
        self,
        figures_dir: str,
        every_n_epochs: int = 1,
        num_slices: int = 3,
        min_positive_voxels: int = 20,
        max_batches_to_search: int = 20,
        min_slice_gap: int = 2,
        per_patient_limit: int = 2,
        seed: int = 54,
        overlay_alpha: float = 0.35,
    ) -> None:
        super().__init__()
        self.figures_dir = figures_dir
        self.every_n_epochs = every_n_epochs
        self.num_slices = num_slices
        self.min_positive_voxels = min_positive_voxels
        self.max_batches_to_search = max_batches_to_search
        self.min_slice_gap = min_slice_gap
        self.per_patient_limit = per_patient_limit
        self.seed = seed
        self.overlay_alpha = overlay_alpha
        os.makedirs(self.figures_dir, exist_ok=True)
        self._saved_this_epoch = 0
        self._saved_per_patient = {}

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._saved_this_epoch = 0
        self._saved_per_patient = {}

    def _ifft_magnitude_from_kspace(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C, V=2, X>=1, H, W) with centered k-space (DC at center)
        # For visualization we:
        #   - take the center slice along X
        #   - if multiple timepoints/channels are present (C > 1), use the second one
        #   - return magnitude image (B, H, W)
        b, c, v, xdim, h, w = inputs.shape
        center = xdim // 2
        k_center = inputs[:, :, :, center, ...]  # (B, C, V, H, W)

        # Choose which timepoint/channel to visualize: second if available, else first
        t_idx = 1 if c > 1 else 0
        k_tp = k_center[:, t_idx, :, :, :]  # (B, V=2, H, W)

        # Ensure float32 FFT to avoid cuFFT half-precision restrictions
        real = k_tp[:, 0, :, :].to(torch.float32)
        imag = k_tp[:, 1, :, :].to(torch.float32)
        if real.is_cuda:
            # Disable any outer autocast to keep FFT in float32
            with torch.amp.autocast('cuda', enabled=False):
                complex_k = torch.complex(real, imag)
                shifted = torch.fft.ifftshift(complex_k, dim=(-2, -1))
                img_c = torch.fft.ifft2(shifted, norm="ortho")
                img_c = torch.fft.fftshift(img_c, dim=(-2, -1))
        else:
            complex_k = torch.complex(real, imag)
            shifted = torch.fft.ifftshift(complex_k, dim=(-2, -1))
            img_c = torch.fft.ifft2(shifted, norm="ortho")
            img_c = torch.fft.fftshift(img_c, dim=(-2, -1))
        return torch.abs(img_c)

    def _pred_probs_pixel(self, pl_module, inputs: torch.Tensor) -> torch.Tensor:
        # Predict pixel-domain class probabilities for visualization.
        # For k-space models, map k-space outputs via fixed iFFT; for image-space
        # (iFFT-front-end) models, use softmax over image logits.
        x = inputs.to(torch.float32)

        if getattr(pl_module, 'output_image_logits', False):
            # Models that produce image-domain logits.
            if getattr(pl_module, 'input_ifft_frontend', False):
                # Image-space UNet with fixed iFFT front-end (e.g. UNet3D_iFFT)
                real = x[:, :, 0, ...]
                imag = x[:, :, 1, ...]
                if real.is_cuda:
                    with torch.amp.autocast('cuda', enabled=False):
                        complex_vol = torch.complex(real, imag)
                        shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
                        img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
                        img_c = torch.fft.fftshift(img_c, dim=(-2, -1))
                else:
                    complex_vol = torch.complex(real, imag)
                    shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
                    img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
                    img_c = torch.fft.fftshift(img_c, dim=(-2, -1))

                # Match the model's expected V dimension (magnitude vs complex Re/Im).
                v_expected = int(getattr(getattr(pl_module, "net", None), "input_shape", (None, 1))[1] or 1)
                if v_expected == 1:
                    img_mag = torch.abs(img_c)  # (B, C_in, D, H, W)
                    x_in = img_mag.unsqueeze(2)  # (B, C_in, 1, D, H, W)
                else:
                    x_in = torch.stack([img_c.real, img_c.imag], dim=2)  # (B, C_in, 2, D, H, W)
            else:
                # e.g. UNet3D_K2Img: consumes k-space directly
                x_in = x

            out = pl_module.net(x_in)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                logits = out[1]
            elif isinstance(out, dict) and "seg_logits" in out:
                logits = out["seg_logits"]
            else:
                logits = out
            preds_pix = torch.softmax(logits.float(), dim=1)  # (B, C_out, 1, D, H, W)
        else:
            # K-space models: map k-space predictions to pixel space via helper
            out = pl_module.net(x)
            if isinstance(out, (tuple, list)) and len(out) >= 1:
                preds_k = out[0]
            elif isinstance(out, dict) and "kspace" in out:
                preds_k = out["kspace"]
            else:
                preds_k = out
            preds_pix = pl_module.kspace_to_pixel_onehot([preds_k])[0]

        # Handle 2D and 3D: if depth (X/D) > 1, take center-depth slice
        if preds_pix.shape[3] > 1:
            center = preds_pix.shape[3] // 2
            preds_pix = preds_pix[:, :, :, center:center + 1, ...]
        preds_pix = preds_pix.squeeze(2).squeeze(2)
        denom = torch.sum(preds_pix, dim=1, keepdim=True) + 1e-6
        return preds_pix / denom

    def _dice(self, pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> float:
        # pred_bin, gt_bin: (H, W) in {0,1}
        inter = (pred_bin & gt_bin).sum().item()
        denom = pred_bin.sum().item() + gt_bin.sum().item()
        return (2.0 * inter) / denom if denom > 0 else 0.0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        # Only log on desired epochs
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if self._saved_this_epoch >= self.num_slices:
            return

        # Work on the batch already fetched by Lightning (avoid own DataLoader iteration)
        true_masks = batch['label_mask']['data'].to(pl_module.device)
        inputs = batch['input']['data'].to(pl_module.device)

        # If 3D patches are provided, reduce to the center-depth slice for visualization
        center_depth = 0
        if inputs.shape[3] > 1:
            center_depth = inputs.shape[3] // 2
        if true_masks.ndim == 4:
            true_masks2d = true_masks[:, center_depth, ...]
        else:
            true_masks2d = true_masks

        # Identify positive slices
        pos_counts = true_masks2d.sum(dim=(1, 2))
        pos_indices = torch.where(pos_counts > self.min_positive_voxels)[0]
        if len(pos_indices) == 0:
            return

        rng = torch.Generator(device=pl_module.device).manual_seed(self.seed + batch_idx)
        perm = torch.randperm(pos_indices.numel(), generator=rng, device=pl_module.device)
        pos_indices = pos_indices[perm]

        img_mag = self._ifft_magnitude_from_kspace(inputs).cpu()
        pred_probs = self._pred_probs_pixel(pl_module, inputs).cpu()

        # Match reconstructed sizes to ground-truth mask size for fair Dice/overlay
        gh, gw = int(true_masks2d.shape[-2]), int(true_masks2d.shape[-1])
        if img_mag.shape[-2:] != (gh, gw):
            img_mag = torch.nn.functional.interpolate(
                img_mag.unsqueeze(1), size=(gh, gw), mode='bilinear', align_corners=False
            ).squeeze(1)
        if pred_probs.shape[-2:] != (gh, gw):
            pred_probs = torch.nn.functional.interpolate(
                pred_probs, size=(gh, gw), mode='bilinear', align_corners=False
            )

        meta = batch.get('meta', {})
        if isinstance(meta, dict) and 'patient_index' in meta and 'slice_idx' in meta:
            patient_indices = meta['patient_index'].detach().cpu().tolist()
            slice_idxs = meta['slice_idx'].detach().cpu().tolist()
        elif isinstance(meta, dict) and 'patient_index' in meta and 'z_start' in meta:
            patient_indices = meta['patient_index'].detach().cpu().tolist()
            z_starts = meta['z_start'].detach().cpu().tolist()
            slice_idxs = [int(z0) + int(center_depth) for z0 in z_starts]
        else:
            patient_indices = [batch_idx for _ in range(inputs.shape[0])]
            slice_idxs = list(range(inputs.shape[0]))

        used_pairs = set()  # to enforce non-adjacent slices per patient within this batch

        for idx_t in pos_indices:
            if self._saved_this_epoch >= self.num_slices:
                break
            i = int(idx_t.item())
            pid_idx = patient_indices[i]
            sidx = int(slice_idxs[i])

            # Enforce within-batch non-adjacency and per-patient cap per epoch
            key = (pid_idx, sidx)
            # Per-patient limit across the whole epoch
            if self.per_patient_limit and self.per_patient_limit > 0:
                if self._saved_per_patient.get(pid_idx, 0) >= self.per_patient_limit:
                    continue
            if any(abs(sidx - other_s) < self.min_slice_gap for (p, other_s) in used_pairs if p == pid_idx):
                continue

            inp_img = img_mag[i].float()
            gt_mask = true_masks2d[i].cpu().float()
            prob_lesion = pred_probs[i, 1].float()

            inp_img = (inp_img - inp_img.min()) / (inp_img.max() - inp_img.min() + 1e-6)

            # Use argmax across classes (consistent with validation metrics)
            pred_bin = (torch.argmax(pred_probs, dim=1)[i] == 1)
            gt_bin = (gt_mask > 0.5).to(torch.bool)
            dice_val = self._dice(pred_bin, gt_bin)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(inp_img, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title('Input Image'); axes[0].axis('off')

            axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Ground Truth Mask'); axes[1].axis('off')

            im = axes[2].imshow(prob_lesion, cmap='viridis', vmin=0, vmax=1)
            axes[2].set_title(f'Predicted Probability (Class 1)  Dice={dice_val:.3f}')
            axes[2].axis('off')
            
            # Final binarized predicted mask (argmax over classes)
            axes[3].imshow(pred_bin.float(), cmap='gray', vmin=0, vmax=1)
            axes[3].set_title('Predicted Mask (Binary)'); axes[3].axis('off')
            plt.tight_layout()

            save_path = os.path.join(
                self.figures_dir,
                f"val_epoch_{trainer.current_epoch+1}_p{pid_idx}_s{sidx}_{self._saved_this_epoch+1}.png",
            )
            fig.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

            logger = getattr(trainer, 'logger', None)
            if isinstance(logger, WandbLogger):
                # Prefer native logger API to avoid direct wandb dependency here
                logger.log_image(
                    key=f"validation/epoch_{trainer.current_epoch+1}/p{pid_idx}_s{sidx}",
                    images=[save_path],
                    caption=[f"Epoch {trainer.current_epoch+1} – p{pid_idx} s{sidx} Dice={dice_val:.3f}"]
                )

            plt.close(fig)

            self._saved_this_epoch += 1
            # Track per-patient saved count for this epoch
            self._saved_per_patient[pid_idx] = self._saved_per_patient.get(pid_idx, 0) + 1
            used_pairs.add((pid_idx, sidx))


