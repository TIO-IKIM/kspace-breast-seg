from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vfunctional
import torchvision.utils as vutils
from einops import rearrange
from torch.optim import Optimizer, lr_scheduler
from torchmetrics import Recall, Specificity
from pytorch_lightning.loggers import WandbLogger
from monai.metrics import DiceMetric

from src.model.modules import (
    UNet3D,
    UNet3D_DualPath,
    UNet3D_K2FeatIfftTinyHead,
    UNet3D_K2IfftHermitianLogits,
    UNet3D_K2IfftLogMagHead,
    UNet3D_K2Img,
    UNet3D_K2Img_Dilated,
    UNet3D_KspaceAttn,
)
from src.utils.kspace_ops import (
    kspace_to_pixel_probs,
    mask_to_onehot_like,
)


class LitModel(pl.LightningModule):
    def __init__(self, model_name: str, config: Dict[str, Any]) -> None:
        """Initialization of the custom Lightning Module.

        Args:
            model: Neural network model name.
            config: Neural network model and training config.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        # Flags are split so we can (a) optionally iFFT the *input* and (b)
        # independently specify whether the model outputs image logits.
        self.input_ifft_frontend = False
        self.output_image_logits = False
        self.lr = config['lr']
        self.criterion = config['criterion']
        self.optimizer_class = config['optimizer_class']
        self.step_size = config.get('step_size', None)
        self.gamma = config.get('gamma', 0.9)
        self.scheduler_class = config['scheduler_class']
        self.scheduler_kwargs = config.get('scheduler_kwargs', {})
        self.scheduler_monitor = config.get('scheduler_monitor', None)
        self.input_domain = config['input_domain']
        self.label_domain = config['label_domain']
        self.save_preds = bool(config.get('save_preds', False))
        # Optional prediction saving (volume-level outputs for reporting)
        self.predictions_dir = (
            Path(config['predictions_dir']) if 'predictions_dir' in config and config['predictions_dir'] else None
        )
        self.patient_index_to_id = config.get('patient_index_to_id', None)
        # MONAI Dice metrics (per-case, exclude background)
        self.val_dice_metric = DiceMetric(include_background=False, reduction="none", ignore_empty=True)
        self.test_dice_metric = DiceMetric(include_background=False, reduction="none", ignore_empty=True)

        # Persistent binary recall/specificity metrics (avoid re-allocation every step)
        self._recall_metric = Recall(task="binary")
        self._specificity_metric = Specificity(task="binary")

        # Volume-level (per-patient) reconstruction from 3D patches (z_start + overlap vote).
        self._test_vol_state = self._new_vol_state()
        self._val_vol_state = self._new_vol_state()

        
        if self.model_name == 'UNet3D':
            self.net = UNet3D(
                config['input_shape'],
                config['output_shape'],
                config.get('hidden_factor', 32),
                config.get('depth', 4),
                config.get('strides', None),
                kernel_size=config.get('kernel_size', 3),
                up_kernel_size=config.get('up_kernel_size', 3),
            )
        elif self.model_name == 'UNet3D_iFFT':
            # 3D UNet operating in image space with a fixed iFFT front-end
            self.input_ifft_frontend = True
            self.output_image_logits = True
            self.net = UNet3D(
                config['input_shape'],
                config['output_shape'],
                config.get('hidden_factor', 32),
                config.get('depth', 4),
                config.get('strides', None),
                kernel_size=config.get('kernel_size', 3),
                up_kernel_size=config.get('up_kernel_size', 3),
            )
        elif self.model_name == 'UNet3D_K2Img':
            # k-space -> (learned k-space features) -> iFFT2 -> image logits
            self.output_image_logits = True
            self.net = UNet3D_K2Img(
                config['input_shape'],
                config['output_shape'],
                config.get('hidden_factor', 24),
                config.get('depth', 4),
                config.get('mid_channels', 8),
                strides=config.get('strides', None),
                strides_k=config.get('strides_k', None),
                strides_img=config.get('strides_img', None),
                kernel_size=config.get('kernel_size', 3),
                up_kernel_size=config.get('up_kernel_size', 3),
                kernel_size_k=config.get('kernel_size_k', None),
                up_kernel_size_k=config.get('up_kernel_size_k', None),
                kernel_size_img=config.get('kernel_size_img', None),
                up_kernel_size_img=config.get('up_kernel_size_img', None),
                hidden_factor_k=config.get('hidden_factor_k', None),
                hidden_factor_img=config.get('hidden_factor_img', None),
                depth_k=config.get('depth_k', None),
            )
        elif self.model_name == 'UNet3D_K2Img_Dilated':
            # k-space Dilated ResNet (no down/up) -> iFFT2 -> image logits
            self.output_image_logits = True
            self.net = UNet3D_K2Img_Dilated(
                config['input_shape'],
                config['output_shape'],
                config.get('hidden_factor', 24),
                config.get('depth', 4),
                config.get('mid_channels', 8),
                strides=config.get('strides', None),
                kernel_size=config.get('kernel_size', 3),
                up_kernel_size=config.get('up_kernel_size', 3),
                num_res_blocks=config.get('num_res_blocks', 6),
            )
        elif self.model_name == 'UNet3D_K2IfftLogMagHead':
            # k-space UNet -> iFFT2 -> log(|img|) -> tiny calibration head (V=1)
            self.output_image_logits = True
            self.net = UNet3D_K2IfftLogMagHead(
                config['input_shape'],
                config['output_shape'],
                config.get('hidden_factor', 24),
                config.get('depth', 4),
                config.get('strides', None),
                eps=config.get('eps', 1e-6),
            )
        elif self.model_name == 'UNet3D_K2FeatIfftTinyHead':
            # k-space UNet -> iFFT2 -> tiny image head -> logits (V=1)
            self.output_image_logits = True
            self.net = UNet3D_K2FeatIfftTinyHead(
                config['input_shape'],
                config['output_shape'],
                config.get('hidden_factor', 16),
                config.get('depth', 3),
                config.get('mid_channels', 16),
                config.get('head_channels', 16),
                config.get('strides', None),
            )
        elif self.model_name == 'UNet3D_K2IfftHermitianLogits':
            # k-space UNet -> Hermitian projection -> iFFT2 -> real logits (V=1)
            self.output_image_logits = True
            self.net = UNet3D_K2IfftHermitianLogits(
                config['input_shape'],
                config['output_shape'],
                config.get('hidden_factor', 24),
                config.get('depth', 4),
                config.get('strides', None),
            )
        elif self.model_name == 'UNet3D_KspaceAttn':
            # K-space attention modulates iFFT image-space UNet
            self.input_ifft_frontend = False  # model handles iFFT internally
            self.output_image_logits = True
            self.net = UNet3D_KspaceAttn(
                config['input_shape'],
                config['output_shape'],
                config.get('hidden_factor', 24),
                config.get('depth', 4),
                config.get('attn_channels', 16),
                config.get('strides', None),
            )
        elif self.model_name == 'UNet3D_DualPath':
            # Spectral-spatial dual path: k-space 1x1 features + image UNet fusion
            self.input_ifft_frontend = False  # model handles iFFT internally
            self.output_image_logits = True
            self.net = UNet3D_DualPath(
                config['input_shape'],
                config['output_shape'],
                config.get('hidden_factor', 24),
                config.get('depth', 4),
                config.get('spectral_channels', 16),
                config.get('strides', None),
            )
        else:
            raise ValueError(f'Model {self.model_name} is not defined')

    @property
    def name(self) -> str:
        """Model name property.

        Returns:
            Model name.
        """
        return self.model_name

    def configure_optimizers(
        self,
    ) -> Tuple[Optimizer, lr_scheduler.LRScheduler]:
        """Configures the optimizer and scheduler based on the learning rate
            and step size.

        Returns:
            Configured optimizer and scheduler.
        """
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.scheduler_kwargs:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            scheduler_config = {"scheduler": scheduler}
            if self.scheduler_monitor:
                scheduler_config["monitor"] = self.scheduler_monitor
            return [optimizer], [scheduler_config]
        scheduler = self.scheduler_class(optimizer, self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]

    def infer_batch(
        self, batch: Dict[str, dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate given batch through the Lightning Module.

        Args:
            batch: Batch containing the subjects.

        Returns:
            Model output and corresponding ground truth.
        """
        x, y = batch['input']['data'], batch['label']['data']
        y = y.float()

        skip_ifft = batch.get('skip_ifft', False)

        # Optional fixed iFFT front-end for k-space volumes
        if self.input_ifft_frontend and not skip_ifft:
            # x: (B, C_in=1, V=2, D, H, W) k-space (real, imag)
            real = x[:, :, 0, ...].to(torch.float32)
            imag = x[:, :, 1, ...].to(torch.float32)

            if real.is_cuda:
                with torch.amp.autocast('cuda', enabled=False):
                    complex_vol = torch.complex(real, imag)  # (B, C_in, D, H, W)
                    shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
                    img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
                    img_c = torch.fft.fftshift(img_c, dim=(-2, -1))
            else:
                complex_vol = torch.complex(real, imag)
                shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
                img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
                img_c = torch.fft.fftshift(img_c, dim=(-2, -1))

            # Choose image representation based on the UNet's expected V dimension:
            #   - V=1: magnitude image |F^{-1}(k)|
            #   - V=2: complex image (real/imag)
            v_expected = int(getattr(self.net, "input_shape", (None, 1))[1] or 1)
            if v_expected == 1:
                img_mag = torch.abs(img_c)  # (B, C_in, D, H, W)
                x_in = img_mag.unsqueeze(2)  # (B, C_in, 1, D, H, W)
            else:
                x_in = torch.stack([img_c.real, img_c.imag], dim=2)  # (B, C_in, 2, D, H, W)
        elif self.input_ifft_frontend and skip_ifft:
            # Input is already image-domain complex: (B, C, V=2, D, H, W)
            v_expected = int(getattr(self.net, "input_shape", (None, 1))[1] or 1)
            if v_expected == 1:
                real = x[:, :, 0, ...].to(torch.float32)
                imag = x[:, :, 1, ...].to(torch.float32)
                x_in = torch.sqrt(real ** 2 + imag ** 2).unsqueeze(2)
            else:
                x_in = x
        else:
            x_in = x

        y_hat = self.net(x_in)
        return y_hat, y

    def _mask_to_onehot_like(self, mask: torch.Tensor, like: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
        # Shared utility (kept method for backward compatibility in this module)
        return mask_to_onehot_like(mask, like, num_classes=num_classes)

    def training_step(self, batch: Dict[str, dict], batch_idx: int) -> float:
        """Infer batch on training data, log metrics and retrieve loss.

        Args:
            batch: Batch containing the subjects.
            batch_idx: Number displaying index of this batch.

        Returns:
            Calculated loss.
        """
        y_hat, y = self.infer_batch(batch)

        # Calculate loss in float32 with AMP disabled to avoid fp16 NaNs
        criterion = self.criterion
        if y_hat.is_cuda:
            with torch.amp.autocast('cuda', enabled=False):
                if getattr(criterion, 'requires_label_mask', False):
                    loss = criterion(y_hat.float(), y.float(), label_mask=batch['label_mask']['data'])
                else:
                    loss = criterion(y_hat.float(), y.float())
        else:
            if getattr(criterion, 'requires_label_mask', False):
                loss = criterion(y_hat.float(), y.float(), label_mask=batch['label_mask']['data'])
            else:
                loss = criterion(y_hat.float(), y.float())

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def kspace_to_pixel_onehot(self, variables: List[torch.Tensor], is_input_image: bool = False) -> List[torch.Tensor]:
        transformed_variables = []
        for var in variables:
            # Default path for complex k-space data
            # var shape: (B, C, V=2, X, Y, Z)
            # 2D models (UNet/UNetMonai): take center slice along X and iFFT2 over (Y, Z)
            # 3D model (UNet3D): iFFTN over (X, Y, Z) to get (D, H, W)
            if self.model_name in (
                'UNet3D',
                'UNet3D_iFFT',
                'UNet3D_K2Img',
                'UNet3D_K2Img_Dilated',
                'UNet3D_K2IfftHermitianLogits',
            ):
                # Build complex volume and invert centered k-space: ifftshift -> ifft2
                real = var[:, :, 0, ...].to(torch.float32)
                imag = var[:, :, 1, ...].to(torch.float32)
                if real.is_cuda:
                    with torch.amp.autocast('cuda', enabled=False):
                        complex_vol = torch.complex(real, imag)  # (B, C, D, H, W)
                        shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
                        img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
                        img_c = torch.fft.fftshift(img_c, dim=(-2, -1))
                else:
                    complex_vol = torch.complex(real, imag)
                    shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
                    img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
                    img_c = torch.fft.fftshift(img_c, dim=(-2, -1))

                pixel_space_mag = torch.abs(img_c)  # (B, C, D, H, W)
                # If single-class (FG-only), synthesize background channel from normalized FG
                if not is_input_image and pixel_space_mag.shape[1] == 1:
                    fg = pixel_space_mag  # (B, 1, D, H, W)
                    denom = fg.amax(dim=(-3, -2, -1), keepdim=True) + 1e-6
                    fg_norm = fg / denom
                    bg = 1.0 - fg_norm
                    pixel_space_mag = torch.cat([bg, fg_norm], dim=1)  # (B, 2, D, H, W)
                # Insert singleton V dimension at index 2 for consistency
                pixel_space_mag = pixel_space_mag.unsqueeze(2)  # -> (B, C_or_2, 1, D, H, W)
                transformed_variables.append(pixel_space_mag)
            else:
                # 2D case: use center slice along X, invert centered k-space: ifftshift -> ifft2
                xdim = var.shape[3]
                center = xdim // 2
                var2d = var[:, :, :, center:center+1, ...]  # (B, C, 2, 1, H, W)
                var2d = var2d.squeeze(3)  # -> (B, C, 2, H, W)

                real = var2d[:, :, 0, :, :].to(torch.float32)
                imag = var2d[:, :, 1, :, :].to(torch.float32)
                if real.is_cuda:
                    with torch.amp.autocast('cuda', enabled=False):
                        complex_var = torch.complex(real, imag)  # (B, C, H, W)
                        shifted_var = torch.fft.ifftshift(complex_var, dim=(-2, -1))
                        pixel_space_complex = torch.fft.ifft2(shifted_var, norm="ortho")
                        pixel_space_complex = torch.fft.fftshift(pixel_space_complex, dim=(-2, -1))
                else:
                    complex_var = torch.complex(real, imag)  # (B, C, H, W)
                    shifted_var = torch.fft.ifftshift(complex_var, dim=(-2, -1))
                    pixel_space_complex = torch.fft.ifft2(shifted_var, norm="ortho")
                    pixel_space_complex = torch.fft.fftshift(pixel_space_complex, dim=(-2, -1))

                pixel_space_mag = torch.abs(pixel_space_complex)  # (B, C, H, W)
                # If single-class (FG-only), synthesize background channel from normalized FG
                if not is_input_image and pixel_space_mag.shape[1] == 1:
                    fg = pixel_space_mag  # (B, 1, H, W)
                    denom = fg.amax(dim=(-2, -1), keepdim=True) + 1e-6
                    fg_norm = fg / denom
                    bg = 1.0 - fg_norm
                    pixel_space_mag = torch.cat([bg, fg_norm], dim=1)  # (B, 2, H, W)
                pixel_space_mag = pixel_space_mag.unsqueeze(2).unsqueeze(2)  # (B, C_or_2, 1, 1, H, W)
                transformed_variables.append(pixel_space_mag)
            
        return transformed_variables

    def _resolve_patient_id(self, patient_index: int) -> str:
        m = self.patient_index_to_id
        if isinstance(m, dict):
            return str(m.get(patient_index, patient_index))
        if isinstance(m, list) and 0 <= patient_index < len(m):
            return str(m[patient_index])
        return str(patient_index)

    @staticmethod
    def _new_vol_state() -> Dict[str, Any]:
        return {"pid": None, "sum": None, "count": None, "gt": None, "case_stats": {}}

    def _write_volume_prediction(self, patient_id: str, pred_s_hw: np.ndarray, gt_s_hw: np.ndarray) -> None:
        if self.predictions_dir is None:
            return
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = self.predictions_dir / 'pred'
        gt_dir = self.predictions_dir / 'gt'
        pred_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        pred_hw_s = np.transpose(pred_s_hw.astype(np.uint8, copy=False), (1, 2, 0))
        gt_hw_s = np.transpose(gt_s_hw.astype(np.uint8, copy=False), (1, 2, 0))
        nib.save(nib.Nifti1Image(pred_hw_s, affine=np.eye(4)), (pred_dir / f"{patient_id}.nii.gz").as_posix())
        nib.save(nib.Nifti1Image(gt_hw_s, affine=np.eye(4)), (gt_dir / f"{patient_id}.nii.gz").as_posix())

    def _start_volume(self, state: Dict[str, Any], pid: int, s_total: int, h: int, w: int) -> None:
        state["pid"] = pid
        state["sum"] = np.zeros((s_total, h, w), dtype=np.uint16)
        state["count"] = np.zeros((s_total,), dtype=np.uint16)
        state["gt"] = np.zeros((s_total, h, w), dtype=np.uint8)

    def _finalize_volume(self, state: Dict[str, Any], write_preds: bool = False) -> None:
        pid = state["pid"]
        sum_pred = state["sum"]
        count = state["count"]
        gt = state["gt"]
        if pid is None or sum_pred is None or count is None or gt is None:
            return

        count_b = count.astype(np.uint16).reshape(-1, 1, 1)
        pred = (sum_pred.astype(np.uint16) * 2 > count_b)
        g = gt.astype(bool)

        tp = int(np.logical_and(pred, g).sum())
        fp = int(np.logical_and(pred, ~g).sum())
        fn = int(np.logical_and(~pred, g).sum())
        nref = int(g.sum())

        state["case_stats"][int(pid)] = [tp, fp, fn, nref]
        if write_preds and self.predictions_dir is not None:
            patient_id = self._resolve_patient_id(int(pid))
            self._write_volume_prediction(patient_id, pred.astype(np.uint8), gt.astype(np.uint8))

        state["pid"] = None
        state["sum"] = None
        state["count"] = None
        state["gt"] = None

    def _accumulate_volume_from_batch(
        self, state: Dict[str, Any], pred_fg: np.ndarray, gt_fg: np.ndarray,
        meta: Dict[str, Any], write_preds: bool = False,
    ) -> None:
        pids = meta['patient_index'].detach().cpu().tolist()
        z_starts = meta['z_start'].detach().cpu().tolist()
        s_totals = meta['s_total'].detach().cpu().tolist()

        h = int(pred_fg.shape[-2])
        w = int(pred_fg.shape[-1])
        d_patch = int(pred_fg.shape[1])

        for i in range(int(pred_fg.shape[0])):
            pid = int(pids[i])
            if pid < 0:
                continue
            z0 = int(z_starts[i])
            s_total = int(s_totals[i])
            if s_total <= 0 or z0 < 0:
                continue

            cur_pid = state["pid"]
            if cur_pid is None:
                self._start_volume(state, pid, s_total, h, w)
            elif int(cur_pid) != pid:
                self._finalize_volume(state, write_preds=write_preds)
                self._start_volume(state, pid, s_total, h, w)

            z_end = min(z0 + d_patch, s_total)
            if z_end <= z0:
                continue
            d_valid = int(z_end - z0)

            state["sum"][z0:z_end] += pred_fg[i, :d_valid].astype(np.uint16, copy=False)
            state["count"][z0:z_end] += 1
            state["gt"][z0:z_end] = gt_fg[i, :d_valid]

    @staticmethod
    def _compute_vol_dice_mean(case_stats: Dict[int, List[int]]) -> float:
        case_dice = []
        for pid, (tp, fp, fn, nref) in case_stats.items():
            denom = 2 * tp + fp + fn
            if denom == 0:
                case_dice.append(float('nan'))
            else:
                case_dice.append((2.0 * tp) / denom)
        arr = np.array(case_dice, dtype=np.float32)
        return float(np.nanmean(arr)) if arr.size > 0 else 0.0

    def validation_step(self, batch: Dict[str, dict], batch_idx: int) -> None:
        """Infer batch on validation data, log metrics and retrieve loss.

        Args:
            batch: Batch containing the subjects.
            batch_idx: Number displaying index of this batch.

        Returns:
            Calculated loss.
        """
        y_hat, y = self.infer_batch(batch)
        x = batch['input']['data']

        # Calculate loss in float32 with AMP disabled to avoid fp16 NaNs
        criterion = self.criterion
        if y_hat.is_cuda:
            with torch.amp.autocast('cuda', enabled=False):
                if getattr(criterion, 'requires_label_mask', False):
                    loss = criterion(y_hat.float(), y.float(), label_mask=batch['label_mask']['data'])
                else:
                    loss = criterion(y_hat.float(), y.float())
        else:
            if getattr(criterion, 'requires_label_mask', False):
                loss = criterion(y_hat.float(), y.float(), label_mask=batch['label_mask']['data'])
            else:
                loss = criterion(y_hat.float(), y.float())

        # Convert predictions to pixel domain probabilities, compare against true pixel masks
        if self.output_image_logits:
            # Image-domain logits -> soft probabilities over classes
            yhat_pix = torch.softmax(y_hat.float(), dim=1)
        else:
            # k-space outputs -> pixel probabilities via fixed iFFT mapping
            # if self.input_domain == 'kspace':
            #     x = self.kspace_to_pixel_onehot([x], is_input_image=True)[0]
            yhat_pix = kspace_to_pixel_probs(y_hat)
        # Build one-hot GT from pixel label mask
        y_mask = batch['label_mask']['data']  # (B, H, W) or (B, D, H, W)
        y = self._mask_to_onehot_like(y_mask, yhat_pix, num_classes=2)
        # Convert predicted magnitudes to discrete one-hot via argmax
        y_hat = self.logits_to_mask(yhat_pix, y.shape[1])

        # Cache pixel-space probs so subclasses can reuse without a second forward pass.
        self._last_yhat_pix = yhat_pix

        # Prepare shapes for MONAI DiceMetric: (B, C, H, W[, D])
        # Drop only the singleton V dimension; keep spatial dims for 2D or 3D
        def _drop_v_dim(t: torch.Tensor) -> torch.Tensor:
            if t.shape[2] == 1:
                t = t.squeeze(2)
            return t
        y2 = _drop_v_dim(y)
        yhat2 = _drop_v_dim(y_hat)
        self.val_dice_metric(yhat2, y2)

        # Optionally keep recall/specificity logs (averaged within batch) for reference
        (
            avg_recall,
            per_class_recall,
            avg_specificity,
            per_class_specificity,
        ) = self.calculate_recall_specificity(y_hat, y, y.shape[1])

        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_recall', avg_recall, on_epoch=True, sync_dist=True)
        self.log('val_specificity', avg_specificity, on_epoch=True, sync_dist=True)

        # Volume-level (per-patient) overlap-vote accumulation for checkpoint selection.
        meta = batch.get('meta', {})
        if isinstance(meta, dict) and all(k in meta for k in ('patient_index', 'z_start', 's_total')):
            pred_fg = (torch.argmax(yhat2, dim=1) == 1).to(torch.uint8).detach().cpu().numpy()
            gt_fg = (torch.argmax(y2, dim=1) == 1).to(torch.uint8).detach().cpu().numpy()
            self._accumulate_volume_from_batch(self._val_vol_state, pred_fg, gt_fg, meta)

    def on_validation_epoch_start(self) -> None:
        self.val_dice_metric.reset()
        self._val_vol_state = self._new_vol_state()

    def on_validation_epoch_end(self) -> None:
        # Patch-based Dice (kept for backward compatibility / logging)
        dice_tensor = self.val_dice_metric.aggregate()
        per_class_mean = torch.nanmean(dice_tensor, dim=0)
        avg_dice = torch.nanmean(per_class_mean)
        self.log('val_avg_dice', avg_dice, sync_dist=True)
        for i, v in enumerate(per_class_mean.tolist(), start=1):
            self.log(f'val_dice_class_{i}', v, sync_dist=True)

        # Volume-level per-patient Dice (overlap-voted reconstruction)
        self._finalize_volume(self._val_vol_state)
        if self._val_vol_state["case_stats"]:
            vol_mean = self._compute_vol_dice_mean(self._val_vol_state["case_stats"])
            self.log('val_vol_dice_class_1', vol_mean, sync_dist=True)

    def test_step(self, batch: Dict[str, dict], batch_idx: int) -> None:
        """Infer batch on test data, log metrics and retrieve loss.

        Args:
            batch: Batch containing the subjects.
            batch_idx: Number displaying index of this batch.

        Returns:
            None.
        """
        y_hat, y = self.infer_batch(batch)
        x = batch['input']['data']

        # Calculate loss in float32 with AMP disabled to avoid fp16 NaNs
        criterion = self.criterion
        if y_hat.is_cuda:
            with torch.amp.autocast('cuda', enabled=False):
                if getattr(criterion, 'requires_label_mask', False):
                    loss = criterion(y_hat.float(), y.float(), label_mask=batch['label_mask']['data'])
                else:
                    loss = criterion(y_hat.float(), y.float())
        else:
            if getattr(criterion, 'requires_label_mask', False):
                loss = criterion(y_hat.float(), y.float(), label_mask=batch['label_mask']['data'])
            else:
                loss = criterion(y_hat.float(), y.float())

        # Convert predictions to pixel domain probabilities, compare against true pixel masks
        if self.input_domain == 'kspace':
            x = self.kspace_to_pixel_onehot([x], is_input_image=True)[0]
        if self.output_image_logits:
            # Image-domain logits -> soft probabilities over classes
            yhat_pix = torch.softmax(y_hat.float(), dim=1)
        else:
            # k-space outputs -> pixel probabilities via fixed iFFT mapping
            yhat_pix = kspace_to_pixel_probs(y_hat)
        y_mask = batch['label_mask']['data']
        y = self._mask_to_onehot_like(y_mask, yhat_pix, num_classes=2)
        y_hat = self.logits_to_mask(yhat_pix, y.shape[1])

        # Cache pixel-space probs so subclasses can reuse without a second forward pass.
        self._last_yhat_pix = yhat_pix

        def _drop_v_dim2(t: torch.Tensor) -> torch.Tensor:
            if t.shape[2] == 1:
                t = t.squeeze(2)
            return t
        y2 = _drop_v_dim2(y)
        yhat2 = _drop_v_dim2(y_hat)

        # Keep loss/aux metrics
        (
            avg_recall,
            per_class_recall,
            avg_specificity,
            per_class_specificity,
        ) = self.calculate_recall_specificity(y_hat, y, y.shape[1])

        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
        self.log('test_recall', avg_recall, on_epoch=True, sync_dist=True)
        self.log('test_specificity', avg_specificity, on_epoch=True, sync_dist=True)

        # Legacy Dice logging removed; use MONAI DiceMetric and exported masks

        if self.save_preds:
            output_dir = Path('/home/l721f/data/mama-mia/test_samples') / f'batch_{batch_idx}'
            self.tensors_to_nifti(x, y, y_hat, output_dir, label_mask=y_mask)

        # Volume-level (per-patient) reconstruction for reporting (binary overlap vote).
        meta = batch.get('meta', {})
        if isinstance(meta, dict) and all(k in meta for k in ('patient_index', 'z_start', 's_total')):
            pred_fg = (torch.argmax(yhat2, dim=1) == 1).to(torch.uint8).detach().cpu().numpy()
            gt_fg = (torch.argmax(y2, dim=1) == 1).to(torch.uint8).detach().cpu().numpy()
            self._accumulate_volume_from_batch(
                self._test_vol_state, pred_fg, gt_fg, meta, write_preds=True,
            )

    def on_test_epoch_start(self) -> None:
        self._test_vol_state = self._new_vol_state()

    def on_test_epoch_end(self) -> None:
        self._finalize_volume(self._test_vol_state, write_preds=True)
        if self._test_vol_state["case_stats"]:
            case_mean = self._compute_vol_dice_mean(self._test_vol_state["case_stats"])
            self.log('test_dice_class_1', case_mean, sync_dist=True)

    def logits_to_mask(self, logits: torch.Tensor, num_classes: int):
        """Converts logits to a one-hot encoded segmentation mask.

        Args:
            logits: Logits of shape (b, c, v, x, y, z). In this specific
                pipeline, these are actually Fourier magnitudes, not true logits.
            num_classes: Number of segmentation classes. Used for one-hot
                encoding.

        Returns:
            Segmentation mask of same shape as logits.
        """
        # The `logits` tensor here is actually the magnitude from the iFFT,
        # which is always non-negative. We should not apply softmax, as it is
        # intended for true logits (unnormalized log-probabilities).
        # We can directly find the argmax of the magnitudes to determine the
        # most likely class for each pixel.
        y_hat = torch.argmax(logits, dim=1)
        # Argmax converted probs to ordinal encoded so we need one hot encoding
        return rearrange(
            F.one_hot(y_hat, num_classes=num_classes),
            'b v x y z c -> b c v x y z',
        )

    def log_per_class_metrics(self, prefix: str, metrics: List):
        """Logs each class metric value separately.

        Args:
            prefix: Prefix for the name of the metric.
            metrics: List containing the values per class of the metric.


        Note:
            If metrics is a single scalar value, this function assumes a
            binary class segmentation was performed so there is no need to log
            the per class metrics.
        """
        try:
            metric_dict = {
                f"{prefix}_class_{i}": score.item()
                for i, score in enumerate(metrics)
            }
            self.log_dict(metric_dict, on_epoch=True, sync_dist=True)
        except (TypeError, AttributeError):
            pass

    def tensors_to_nifti(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        output_dir: Path,
        label_mask: torch.Tensor = None,
    ) -> None:
        """Save a PyTorch tensor as a NIfTI file.

        Args:
            x: Input tensor to save.
            y: Ground truth tensor to save.
            y_hat: Prediction tensor to save.
            output_dir: The directory to save the tensors.

        Note:
            Saves only the first sample of each batch.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        x = x.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()
        # Prefer true pixel masks if provided
        if label_mask is not None:
            y = label_mask.detach().cpu().numpy()
        else:
            y = y.detach().cpu().numpy()

        # Transform shapes
        # x: (b, C, v, x, y, z). For visualization we keep a single channel:
        #   - legacy case: C=1, v=1 -> (b, x, y, z)
        #   - multi-channel (e.g. multiple timepoints): take first along C and v.
        if x.ndim == 6:
            x = x[:, 0, 0, ...]
        x = x.astype('float')
        # y: if from one-hot -> argmax; if from label_mask already ordinal
        if y.ndim == 6:
            y = np.argmax(y, axis=1).squeeze(axis=1).astype('float')
        else:
            # label_mask shapes: (b, h, w) or (b, d, h, w)
            y = y.astype('float')
        y_hat = np.argmax(y_hat, axis=1).squeeze(axis=1).astype('float')

        for batch in range(x.shape[0]):
            # Squeeze the singleton dimension for 2D slices before saving
            img_x = x[batch].squeeze()
            img_y = y[batch].squeeze()
            img_y_hat = y_hat[batch].squeeze()

            nifti_img_x = nib.Nifti1Image(img_x, affine=np.eye(4))
            nib.save(nifti_img_x, output_dir / f'input_{batch}.nii.gz')
            nifti_img_y = nib.Nifti1Image(img_y, affine=np.eye(4))
            nib.save(nifti_img_y, output_dir / f'gt_{batch}.nii.gz')
            nifti_img_y_hat = nib.Nifti1Image(img_y_hat, affine=np.eye(4))
            nib.save(nifti_img_y_hat, output_dir / f'pred_{batch}.nii.gz')

    def calculate_recall_specificity(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        num_classes: int,
    ) -> Tuple[float]:
        """Calculates binary recall and specificity using persistent metrics."""
        pred = torch.flatten(torch.argmax(pred, dim=1))
        gt = torch.flatten(torch.argmax(gt, dim=1))

        avg_recall = self._recall_metric(pred, gt)
        avg_specificity = self._specificity_metric(pred, gt)
        return (
            avg_recall,
            avg_recall.unsqueeze(0),
            avg_specificity,
            avg_specificity.unsqueeze(0),
        )
