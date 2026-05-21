"""Full-dataset losses: Focal on ALL patches, Dice on positive patches only.

Focal loss provides per-voxel supervision on every patch (including negatives,
teaching the model "everything here is background").  Dice loss is applied only
to patches with at least one foreground voxel (avoids the degenerate 0/0 case).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List

from monai.losses import DiceLoss, FocalLoss

from src.utils.kspace_ops import kspace_to_pixel_probs, mask_to_onehot_like


class DiceFocalDetectLossMonai(nn.Module):
    """MONAI Focal (all patches) + Dice (positive patches only), image-space."""

    requires_label_mask = True

    def __init__(
        self,
        include_background: bool = True,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
        gamma: float = 2.0,
        class_weights: Union[Tuple[float, ...], List[float], None] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_dice = float(lambda_dice)
        self.lambda_focal = float(lambda_focal)

        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            smooth_nr=eps,
            smooth_dr=eps,
            reduction="mean",
        )
        self.focal_loss = FocalLoss(
            include_background=include_background,
            to_onehot_y=True,
            gamma=gamma,
            weight=class_weights,
            reduction="mean",
            use_softmax=True,
        )

    @staticmethod
    def _prep(
        y_hat_img: torch.Tensor, label_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Squeeze singleton V dim and add channel dim to target."""
        if y_hat_img.ndim == 6 and y_hat_img.shape[2] == 1:
            logits = y_hat_img.squeeze(2)
        elif y_hat_img.ndim == 5 and y_hat_img.shape[2] == 1:
            logits = y_hat_img.squeeze(2)
        else:
            logits = y_hat_img

        if label_mask.ndim in (3, 4):
            target = label_mask.unsqueeze(1)
        else:
            target = label_mask

        return logits.float(), target.long()

    def forward(
        self,
        y_hat_img: torch.Tensor,
        y_img: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if label_mask is None:
            raise ValueError("DiceFocalDetectLossMonai requires `label_mask`.")

        logits, target = self._prep(y_hat_img, label_mask)

        # Focal loss on ALL patches (gives negatives per-voxel supervision)
        loss = self.lambda_focal * self.focal_loss(logits, target)

        # Dice loss on positive patches only (avoids degenerate 0/0)
        pos_mask = label_mask.reshape(label_mask.shape[0], -1).sum(dim=1) > 0.0
        if pos_mask.any():
            loss = loss + self.lambda_dice * self.dice_loss(
                logits[pos_mask], target[pos_mask],
            )

        return loss


class DiceFocalDetectKspaceMSELoss(nn.Module):
    """Focal (all) + Dice (positives) in pixel space from k-space + k-space MSE on positives."""

    requires_label_mask = True

    def __init__(
        self,
        include_background: bool = False,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        gamma: float = 2.0,
        kspace_mse_weight: float = 0.1,
        class_weights: Union[Tuple[float, ...], List[float], None] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.kspace_mse_weight = float(kspace_mse_weight)
        self.include_background = bool(include_background)
        self.gamma = float(gamma)
        self.eps = float(eps)

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def _focal_from_probs(
        self, probs: torch.Tensor, label_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pixel-wise Focal loss from pre-computed softmax probabilities."""
        pred = probs
        if pred.shape[2] == 1:
            pred = pred.squeeze(2)
        if pred.ndim >= 4 and pred.shape[2] == 1 and label_mask.ndim == 3:
            pred = pred.squeeze(2)

        target = label_mask.long()
        b, c = pred.shape[0], pred.shape[1]
        pred_flat = pred.reshape(b, c, -1)
        target_flat = target.reshape(b, -1)

        p_t = pred_flat.gather(1, target_flat.unsqueeze(1)).squeeze(1)
        log_p_t = torch.log(p_t + self.eps)
        focal_term = (1.0 - p_t).pow(self.gamma)

        if self.class_weights is not None:
            alpha = self.class_weights[target_flat]
            return (-alpha * focal_term * log_p_t).mean()
        return (-focal_term * log_p_t).mean()

    def _dice_from_probs(
        self, probs: torch.Tensor, gt: torch.Tensor,
    ) -> torch.Tensor:
        """Soft Dice loss from pre-computed probabilities and one-hot GT."""
        p = probs
        g = gt
        if not self.include_background and p.shape[1] >= 2:
            p = p[:, 1:2, ...]
            g = g[:, 1:2, ...]
        dims = tuple(range(2, p.ndim))
        intersection = (p * g).sum(dim=dims)
        denom = (p * p).sum(dim=dims) + (g * g).sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()

    def forward(
        self,
        y_hat_k: torch.Tensor,
        y_k: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if label_mask is None:
            raise ValueError("DiceFocalDetectKspaceMSELoss requires `label_mask`.")

        probs = kspace_to_pixel_probs(y_hat_k, eps=self.eps)
        gt = mask_to_onehot_like(label_mask, probs, num_classes=2)
        if gt.shape[2:] != probs.shape[2:]:
            gt = F.interpolate(
                gt,
                size=probs.shape[-3:] if gt.ndim > 4 else probs.shape[-2:],
                mode="nearest",
            )

        pos_mask = label_mask.reshape(label_mask.shape[0], -1).sum(dim=1) > 0.0

        loss = self.focal_weight * self._focal_from_probs(probs, label_mask)

        if pos_mask.any():
            loss = loss + self.dice_weight * self._dice_from_probs(
                probs[pos_mask], gt[pos_mask],
            )

        if self.kspace_mse_weight > 0 and y_k is not None and pos_mask.any():
            loss = loss + self.kspace_mse_weight * F.mse_loss(
                y_hat_k[pos_mask].float(), y_k[pos_mask].float(), reduction="mean",
            )

        return loss
