import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, FocalLoss, DiceFocalLoss

from einops import reduce
from typing import Union, Tuple, List
from src.utils.kspace_ops import (
    kspace_to_pixel_probs,
    mask_to_onehot_like,
)


class HighFreqMSELoss(nn.Module):
    def __init__(
        self,
        reduction: str = 'mean',
        fourier_plane_enc: str = 'sagittal',
    ) -> None:
        """Initialization of the high frequency MSE loss.

        Args:
            reduction: Method to combine loss of each matrix entry.
                Defaults to 'mean'.
            fourier_plane_enc: Determines the plane along which the FFT is
                done. Defaults to 'sagittal'.
        """
        super(HighFreqMSELoss, self).__init__()
        self.reduction = reduction
        self.fourier_plane_enc = fourier_plane_enc

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Union[float, torch.Tensor]:
        """Calculates the high frequency MSE loss.

        Args:
            inputs: Predicted values.
            targets: Ground truth values.

        Returns:
            Combined loss or loss for each matrix entry.
        """
        if not (inputs.size() == targets.size()):
            print(
                f"""Using a prediction size ({inputs.size()}) that is different
                to the ground truth size ({targets.size()}). This will likely
                lead to incorrect results due to broadcasting."""
            )
        # Compute the squared error
        diff = torch.square(inputs - targets)

        # Generate frequency-dependent weights on the k-space plane (H, W)
        # We always weight across the last two spatial dims and broadcast
        h, w = diff.shape[-2], diff.shape[-1]
        freq_weights = self.get_freq_weights((h, w), diff.device.type)
        # Broadcast to (B, C, V, X/D, H, W)
        freq_weights = freq_weights.view(1, 1, 1, 1, h, w)

        # Multiply the differences by the weights
        if self.reduction == 'mean':
            return torch.mean(diff * freq_weights)

        return diff * freq_weights

    def get_freq_weights(self, shape: Tuple[int], device: str) -> torch.Tensor:
        """Creates a grid to weight the frequency components in k-space.

        Args:
            shape: Shape of the k-space which will be weighted.
            device: Device on which a tensor will be allocated
                ('cpu', 'cuda' or 'mps').

        Returns:
            Grid with high weights for high frequency components.
        """
        # Generate a grid
        cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2  # center coordinates
        y = torch.arange(0, shape[0]).to(device)
        x = torch.arange(0, shape[1]).to(device)
        y, x = torch.meshgrid(y, x, indexing='ij')

        # Compute the frequency magnitude from the center
        freq_magnitude = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Normalize the frequency magnitude to [0, 1]
        max_distance = (
            np.sqrt((shape[0] - 1) ** 2 + (shape[1] - 1) ** 2) / 2
        )  # Maximum possible distance
        freq_magnitude = freq_magnitude / max_distance

        # Compute the weights as the square of the frequency magnitude
        # This will give higher weights to high frequencies
        weights = freq_magnitude**2

        return weights


class NMSELoss(nn.Module):
    def __init__(self) -> None:
        """Initilization for normalize MSE loss"""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate the normalize MSE loss
        
        Args:
            inputs: Predicted values.
            targets: Ground truth values.

        Returns:
            Normalized MSE loss.
        """
        return torch.sum((inputs - targets) ** 2) / torch.sum(targets ** 2)


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_hat, y, reduction='mean')


class WeightedHighFreqMSELoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        reduction: str = 'mean',
        fourier_plane_enc: str = 'sagittal',
    ) -> None:
        """Weighted high-frequency MSE loss.

        This combines per-class weighting with spatial high-frequency
        emphasis using a simple radial weighting mask in the selected
        Fourier-encoded plane.

        Args:
            weight: Per-class weights of length equal to number of classes.
            reduction: Reduction method. Only 'mean' is supported.
            fourier_plane_enc: Plane used for radial weighting; one of
                'sagittal', 'coronal', or 'axial'.
        """
        super().__init__()
        weight_tensor = torch.as_tensor(weight, dtype=torch.float32)
        self.register_buffer('weight', weight_tensor)
        self.reduction = reduction
        self.fourier_plane_enc = fourier_plane_enc

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Union[float, torch.Tensor]:
        """Compute weighted HF-MSE.

        Args:
            y_hat: Prediction tensor of shape (b, c, ...).
            y: Ground truth tensor of shape (b, c, ...).

        Returns:
            Loss value.
        """
        if len(self.weight) != y_hat.shape[1]:
            raise ValueError('Length of weight list must match number of classes.')

        diff = torch.square(y_hat - y)

        # Build frequency mask on the k-space plane (H, W) and broadcast across other dims
        h, w = diff.shape[-2], diff.shape[-1]
        freq_weights = self._get_freq_weights((h, w), diff.device.type)
        # Broadcast to (B, C, V, X/D, H, W)
        freq_weights = freq_weights.view(1, 1, 1, 1, h, w)

        loss = 0.0
        for i, w_i in enumerate(self.weight):
            weighted_diff_class = w_i * diff[:, i, ...] * freq_weights
            loss += weighted_diff_class.sum()

        if self.reduction == 'mean':
            return loss / y_hat.numel()
        return loss

    def _get_freq_weights(self, shape: Tuple[int], device: str) -> torch.Tensor:
        """Create a radial weighting grid with higher weights at high frequency.

        Args:
            shape: 2D shape (H, W) for the Fourier plane.
            device: 'cpu', 'cuda' or 'mps'.

        Returns:
            Weight grid with values in [0, 1], emphasizing high frequencies.
        """
        cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
        y = torch.arange(0, shape[0]).to(device)
        x = torch.arange(0, shape[1]).to(device)
        y, x = torch.meshgrid(y, x, indexing='ij')

        freq_magnitude = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_distance = np.sqrt((shape[0] - 1) ** 2 + (shape[1] - 1) ** 2) / 2
        freq_magnitude = freq_magnitude / max_distance
        return freq_magnitude ** 2


class MAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(y_hat, y, reduction='mean')


class ImageSpaceDiceLoss(nn.Module):
    """Soft Dice loss computed in pixel space from k-space outputs.

    Expects network outputs in k-space of shape (B, C, V=2, X/D, H, W), where V indexes
    real/imag components. Converts predictions to pixel-space magnitudes via iFFT2 over (H, W),
    synthesizes background when only a single foreground class is predicted, and computes a
    differentiable Dice loss against the provided pixel-domain label mask.
    """

    # Flag used by the Lightning module to pass label_mask from the batch
    requires_label_mask = True

    def __init__(self, include_background: bool = False, eps: float = 1e-6) -> None:
        super().__init__()
        self.include_background = bool(include_background)
        self.eps = float(eps)

    

    def forward(
        self,
        y_hat_k: torch.Tensor,
        y_k: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if label_mask is None:
            raise ValueError('ImageSpaceDiceLoss requires `label_mask` in the batch.')

        probs = kspace_to_pixel_probs(y_hat_k, eps=self.eps)

        # Align GT to probabilities' spatial shape
        gt = mask_to_onehot_like(label_mask, probs, num_classes=2)

        # If shapes mismatch due to preprocessing, interpolate gt to match probs
        if gt.shape[2:] != probs.shape[2:]:
            # Deduce interpolation mode from dimensionality
            if gt.ndim == 4:
                gt = F.interpolate(gt, size=probs.shape[-2:], mode='nearest')
            else:
                gt = F.interpolate(gt, size=probs.shape[-3:], mode='nearest')

        # Optionally drop background channel
        if not self.include_background:
            if probs.shape[1] >= 2:
                probs = probs[:, 1:2, ...]
                gt = gt[:, 1:2, ...]

        # Soft Dice
        dims = tuple(range(2, probs.ndim))
        intersection = (probs * gt).sum(dim=dims)
        denom = (probs * probs).sum(dim=dims) + (gt * gt).sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        loss = 1.0 - dice.mean()
        return loss


class ImageSpaceDiceCELoss(nn.Module):
    """Compound Dice + CrossEntropy loss in image space.
    
    Combines ImageSpaceDiceLoss with a pixel-wise CrossEntropy (NLL) loss
    on the derived pixel-space probabilities. Optimizes for one iFFT call.
    """
    
    requires_label_mask = True

    def __init__(
        self, 
        include_background: bool = False, 
        ce_weight: float = 1.0, 
        dice_weight: float = 1.0, 
        class_weights: Union[Tuple[float, ...], List[float], None] = None,
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.include_background = bool(include_background)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(
        self,
        y_hat_k: torch.Tensor,
        y_k: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if label_mask is None:
            raise ValueError('ImageSpaceDiceCELoss requires `label_mask`.')

        # 1. Compute Pixel Probs (Shared for both losses)
        probs = kspace_to_pixel_probs(y_hat_k, eps=self.eps)
        
        # 2. Prepare GT
        gt = mask_to_onehot_like(label_mask, probs, num_classes=2)
        if gt.shape[2:] != probs.shape[2:]:
            if gt.ndim == 4:
                gt = F.interpolate(gt, size=probs.shape[-2:], mode='nearest')
            else:
                gt = F.interpolate(gt, size=probs.shape[-3:], mode='nearest')

        # 3. Compute Dice Loss
        probs_dice = probs
        gt_dice = gt
        if not self.include_background:
            if probs.shape[1] >= 2:
                probs_dice = probs[:, 1:2, ...]
                gt_dice = gt[:, 1:2, ...]

        dims = tuple(range(2, probs_dice.ndim))
        intersection = (probs_dice * gt_dice).sum(dim=dims)
        denom = (probs_dice * probs_dice).sum(dim=dims) + (gt_dice * gt_dice).sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        dice_loss = 1.0 - dice.mean()
        
        # 4. Compute CE Loss
        pred = probs
        # Squeeze singleton V and D dims to get (B, C, spatial...) for NLL
        if pred.shape[2] == 1:
            pred = pred.squeeze(2)
        if pred.shape[2] == 1 and label_mask.ndim == 3: # 2D case
            pred = pred.squeeze(2)
            
        target = label_mask.long()
        log_probs = torch.log(pred + self.eps)
        
        ce_loss = F.nll_loss(log_probs, target, weight=self.class_weights)
        
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class ImageSpaceDiceFocalLoss(nn.Module):
    """Compound Dice + Focal loss in image space.
    
    Combines ImageSpaceDiceLoss with a pixel-wise Focal loss
    on the derived pixel-space probabilities.
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    requires_label_mask = True

    def __init__(
        self, 
        include_background: bool = False, 
        focal_weight: float = 1.0, 
        dice_weight: float = 1.0, 
        gamma: float = 2.0,
        class_weights: Union[Tuple[float, ...], List[float], None] = None,
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.include_background = bool(include_background)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.gamma = float(gamma)
        self.eps = float(eps)
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(
        self,
        y_hat_k: torch.Tensor,
        y_k: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if label_mask is None:
            raise ValueError('ImageSpaceDiceFocalLoss requires `label_mask`.')

        # 1. Compute Pixel Probs (Shared for both losses)
        probs = kspace_to_pixel_probs(y_hat_k, eps=self.eps)
        
        # 2. Prepare GT
        gt = mask_to_onehot_like(label_mask, probs, num_classes=2)
        if gt.shape[2:] != probs.shape[2:]:
            if gt.ndim == 4:
                gt = F.interpolate(gt, size=probs.shape[-2:], mode='nearest')
            else:
                gt = F.interpolate(gt, size=probs.shape[-3:], mode='nearest')

        # 3. Compute Dice Loss
        probs_dice = probs
        gt_dice = gt
        if not self.include_background:
            if probs.shape[1] >= 2:
                probs_dice = probs[:, 1:2, ...]
                gt_dice = gt[:, 1:2, ...]

        dims = tuple(range(2, probs_dice.ndim))
        intersection = (probs_dice * gt_dice).sum(dim=dims)
        denom = (probs_dice * probs_dice).sum(dim=dims) + (gt_dice * gt_dice).sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        dice_loss = 1.0 - dice.mean()
        
        # 4. Compute Focal Loss
        # We compute manual Focal Loss using probabilities
        # FL = -alpha * (1-p_t)^gamma * log(p_t)
        
        # Reshape probs to (B, C, N) and gather p_t
        pred = probs
        if pred.shape[2] == 1:
            pred = pred.squeeze(2)
        if pred.shape[2] == 1 and label_mask.ndim == 3: # 2D case
            pred = pred.squeeze(2)
        
        target = label_mask.long()
        # Flatten for simpler indexing
        b, c = pred.shape[0], pred.shape[1]
        pred_flat = pred.view(b, c, -1)     # (B, C, N)
        target_flat = target.view(b, -1)    # (B, N)
        
        # Gather probabilities of ground truth classes
        # p_t: (B, N)
        p_t = pred_flat.gather(1, target_flat.unsqueeze(1)).squeeze(1)
        
        log_p_t = torch.log(p_t + self.eps)
        
        # Calculate focal term
        focal_term = (1.0 - p_t).pow(self.gamma)
        
        # Calculate alpha weights if provided
        if self.class_weights is not None:
            # Map target indices to weights
            # class_weights: (C,) -> (B, N) via gather? or manual indexing
            # weights[target]
            alpha = self.class_weights[target_flat]
            focal_loss = -alpha * focal_term * log_p_t
        else:
            focal_loss = -focal_term * log_p_t
            
        focal_loss = focal_loss.mean()
        
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


class ImageSpaceDiceCEKspaceMSELoss(nn.Module):
    """Pixel-space Dice+CE plus auxiliary k-space MSE.

    This encourages the network to match the known FFT(mask) target in k-space
    (from `target_kspace_stack.npy`) while still optimizing segmentation quality
    in image space via Dice+CE computed from iFFT magnitudes.
    """

    requires_label_mask = True

    def __init__(
        self,
        include_background: bool = False,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        kspace_mse_weight: float = 0.1,
        class_weights: Union[Tuple[float, ...], List[float], None] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.pixel_loss = ImageSpaceDiceCELoss(
            include_background=include_background,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            class_weights=class_weights,
            eps=eps,
        )
        self.kspace_mse_weight = float(kspace_mse_weight)

    def forward(
        self,
        y_hat_k: torch.Tensor,
        y_k: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        loss_pix = self.pixel_loss(y_hat_k, y_k, label_mask=label_mask)

        if self.kspace_mse_weight <= 0.0 or y_k is None:
            return loss_pix

        loss_k = F.mse_loss(y_hat_k.float(), y_k.float(), reduction="mean")
        return loss_pix + self.kspace_mse_weight * loss_k


class ImageSpaceDiceFocalKspaceMSELoss(nn.Module):
    """Pixel-space Dice+Focal plus auxiliary k-space MSE."""

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
        self.pixel_loss = ImageSpaceDiceFocalLoss(
            include_background=include_background,
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            gamma=gamma,
            class_weights=class_weights,
            eps=eps,
        )
        self.kspace_mse_weight = float(kspace_mse_weight)

    def forward(
        self,
        y_hat_k: torch.Tensor,
        y_k: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        loss_pix = self.pixel_loss(y_hat_k, y_k, label_mask=label_mask)

        if self.kspace_mse_weight <= 0.0 or y_k is None:
            return loss_pix

        loss_k = F.mse_loss(y_hat_k.float(), y_k.float(), reduction="mean")
        return loss_pix + self.kspace_mse_weight * loss_k


class ImageDiceLoss(nn.Module):
    """Soft Dice loss in image space for multi-class logits and ordinal masks.

    Expects network outputs in the image domain of shape (B, C, V, D_or_X, H, W),
    where C is the number of classes (≥2). Uses the ordinal `label_mask`
    supplied in the batch to build a matching one-hot target and computes
    a differentiable Dice loss.
    """

    # Flag used by the Lightning module to pass label_mask from the batch
    requires_label_mask = True

    def __init__(self, include_background: bool = False, eps: float = 1e-6) -> None:
        super().__init__()
        self.include_background = bool(include_background)
        self.eps = float(eps)

    def forward(
        self,
        y_hat_img: torch.Tensor,
        y_img: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if label_mask is None:
            raise ValueError('ImageDiceLoss requires `label_mask` in the batch.')

        if y_hat_img.dim() < 4 or y_hat_img.shape[1] < 2:
            raise ValueError('ImageDiceLoss expects logits with at least 2 channels.')

        # Treat y_hat_img as logits over classes
        probs = torch.softmax(y_hat_img, dim=1)

        # Build one-hot GT to match probabilities' spatial shape
        gt = mask_to_onehot_like(label_mask, probs, num_classes=2)

        # Optionally drop background channel
        if not self.include_background:
            probs = probs[:, 1:2, ...]
            gt = gt[:, 1:2, ...]

        # Soft Dice over all non-channel dimensions
        dims = tuple(range(2, probs.ndim))
        intersection = (probs * gt).sum(dim=dims)
        denom = (probs * probs).sum(dim=dims) + (gt * gt).sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        loss = 1.0 - dice.mean()
        return loss


class ImageDiceCELossMonai(nn.Module):
    """
    Dice + CE (MONAI) for image-space logits with ordinal masks.

    Expects logits shaped (B, C, V, D_or_H, H, W) or (B, C, V, H, W) with a
    singleton V. Uses label_mask (ordinal) and lets MONAI handle one-hot
    expansion. Keeps background to ensure negative-only patches produce
    gradients via CE.
    """

    requires_label_mask = True

    def __init__(
        self,
        include_background: bool = True,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.loss = DiceCELoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            smooth_nr=eps,
            smooth_dr=eps,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            reduction="mean",
        )

    def forward(
        self,
        y_hat_img: torch.Tensor,
        y_img: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if label_mask is None:
            raise ValueError('ImageDiceCELossMonai requires `label_mask` in the batch.')

        # Drop singleton V dim if present
        if y_hat_img.ndim == 6 and y_hat_img.shape[2] == 1:
            logits = y_hat_img.squeeze(2)
        elif y_hat_img.ndim == 5 and y_hat_img.shape[2] == 1:
            logits = y_hat_img.squeeze(2)
        else:
            logits = y_hat_img

        # Ensure target has channel dim for MONAI (B,1,...) shape
        if label_mask.ndim == 3:
            target = label_mask.unsqueeze(1)
        elif label_mask.ndim == 4:
            target = label_mask.unsqueeze(1)
        else:
            target = label_mask

        return self.loss(logits.float(), target.long())


class ImageDiceFocalLossMonai(nn.Module):
    """
    Dice + Focal (MONAI) for image-space logits with ordinal masks.

    Combines MONAI's DiceFocalLoss, which accepts logits.
    """

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
        self.loss = DiceFocalLoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            smooth_nr=eps,
            smooth_dr=eps,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
            gamma=gamma,
            weight=class_weights,
            reduction="mean",
        )

    def forward(
        self,
        y_hat_img: torch.Tensor,
        y_img: torch.Tensor = None,
        label_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if label_mask is None:
            raise ValueError('ImageDiceFocalLossMonai requires `label_mask` in the batch.')

        # Drop singleton V dim if present
        if y_hat_img.ndim == 6 and y_hat_img.shape[2] == 1:
            logits = y_hat_img.squeeze(2)
        elif y_hat_img.ndim == 5 and y_hat_img.shape[2] == 1:
            logits = y_hat_img.squeeze(2)
        else:
            logits = y_hat_img

        # Ensure target has channel dim for MONAI (B,1,...) shape
        if label_mask.ndim == 3:
            target = label_mask.unsqueeze(1)
        elif label_mask.ndim == 4:
            target = label_mask.unsqueeze(1)
        else:
            target = label_mask

        return self.loss(logits.float(), target.long())
