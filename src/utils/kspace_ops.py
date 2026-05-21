import torch
import torch.nn.functional as F


def kspace_to_pixel_probs(y_hat_k: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert k-space predictions to pixel-space soft class probabilities.

    Input y_hat_k shape: (B, C, V=2, X_or_D, H, W)
    - For 2D models, X_or_D == 1 (center slice along X).
    - For 3D stacks, X_or_D == D (depth); iFFT2 is applied per depth slice.

    Returns:
      - 2D: (B, C_or_2, 1, 1, H, W)
      - 3D: (B, C_or_2, 1, D, H, W)
    """
    b, c, v, xdim, h, w = y_hat_k.shape
    real = y_hat_k[:, :, 0, ...].to(torch.float32)
    imag = y_hat_k[:, :, 1, ...].to(torch.float32)

    if xdim == 1:
        # 2D case
        real2d = real.squeeze(2)  # (B, C, H, W)
        imag2d = imag.squeeze(2)  # (B, C, H, W)
        complex2d = torch.complex(real2d, imag2d)
        # Centered iFFT: ifftshift -> ifft2 -> fftshift
        shifted = torch.fft.ifftshift(complex2d, dim=(-2, -1))
        img_c = torch.fft.ifft2(shifted, norm="ortho")
        img_c = torch.fft.fftshift(img_c, dim=(-2, -1))
        mag = torch.abs(img_c)  # (B, C, H, W)
        # Build class probabilities
        if c == 1:
            fg = mag
            denom = fg.amax(dim=(-2, -1), keepdim=True) + eps
            fg_norm = fg / denom
            bg = 1.0 - fg_norm
            probs = torch.cat([bg, fg_norm], dim=1)  # (B, 2, H, W)
        else:
            denom = mag.sum(dim=1, keepdim=True) + eps
            probs = mag / denom  # (B, C, H, W)
        # Expand to (B, C, 1, 1, H, W)
        probs = probs.unsqueeze(2).unsqueeze(2)
        return probs
    else:
        # 3D stack (depth D)
        complex3d = torch.complex(real, imag)  # (B, C, D, H, W)
        # Centered iFFT: ifftshift -> ifft2 -> fftshift
        shifted = torch.fft.ifftshift(complex3d, dim=(-2, -1))
        img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
        img_c = torch.fft.fftshift(img_c, dim=(-2, -1))
        mag = torch.abs(img_c)  # (B, C, D, H, W)
        if c == 1:
            fg = mag
            denom = fg.amax(dim=(-3, -2, -1), keepdim=True) + eps
            fg_norm = fg / denom
            bg = 1.0 - fg_norm
            probs = torch.cat([bg, fg_norm], dim=1)  # (B, 2, D, H, W)
        else:
            denom = mag.sum(dim=1, keepdim=True) + eps
            probs = mag / denom  # (B, C, D, H, W)
        # Expand to (B, C, 1, D, H, W)
        probs = probs.unsqueeze(2)
        return probs


def mask_to_onehot_like(mask: torch.Tensor, like: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """Convert ordinal mask to one-hot, shaped like `like`'s spatial/depth dims.

    Inputs:
      - mask: (B, H, W) or (B, D, H, W)
      - like: (B, C, 1, 1, H, W) or (B, C, 1, D, H, W)
    Returns matching shape in channel dimension with inserted singleton dims:
      - 2D: (B, num_classes, 1, 1, H, W)
      - 3D: (B, num_classes, 1, D, H, W)
    """
    if mask.ndim == 3:
        oh = F.one_hot(mask.long(), num_classes=num_classes).permute(0, 3, 1, 2)  # (B, C, H, W)
        oh = oh.unsqueeze(2).unsqueeze(2)  # (B, C, 1, 1, H, W)
    else:
        oh = F.one_hot(mask.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        oh = oh.unsqueeze(2)  # (B, C, 1, D, H, W)
    return oh.to(dtype=like.dtype, device=like.device)


