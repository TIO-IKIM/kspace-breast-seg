import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Sequence, Tuple, Union
from monai.networks.nets import UNet

class DiceScore(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        """Initialization of the dice score module.

        Args:
            smooth: Factor to ensure differentiability. Defaults to 1.0.
        """
        super().__init__()
        self.smooth = smooth

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """Calculates the dice score.

        Args:
            y_pred: Predicted values.
            y_true: Ground truth values.

        Returns:
            Average Dice score, per class Dice score.
        """
        assert (
            y_pred.size() == y_true.size()
        ), f"y_pred.size(): {y_pred.size()}, y_true.size(): {y_true.size()}"

        # Calculate intersection and union
        intersection = torch.sum(y_pred * y_true, dim=(0, 2, 3, 4, 5))
        union = torch.sum(y_pred + y_true, dim=(0, 2, 3, 4, 5))

        # Calculate Dice score for each class
        dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return average Dice score and per class Dice score
        return dice_scores.mean(), dice_scores

class UNet3D(nn.Module):
    """3D U-Net backbone using MONAI's `UNet`.

    Expects input shaped (B, C_in=1, V=2, D, H, W) and outputs (B, C_out=2, V=2, D, H, W).
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 32,
        depth: int = 4,
        strides: Optional[Sequence[Union[int, Sequence[int]]]] = None,
        kernel_size: Union[int, Sequence[int]] = 3,
        up_kernel_size: Union[int, Sequence[int]] = 3,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Treat the split real/imag (V) as additional input channels
        # Input: (C_in, V, D, H, W) -> in_channels = C_in * V
        in_channels = input_shape[0] * input_shape[1]
        # Output channels flatten classes and vector dimension (real/imag)
        out_channels = output_shape[0] * output_shape[1]

        depth = int(depth)
        channels = tuple(int(hidden_factor * (2**i)) for i in range(depth + 1))
        if strides is None:
            strides = tuple((2, 2, 2) for _ in range(depth))

        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=("group", {"num_groups": 2}),
            bias=True,
            dropout=0.0,
            num_res_units=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, V=2, D, H, W) → (B, C_in*V, D, H, W)
        b, c, v, d, h, w = x.shape
        x5 = x.view(b, c * v, d, h, w)
        out5 = self.net(x5)  # (B, C_out*V, D, H, W)
        # Reshape back to 6D (B, C_out, V, D, H, W)
        out6 = out5.view(
            b,
            self.output_shape[0],
            self.output_shape[1],
            d,
            h,
            w,
        )
        return out6


class KspaceDilatedResNet(nn.Module):
    """
    K-space backbone using dilated residual blocks.
    Avoids downsampling (aliasing) and upsampling (checkerboard/ghosting).
    Uses alternating dilations to capture context on the frequency grid.
    """
    def __init__(
        self, 
        input_shape: Tuple[int], 
        output_shape: Tuple[int], 
        hidden_dim: int = 32, 
        num_blocks: int = 6, 
        kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # Flatten V dimension: (B, C, V, ...) -> (B, C*V, ...)
        c_in = input_shape[0] * 2
        c_out = output_shape[0] * 2
        
        self.head = nn.Conv3d(c_in, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        
        self.blocks = nn.ModuleList()
        # Alternating dilations: 1, 2, 4, 1, 2, 4...
        dilations = [1, 2, 4] 
        
        for i in range(num_blocks):
            d = dilations[i % len(dilations)]
            self.blocks.append(self._make_block(hidden_dim, kernel_size, d))
             
        self.tail = nn.Conv3d(hidden_dim, c_out, kernel_size=kernel_size, padding=kernel_size//2)

    def _make_block(self, dim, k, d):
        # ResBlock: Conv(d) - Norm - Act - Conv(1) - Norm - Act
        padding = d * (k // 2)
        return nn.Sequential(
            nn.Conv3d(dim, dim, k, padding=padding, dilation=d, bias=True),
            nn.GroupNorm(2, dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(dim, dim, k, padding=k//2, dilation=1, bias=True),
            nn.GroupNorm(2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, V=2, D, H, W)
        b, c, v, d, h, w = x.shape
        x = x.view(b, c*v, d, h, w)
        
        x = self.head(x)
        
        # Residual blocks
        for blk in self.blocks:
            res = x
            out = blk(x)
            x = F.leaky_relu(res + out, negative_slope=0.1, inplace=True)
            
        x = self.tail(x)
        
        # Reshape: (B, C_out, V=2, D, H, W)
        return x.view(b, self.output_shape[0], 2, d, h, w)


class UNet3D_K2Img(nn.Module):
    """Two-stage k-space -> image segmentation model.

    Pipeline (3D patches):
      1) UNet in k-space (real/imag as V=2) to produce learned complex features
      2) Fixed centered iFFT2 over (H, W) on those features
      3) UNet in image space (still complex features as V=2) to produce image logits (V=1)

    This keeps the *input* as full complex k-space, but avoids forcing the
    network to synthesize FFT(mask) as an explicit output representation.
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 24,
        depth: int = 4,
        mid_channels: int = 8,
        strides: Optional[Sequence[Union[int, Sequence[int]]]] = None,
        strides_k: Optional[Sequence[Union[int, Sequence[int]]]] = None,
        strides_img: Optional[Sequence[Union[int, Sequence[int]]]] = None,
        kernel_size: Union[int, Sequence[int]] = 3,
        up_kernel_size: Union[int, Sequence[int]] = 3,
        kernel_size_k: Optional[Union[int, Sequence[int]]] = None,
        up_kernel_size_k: Optional[Union[int, Sequence[int]]] = None,
        kernel_size_img: Optional[Union[int, Sequence[int]]] = None,
        up_kernel_size_img: Optional[Union[int, Sequence[int]]] = None,
        hidden_factor_k: Optional[int] = None,
        hidden_factor_img: Optional[int] = None,
        depth_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.mid_channels = int(mid_channels)

        hf_k = hidden_factor_k if hidden_factor_k is not None else hidden_factor
        hf_img = hidden_factor_img if hidden_factor_img is not None else hidden_factor
        d_k = depth_k if depth_k is not None else depth

        d, h, w = int(input_shape[2]), int(input_shape[3]), int(input_shape[4])
        kfeat_shape = (self.mid_channels, 2, d, h, w)

        self.k_unet = UNet3D(
            input_shape=input_shape,
            output_shape=kfeat_shape,
            hidden_factor=hf_k,
            depth=d_k,
            strides=strides_k if strides_k is not None else strides,
            kernel_size=kernel_size_k if kernel_size_k is not None else kernel_size,
            up_kernel_size=up_kernel_size_k if up_kernel_size_k is not None else up_kernel_size,
        )
        self.img_unet = UNet3D(
            input_shape=kfeat_shape,
            output_shape=output_shape,
            hidden_factor=hf_img,
            depth=depth,
            strides=strides_img if strides_img is not None else strides,
            kernel_size=kernel_size_img if kernel_size_img is not None else kernel_size,
            up_kernel_size=up_kernel_size_img if up_kernel_size_img is not None else up_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: k-space UNet -> complex features in k-space (B, Cmid, V=2, D, H, W)
        kfeat = self.k_unet(x)

        # Stage 2: centered iFFT2 over (H, W) for each (B, Cmid, D) slice
        real = kfeat[:, :, 0, ...].to(torch.float32)
        imag = kfeat[:, :, 1, ...].to(torch.float32)

        if real.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                complex_vol = torch.complex(real, imag)  # (B, Cmid, D, H, W)
                shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
                img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
                img_c = torch.fft.fftshift(img_c, dim=(-2, -1))
        else:
            complex_vol = torch.complex(real, imag)
            shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
            img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
            img_c = torch.fft.fftshift(img_c, dim=(-2, -1))

        imgfeat = torch.stack([img_c.real, img_c.imag], dim=2)  # (B, Cmid, V=2, D, H, W)

        # Stage 3: image UNet -> logits (typically (B, 2, V=1, D, H, W))
        return self.img_unet(imgfeat)


class UNet3D_K2Img_Dilated(nn.Module):
    """
    Same as UNet3D_K2Img, but replaces the k-space UNet with a Dilated ResNet.
    This avoids downsampling/upsampling artifacts in the k-space domain.
    """
    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 24,
        depth: int = 4,
        mid_channels: int = 8,
        strides: Optional[Sequence[Union[int, Sequence[int]]]] = None,
        kernel_size: Union[int, Sequence[int]] = 3,
        up_kernel_size: Union[int, Sequence[int]] = 3,
        num_res_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.mid_channels = int(mid_channels)

        d, h, w = int(input_shape[2]), int(input_shape[3]), int(input_shape[4])
        kfeat_shape = (self.mid_channels, 2, d, h, w)

        # Stage 1: Dilated ResNet in k-space (NO DOWN/UP)
        self.k_net = KspaceDilatedResNet(
            input_shape=input_shape,
            output_shape=kfeat_shape,
            hidden_dim=int(hidden_factor), 
            num_blocks=num_res_blocks,
            kernel_size=3
        )

        # Stage 3: Standard Image-Space UNet
        self.img_unet = UNet3D(
            input_shape=kfeat_shape,
            output_shape=output_shape,
            hidden_factor=hidden_factor,
            depth=depth,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: k-space Dilated ResNet -> complex features
        kfeat = self.k_net(x)

        # Stage 2: centered iFFT2
        real = kfeat[:, :, 0, ...].to(torch.float32)
        imag = kfeat[:, :, 1, ...].to(torch.float32)

        if real.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                complex_vol = torch.complex(real, imag)
                shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
                img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
                img_c = torch.fft.fftshift(img_c, dim=(-2, -1))
        else:
            complex_vol = torch.complex(real, imag)
            shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
            img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
            img_c = torch.fft.fftshift(img_c, dim=(-2, -1))

        imgfeat = torch.stack([img_c.real, img_c.imag], dim=2)

        # Stage 3: Image UNet
        return self.img_unet(imgfeat)


class UNet3D_K2IfftLogMagHead(nn.Module):
    """k-space UNet -> iFFT2 -> log-magnitude -> tiny calibration head.

    Notes:
    - In the baseline pipeline, pixel probs are `mag / sum(mag)`.
      This is exactly `softmax(log(mag))`.
    - This model uses `log(|iFFT|)` as stable features and learns a small 1x1
      calibration to make the logit threshold/temperature easier.
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 24,
        depth: int = 4,
        strides: Optional[Sequence[Union[int, Sequence[int]]]] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.eps = float(eps)

        c_out = int(output_shape[0])
        d, h, w = int(output_shape[2]), int(output_shape[3]), int(output_shape[4])
        k_out_shape = (c_out, 2, d, h, w)

        self.k_unet = UNet3D(
            input_shape=input_shape,
            output_shape=k_out_shape,
            hidden_factor=hidden_factor,
            depth=depth,
            strides=strides,
        )

        # Very small per-class calibration on log-magnitude features.
        # Initialize as identity so training starts from the baseline mapping:
        # softmax(log(|iFFT|)) == |iFFT| / sum(|iFFT|).
        self.head = nn.Conv3d(
            in_channels=c_out,
            out_channels=c_out,
            kernel_size=1,
            groups=c_out,
            bias=True,
        )
        with torch.no_grad():
            self.head.weight.fill_(1.0)
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kpred = self.k_unet(x)  # (B, C, V=2, D, H, W)

        real = kpred[:, :, 0, ...].to(torch.float32)
        imag = kpred[:, :, 1, ...].to(torch.float32)

        if real.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                complex_vol = torch.complex(real, imag)  # (B, C, D, H, W)
                shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
                img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
                img_c = torch.fft.fftshift(img_c, dim=(-2, -1))
        else:
            complex_vol = torch.complex(real, imag)
            shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
            img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
            img_c = torch.fft.fftshift(img_c, dim=(-2, -1))

        mag = torch.abs(img_c)  # (B, C, D, H, W)
        logmag = torch.log(mag + self.eps)
        logits = self.head(logmag)  # (B, C, D, H, W)
        return logits.unsqueeze(2)  # (B, C, V=1, D, H, W)


class UNet3D_K2FeatIfftTinyHead(nn.Module):
    """k-space UNet -> iFFT2 -> tiny image head -> logits.

    Unlike UNet3D (which predicts per-class k-space), this predicts a small set
    of complex k-space features, converts to image features via iFFT2, then uses
    a tiny image-space head to produce logits.
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 16,
        depth: int = 3,
        mid_channels: int = 16,
        head_channels: int = 16,
        strides: Optional[Sequence[Union[int, Sequence[int]]]] = None,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.mid_channels = int(mid_channels)
        head_channels = int(head_channels)

        c_out = int(output_shape[0])
        d, h, w = int(output_shape[2]), int(output_shape[3]), int(output_shape[4])
        kfeat_shape = (self.mid_channels, 2, d, h, w)

        self.k_unet = UNet3D(
            input_shape=input_shape,
            output_shape=kfeat_shape,
            hidden_factor=hidden_factor,
            depth=depth,
            strides=strides,
        )

        in_ch = self.mid_channels * 2
        self.head = nn.Sequential(
            nn.Conv3d(in_ch, head_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(4, head_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(head_channels, c_out, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kfeat = self.k_unet(x)  # (B, Cmid, V=2, D, H, W)

        real = kfeat[:, :, 0, ...].to(torch.float32)
        imag = kfeat[:, :, 1, ...].to(torch.float32)

        if real.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                complex_vol = torch.complex(real, imag)  # (B, Cmid, D, H, W)
                shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
                img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
                img_c = torch.fft.fftshift(img_c, dim=(-2, -1))
        else:
            complex_vol = torch.complex(real, imag)
            shifted = torch.fft.ifftshift(complex_vol, dim=(-2, -1))
            img_c = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
            img_c = torch.fft.fftshift(img_c, dim=(-2, -1))

        img_ri = torch.stack([img_c.real, img_c.imag], dim=2)  # (B, Cmid, 2, D, H, W)
        b, c, v, d, h, w = img_ri.shape
        feat = img_ri.view(b, c * v, d, h, w)  # (B, Cmid*2, D, H, W)

        logits = self.head(feat)  # (B, C_out, D, H, W)
        return logits.unsqueeze(2)  # (B, C_out, 1, D, H, W)


class UNet3D_K2IfftHermitianLogits(nn.Module):
    """k-space UNet -> Hermitian projection -> iFFT2 -> real logits (V=1)."""

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 24,
        depth: int = 4,
        strides: Optional[Sequence[Union[int, Sequence[int]]]] = None,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        c_out = int(output_shape[0])
        d, h, w = int(output_shape[2]), int(output_shape[3]), int(output_shape[4])
        k_out_shape = (c_out, 2, d, h, w)

        self.k_unet = UNet3D(
            input_shape=input_shape,
            output_shape=k_out_shape,
            hidden_factor=hidden_factor,
            depth=depth,
            strides=strides,
        )

        # Per-class logit calibration (init as identity).
        self.head = nn.Conv3d(
            in_channels=c_out,
            out_channels=c_out,
            kernel_size=1,
            groups=c_out,
            bias=True,
        )
        with torch.no_grad():
            self.head.weight.fill_(1.0)
            self.head.bias.zero_()

    @staticmethod
    def _hermitian_symmetrize_2d(k_uncentered: torch.Tensor) -> torch.Tensor:
        # k_uncentered: (..., H, W) complex with DC at [0,0]
        k_neg = torch.roll(
            torch.flip(k_uncentered, dims=(-2, -1)),
            shifts=(1, 1),
            dims=(-2, -1),
        )
        return 0.5 * (k_uncentered + torch.conj(k_neg))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict per-class complex k-space (centered): (B, C_out, V=2, D, H, W)
        kpred = self.k_unet(x)
        real = kpred[:, :, 0, ...].to(torch.float32)
        imag = kpred[:, :, 1, ...].to(torch.float32)

        if real.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                k_c = torch.complex(real, imag)  # centered k-space
                k_unc = torch.fft.ifftshift(k_c, dim=(-2, -1))
                k_unc = self._hermitian_symmetrize_2d(k_unc)
                img_c = torch.fft.ifft2(k_unc, dim=(-2, -1), norm="ortho")
                img_c = torch.fft.fftshift(img_c, dim=(-2, -1))
        else:
            k_c = torch.complex(real, imag)
            k_unc = torch.fft.ifftshift(k_c, dim=(-2, -1))
            k_unc = self._hermitian_symmetrize_2d(k_unc)
            img_c = torch.fft.ifft2(k_unc, dim=(-2, -1), norm="ortho")
            img_c = torch.fft.fftshift(img_c, dim=(-2, -1))

        logits = self.head(img_c.real)  # (B, C_out, D, H, W)
        return logits.unsqueeze(2)  # (B, C_out, 1, D, H, W)


def _ifft2_centered(k: torch.Tensor) -> torch.Tensor:
    """Centered iFFT2 over last two dims. Input/output: (..., H, W) complex."""
    shifted = torch.fft.ifftshift(k, dim=(-2, -1))
    img = torch.fft.ifft2(shifted, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(img, dim=(-2, -1))


class UNet3D_KspaceAttn(nn.Module):
    """Architecture 3: K-space attention on iFFT baseline.

    Computes a spatial attention map from k-space features (via 1x1 convs) and
    uses it to modulate image-space UNet features. The 1x1 convs in k-space
    respect the global nature of frequency domain (each k-space point influences
    the whole image).

    Pipeline:
      1) K-space attention: 1x1 convs -> iFFT -> sigmoid attention map
      2) Image path: iFFT(input) -> UNet encoder -> features
      3) Modulate encoder features with attention
      4) UNet decoder -> logits
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 24,
        depth: int = 4,
        attn_channels: int = 16,
        strides: Optional[Sequence[Union[int, Sequence[int]]]] = None,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        c_out = int(output_shape[0])
        d, h, w = int(output_shape[2]), int(output_shape[3]), int(output_shape[4])

        # K-space attention path: 1x1 convs (global in frequency domain)
        self.kspace_proj = nn.Sequential(
            nn.Conv3d(2, attn_channels, kernel_size=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(attn_channels, attn_channels, kernel_size=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.attn_head = nn.Conv3d(attn_channels, 1, kernel_size=1, bias=True)

        # Image-space UNet for segmentation
        img_input_shape = (1, 2, d, h, w)  # complex image as real/imag
        self.image_unet = UNet3D(
            input_shape=img_input_shape,
            output_shape=output_shape,
            hidden_factor=hidden_factor,
            depth=depth,
            strides=strides,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 2, D, H, W) k-space
        b, c_in, v, d, h, w = x.shape
        k = x[:, 0]  # (B, 2, D, H, W) real/imag

        # K-space attention branch (1x1 convs respect global frequency structure)
        k_feat = self.kspace_proj(k)  # (B, attn_ch, D, H, W)
        k_attn_kspace = self.attn_head(k_feat)  # (B, 1, D, H, W)

        # Transform attention to image space
        real_attn = k_attn_kspace.to(torch.float32)
        if real_attn.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                # Treat as real-valued k-space, iFFT gives complex with small imag
                complex_attn = torch.complex(real_attn, torch.zeros_like(real_attn))
                attn_img = _ifft2_centered(complex_attn)
        else:
            complex_attn = torch.complex(real_attn, torch.zeros_like(real_attn))
            attn_img = _ifft2_centered(complex_attn)
        attn_map = torch.sigmoid(attn_img.real)  # (B, 1, D, H, W)

        # Image path: iFFT of input k-space
        real = k[:, 0:1].to(torch.float32)
        imag = k[:, 1:2].to(torch.float32)
        if real.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                kcomplex = torch.complex(real, imag)
                img_c = _ifft2_centered(kcomplex)
        else:
            kcomplex = torch.complex(real, imag)
            img_c = _ifft2_centered(kcomplex)
        img_ri = torch.cat([img_c.real, img_c.imag], dim=1)  # (B, 2, D, H, W)

        # Modulate with attention before UNet
        img_modulated = img_ri * attn_map  # (B, 2, D, H, W)

        # Reshape for UNet: (B, 1, 2, D, H, W)
        img_in = img_modulated.unsqueeze(1)
        logits = self.image_unet(img_in)  # (B, C_out, V=1, D, H, W)
        return logits


class UNet3D_DualPath(nn.Module):
    """Architecture 2: Spectral-Spatial Dual Path (lightweight).

    Two parallel paths:
      Path A (Spectral): k-space -> 1x1 convs -> global pool -> conditioning vector
      Path B (Spatial): iFFT(k-space) -> image UNet (with FiLM conditioning)

    The spectral path extracts global frequency statistics that condition
    the image-space segmentation via FiLM (feature-wise linear modulation).
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: Tuple[int],
        hidden_factor: int = 24,
        depth: int = 4,
        spectral_channels: int = 16,
        strides: Optional[Sequence[Union[int, Sequence[int]]]] = None,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        c_out = int(output_shape[0])
        d, h, w = int(output_shape[2]), int(output_shape[3]), int(output_shape[4])

        # Path A: Spectral encoder (1x1 convs in k-space -> global pool)
        self.spectral_encoder = nn.Sequential(
            nn.Conv3d(2, spectral_channels, kernel_size=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(spectral_channels, spectral_channels, kernel_size=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool3d(1),  # (B, spectral_ch, 1, 1, 1)
        )

        # FiLM: spectral features -> scale/shift for image features
        # Applied after first conv of image UNet
        img_first_ch = hidden_factor
        self.film_gamma = nn.Linear(spectral_channels, img_first_ch)
        self.film_beta = nn.Linear(spectral_channels, img_first_ch)

        # Path B: Standard image-space UNet
        img_input_shape = (1, 2, d, h, w)  # complex image as real/imag
        self.image_unet = UNet3D(
            input_shape=img_input_shape,
            output_shape=output_shape,
            hidden_factor=hidden_factor,
            depth=depth,
            strides=strides,
        )

        # Hook to get first encoder output for FiLM modulation
        self._first_enc_feat = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 2, D, H, W) k-space
        b, c_in, v, d, h, w = x.shape
        k = x[:, 0]  # (B, 2, D, H, W) real/imag

        # Path A: Spectral features (1x1 convs + global pool)
        spectral_vec = self.spectral_encoder(k).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, spectral_ch)

        # FiLM parameters
        gamma = self.film_gamma(spectral_vec)  # (B, img_first_ch)
        beta = self.film_beta(spectral_vec)    # (B, img_first_ch)

        # Path B: iFFT -> image
        real = k[:, 0:1].to(torch.float32)
        imag = k[:, 1:2].to(torch.float32)
        if real.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                kcomplex = torch.complex(real, imag)
                img_c = _ifft2_centered(kcomplex)
        else:
            kcomplex = torch.complex(real, imag)
            img_c = _ifft2_centered(kcomplex)

        img_ri = torch.cat([img_c.real, img_c.imag], dim=1)  # (B, 2, D, H, W)

        # Run through UNet with FiLM modulation on first layer
        # Access the inner MONAI UNet
        inner_net = self.image_unet.net

        # Manual forward with FiLM after first conv block
        # MONAI UNet structure: model.0 is first down block
        img_in = img_ri  # (B, 2, D, H, W)

        # First down block
        x_enc = inner_net.model[0](img_in)  # (B, hidden_factor, D, H, W)

        # Apply FiLM modulation: x = gamma * x + beta
        gamma_bc = gamma.view(b, -1, 1, 1, 1)
        beta_bc = beta.view(b, -1, 1, 1, 1)
        x_enc = gamma_bc * x_enc + beta_bc

        # Continue through rest of network
        for i in range(1, len(inner_net.model)):
            x_enc = inner_net.model[i](x_enc)

        # Reshape output
        out = x_enc.view(
            b,
            self.output_shape[0],
            self.output_shape[1],
            d, h, w,
        )
        return out
