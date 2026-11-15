"""Shared layers for the image U-Nets and hybrid encoder/decoder."""
import torch

import torch.nn as nn

import torch.nn.functional as F

from typing import Optional, Sequence, Tuple, Union

from monai.networks.nets import UNet

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
