"""Native radial Fourier operators used by preprocessing and all new models.

The forward operator and ``adj_unweighted`` form a Hermitian pair. ``adj``
additionally applies density weights, outside the NUFFT autograd function.
Their scalar normalization gives the underlying discrete Fourier transform a
unit central-impulse response in ``adj(fwd(delta))`` at every spoke budget.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal FFT of the last two dimensions."""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal inverse FFT of the last two dimensions."""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def sense_combine(
    coil_images: torch.Tensor, maps: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Combine ``[..., coil, height, width]``; zero maps give zero output."""
    denominator = maps.abs().square().sum(dim=-3).clamp_min(eps)
    return (maps.conj() * coil_images).sum(dim=-3) / denominator


def _numpy_float32(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32, order="C", copy=True)


class RadialOperator(nn.Module):
    """Lazy cuFINUFFT on a fixed trajectory, without sensitivity maps.

    ``trajectory_cycles`` is ``(N, 2)`` in cycles/pixel, within [-0.5, 0.5].
    Images have shape ``(B, C, H, W)`` and samples ``(B, C, N)``; all methods
    return complex64. Construction works on CPU; computation requires CUDA.

    MRI-NUFFT divides both transforms by ``sqrt(4 * H * W)``. Consequently
    ``A^H D A`` has central-impulse gain ``sum(D)/(4*H*W)``. We absorb its
    reciprocal into ``dcf``. This normalization depends only on geometry, not
    measured signal or withheld samples. NUFFT approximation error remains.
    """

    def __init__(
        self, trajectory_cycles, dcf, im_size=(320, 320), *, eps: float = 1e-5
    ):
        super().__init__()
        trajectory = _numpy_float32(trajectory_cycles)
        density = _numpy_float32(dcf).reshape(-1)
        self.im_size = tuple(int(v) for v in im_size)
        self.eps = float(eps)
        if len(self.im_size) != 2 or min(self.im_size) < 1:
            raise ValueError("im_size must contain two positive dimensions")
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError("trajectory_cycles must have shape (N, 2)")
        if len(trajectory) == 0 or len(density) != len(trajectory):
            raise ValueError("trajectory and DCF must have the same nonzero length")
        if not np.isfinite(trajectory).all() or np.abs(trajectory).max() > 0.50001:
            raise ValueError("trajectory must be finite, in cycles/pixel [-0.5, 0.5]")
        if not np.isfinite(density).all() or (density < 0).any():
            raise ValueError("DCF must be finite and nonnegative")
        density_sum = float(density.sum(dtype=np.float64))
        if density_sum <= 0:
            raise ValueError("DCF must have positive total weight")
        self.adjoint_scale = float(4 * np.prod(self.im_size) / density_sum)
        # Geometry is supplied by the authoritative cache manifest, not learned
        # state: evaluation may instantiate budgets absent during training.
        self.register_buffer("trajectory_cycles", torch.from_numpy(trajectory), persistent=False)
        self.register_buffer(
            "dcf", torch.from_numpy((density * self.adjoint_scale).astype(np.float32)),
            persistent=False,
        )
        self.__dict__["_operator"] = None
        self.__dict__["_operator_key"] = None

    @property
    def n_samples(self) -> int:
        return self.trajectory_cycles.shape[0]

    def clear_plan(self) -> None:
        """Release the cached plan when switching acquisition budgets."""
        self.__dict__["_operator"] = None
        self.__dict__["_operator_key"] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_operator"] = None
        state["_operator_key"] = None
        return state

    def _ensure_operator(self, batch: int, coils: int, device: torch.device):
        if device.type != "cuda":
            raise RuntimeError("RadialOperator computation requires the cuFINUFFT CUDA backend")
        key = (batch, coils, str(device))
        if key != self._operator_key:
            import cupy as cp
            from mrinufft import get_operator

            self.clear_plan()
            n_trans = min(32, batch * coils)
            while batch * coils % n_trans:
                n_trans -= 1
            samples = self.trajectory_cycles.detach().cpu().numpy()
            with cp.cuda.Device(device.index if device.index is not None else 0):
                operator = get_operator("cufinufft")(
                    samples=samples,
                    shape=self.im_size,
                    n_coils=coils,
                    n_batchs=batch,
                    n_trans=n_trans,
                    density=None,
                    squeeze_dims=False,
                    eps=self.eps,
                    upsampfac=2.0,
                    gpu_method=2,
                ).make_autograd(wrt_data=True)
            self.__dict__["_operator"] = operator
            self.__dict__["_operator_key"] = key
        return self._operator

    def fwd(self, image: torch.Tensor) -> torch.Tensor:
        """``(B,C,H,W) -> (B,C,N)``; density weights are not part of encoding."""
        if image.ndim != 4 or tuple(image.shape[-2:]) != self.im_size:
            raise ValueError(f"image must have shape (B,C,{self.im_size[0]},{self.im_size[1]})")
        if not image.is_complex():
            raise ValueError("fwd expects a complex image")
        operator = self._ensure_operator(image.shape[0], image.shape[1], image.device)
        with torch.amp.autocast("cuda", enabled=False):
            return operator.op(image.to(torch.complex64).contiguous())

    def adj_unweighted(self, samples: torch.Tensor) -> torch.Tensor:
        """``(B,C,N) -> (B,C,H,W)``; the Hermitian adjoint of ``fwd``."""
        if samples.ndim != 3 or samples.shape[-1] != self.n_samples:
            raise ValueError(f"samples must have shape (B,C,{self.n_samples})")
        if not samples.is_complex():
            raise ValueError("adj_unweighted expects complex samples")
        operator = self._ensure_operator(samples.shape[0], samples.shape[1], samples.device)
        with torch.amp.autocast("cuda", enabled=False):
            return operator.adj_op(samples.to(torch.complex64).contiguous())

    def adj(self, samples: torch.Tensor) -> torch.Tensor:
        """Normalized density-weighted adjoint, with correct sample gradients."""
        density = self.dcf.to(device=samples.device).view(1, 1, -1)
        return self.adj_unweighted(samples.to(torch.complex64) * density)
