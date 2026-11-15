"""Acquisition conditions and geometry-only weighted-adjoint normalization."""
from __future__ import annotations

import numpy as np
import torch

from src.native_spokes.physics import RadialOperator

LEGACY_VERSION = 'native-revision-center64-constant-gain-v1'
ANGULAR_VERSION = 'native-revision-angular-center64-fallback32-v2'
VERSION = 'native-revision-angular-retained-center-v3'
READOUTS = (100, 75, 50, 25, 10, 5, 2)
ADDITIONAL_READOUTS = ()
SPOKES = (1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 48)
EVALUATION_CONDITIONS = tuple((r, 100) for r in SPOKES) + tuple((1, f) for f in READOUTS[1:])


def condition_key(r, percent=100):
    return f'R{int(r)}_F{int(percent)}'


def readout_slice(percent):
    if percent not in READOUTS + ADDITIONAL_READOUTS:
        raise ValueError(f'Unsupported readout percentage: {percent}')
    count = 640 * percent // 100
    start = (640-count)//2
    return slice(start, start+count)


def normalization_indices(percent):
    """Use up to 64 retained central samples (32 at 5%, 12 at nominal 2%)."""
    cut = readout_slice(percent)
    count = min(64, cut.stop-cut.start)
    start = (640-count)//2
    return np.arange(start, start+count, dtype=np.int64)


class RevisionRadialOperator(RadialOperator):
    def __init__(self, trajectory, dcf, constant_image_gain, im_size=(320, 320)):
        super().__init__(trajectory, dcf, im_size)
        gain = float(constant_image_gain)
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError('A finite positive geometry-probe gain is required')
        self.constant_image_gain = gain
        self.adjoint_scale /= gain
        # Apply AFTER the legacy constructor's impulse-peak normalization;
        # passing a scaled DCF into that constructor alone would cancel it.
        self.dcf.div_(gain)


def operators(item):
    if str(np.asarray(item['preprocessing_version']).item()) not in (VERSION, ANGULAR_VERSION, LEGACY_VERSION):
        raise ValueError('Revision physics requires revision geometry')
    return [RevisionRadialOperator(item['trajectory'][t], item['dcf'][t],
                                  item['constant_image_gain'][t]) for t in range(2)]


def analytic_constant_gain(trajectory, density, size=320):
    """Center response under the original impulse-peak-normalized operator."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    transform = size*np.sinc(size*trajectory)/np.sinc(trajectory)*np.exp(1j*np.pi*trajectory)
    return float(abs(np.sum(density*transform[:, 0]*transform[:, 1])/density.sum()))


def check_fourier_operator(device='cuda'):
    """Small direct-DFT and Hermitian-pair execution checks, not a test suite."""
    rng = np.random.default_rng(123)
    n, count = 16, 64
    trajectory = rng.uniform(-.45, .45, (count, 2)).astype(np.float32)
    op = RadialOperator(trajectory, np.ones(count, np.float32), im_size=(n, n)).to(device)
    x = torch.as_tensor(rng.normal(size=(1, 1, n, n))+1j*rng.normal(size=(1, 1, n, n)),
                        dtype=torch.complex64, device=device)
    y = torch.as_tensor(rng.normal(size=(1, 1, count))+1j*rng.normal(size=(1, 1, count)),
                        dtype=torch.complex64, device=device)
    yy, xx = np.meshgrid(np.arange(n)-n//2, np.arange(n)-n//2, indexing='ij')
    matrix = np.exp(-2j*np.pi*(trajectory[:, 0, None]*yy.reshape(-1)
                              + trajectory[:, 1, None]*xx.reshape(-1)))/(2*n)
    direct = torch.as_tensor(matrix, dtype=torch.complex64, device=device) @ x.reshape(-1)
    measured = op.fwd(x).reshape(-1)
    forward_error = float((measured-direct).norm()/direct.norm())
    lhs = torch.vdot(measured, y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), op.adj_unweighted(y).reshape(-1))
    adjoint_error = float((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-12))
    if forward_error > 2e-4 or adjoint_error > 2e-4:
        raise RuntimeError(f'Fourier convention/adjoint check failed: {forward_error}, {adjoint_error}')
    return dict(direct_dft_relative_error=forward_error, adjoint_relative_error=adjoint_error)
