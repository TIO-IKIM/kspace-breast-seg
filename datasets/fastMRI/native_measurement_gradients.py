"""Fixed-calibration native-measurement frontend for frozen-model diagnostics.

The input is the acquired complex partition data, before the partition FFT.
Only the existing retained spokes enter this analysis. Calibration, density
weights, normalization and geometry are constants in the differentiated map.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from src.native_spokes.physics import fft2c, sense_combine
from src.native_spokes.acquisition import operators


def reference(root):
    path = Path(__file__).with_name('native.py')
    spec = importlib.util.spec_from_file_location('gradient_preparation_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_measurements(path, selected):
    """Return (time, acquired partition, physical coil, spoke, readout)."""
    frames = []
    with h5py.File(path, 'r') as handle:
        data = handle['kspace']
        if data.shape[:4] != (2, 288, 640, 16) or not 1 <= data.shape[-1] <= 192:
            raise ValueError(f'Unexpected acquisition shape: {data.shape}')
        for indices in selected:
            values = data[:, indices, :, :, :]
            frames.append((values[0] + 1j * values[1]).astype(np.complex64).transpose(3, 2, 0, 1))
    result = np.ascontiguousarray(np.stack(frames))
    if not np.isfinite(result).all() or not np.any(result):
        raise ValueError('Invalid acquired measurements')
    return result


def partition_fft(values, start, depth):
    """Match SigPy's centered orthonormal FFT and the acquisition's zero pad."""
    partitions = values.shape[1]
    offset = 96 - partitions // 2
    shape = (values.shape[0], 192, *values.shape[2:])
    padded = values.new_zeros(shape)
    padded[:, offset:offset + partitions] = values
    slices = torch.fft.fftshift(torch.fft.fft(
        torch.fft.ifftshift(padded, dim=(1,)), dim=1, norm='ortho'), dim=(1,))
    return slices[:, start:start + depth].contiguous()


class MeasurementFrontend(nn.Module):
    """Linear map with fixed ESPIRiT maps and the original two time windows."""

    def __init__(self, geometry, maps, start):
        super().__init__()
        self.ops = nn.ModuleList(operators(geometry))
        self.register_buffer('maps', maps.detach())
        self.start, self.depth = int(start), len(maps)

    def forward(self, acquired):
        slices = partition_fft(acquired, self.start, self.depth)
        combined = []
        # The original preparation uses one slice per NUFFT call. Preserve that
        # convention for numerical equivalence and modest workspace memory.
        for z in range(self.depth):
            frames = [sense_combine(self.ops[t].adj(slices[t, z:z + 1].reshape(1, 16, -1)),
                                    self.maps[z:z + 1]) for t in range(2)]
            combined.append(fft2c(frames[1] - frames[0])[0])
        return torch.stack(combined)[None]

    def close(self):
        for op in self.ops:
            op.clear_plan()


def prepare_frontend(case, cache, geometry, start, depth, source_root, saved_maps=None):
    """Recreate retained calibration and verify the frozen cache's input."""
    import json
    import cupy as cp

    cache = Path(cache)
    with h5py.File(cache, 'r') as handle:
        metadata = json.loads(handle.attrs['metadata_json'])
        cached = np.asarray(handle['combined_kspace'][start:start + depth])
    raw = load_measurements(case['raw_path'], geometry['selected_indices'])
    # Hash acquired values as well as the file signature in the result receipt.
    measurement_hash = hashlib.sha256(raw.tobytes()).hexdigest()
    center = raw[0, :, :, :, 288:352]
    scale = float(np.sqrt(np.mean(np.abs(center) ** 2, dtype=np.float64)))
    if not np.isclose(scale, metadata['normalization_scale'], rtol=2e-6, atol=0):
        raise ValueError('Retained normalization disagrees with the parent cache')
    scale = float(metadata['normalization_scale'])
    values = torch.as_tensor(raw / scale, device='cuda')
    del raw
    if saved_maps is not None:
        maps = torch.as_tensor(np.load(saved_maps, allow_pickle=False), device='cuda')
    else:
        ref = reference(source_root)
        # Reproduce the cache's calibration inputs in exactly its operation
        # order: CPU partition FFT, complex64 cast, then normalization. Small
        # FFT roundoff differences can otherwise change ESPIRiT support/phase.
        preparation = ref
        first, original_scale, _, center_hash = preparation.load_frame(
            case['raw_path'], geometry['selected_indices'][0], 100)
        if (original_scale != scale or center_hash != metadata['normalization_reference_sha256']):
            raise ValueError('Original calibration reference or scale changed')
        first /= scale
        op = operators(geometry)[0].cuda()
        maps_list = []
        with torch.no_grad():
            slices = torch.as_tensor(first[start:start + depth], device='cuda')
            for z in range(depth):
                image = op.adj(slices[z:z + 1].reshape(1, 16, -1))
                maps, _ = ref._espirit(image, torch.cuda.current_device())
                maps_list.append(maps[0])
            maps = torch.stack(maps_list)
        del slices, image, maps_list, first
        op.clear_plan()
        cp.get_default_memory_pool().free_all_blocks()
    frontend = MeasurementFrontend(geometry, maps, start).cuda()
    with torch.no_grad():
        rebuilt = frontend(values)
        target = torch.as_tensor(cached[None], device='cuda')
        relative_error = float((rebuilt - target).norm() / target.norm())
    if relative_error > 5e-4:
        raise ValueError(f'Recreated fixed-calibration input differs from cache: {relative_error}')
    metadata = dict(normalization_scale=scale, native_shape=list(values.shape),
                    calibration_reference='Original frozen CPU partition FFT, complex64 cast, then normalization',
                    native_measurement_sha256=measurement_hash,
                    cache_relative_error=relative_error,
                    cached_input_sha256=hashlib.sha256(cached.tobytes()).hexdigest())
    return frontend, values, target, metadata
