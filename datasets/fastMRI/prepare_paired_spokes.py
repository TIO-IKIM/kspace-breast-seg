"""Rebuild only R>8 combined inputs from retained native physical-coil data."""
from pathlib import Path
import json

import h5py
import numpy as np


def prepare_case(case, original_cache, output_cache, geometry, reduction):
    import torch
    import cupy as cp
    from src.native_spokes.common import module
    from src.native_spokes.paired_spokes import paired_operators, VERSION
    from src.native_spokes.physics import fft2c, sense_combine

    native = module('datasets/fastMRI/native.py', 'paired_native_loading')
    directory = Path(output_cache) / 'f100' / case['exam_id']; directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'R{reduction}.h5'
    if path.exists():
        return path
    selected = geometry['selected_indices']
    with h5py.File(Path(original_cache) / 'f100' / case['exam_id'] / f'R{reduction}.h5', 'r') as handle:
        previous = json.loads(handle.attrs['metadata_json'])
    first, _, partitions, _ = native.load_frame(case['raw_path'], selected[0], 100)
    last, _, _, _ = native.load_frame(case['raw_path'], selected[1], 100)
    scale = float(previous['normalization_scale']); first /= scale; last /= scale
    metadata = dict(normalization_scale=scale, selected_original_spokes=selected.tolist(),
        normalization_reference_indices=list(range(288, 352)), acquired_partitions=partitions,
        parent_geometry_fingerprint=previous['geometry_fingerprint'])
    operators = [operator.cuda() for operator in paired_operators(geometry)]
    metadata.update(constant_image_gain=geometry['constant_image_gain'].tolist(),
        calibration='Original retained baseline, CPU partition FFT, physical-coil ESPIRiT; shared across windows')
    support_pixels = []
    temporary = path.with_suffix('.h5.tmp')
    try:
        with h5py.File(temporary, 'w') as output, torch.inference_mode():
            values = output.create_dataset('combined_kspace', shape=(192, 320, 320), dtype=np.complex64,
                chunks=(1, 320, 320), compression='lzf', shuffle=True)
            for z in range(192):
                raw = [torch.as_tensor(frame[z].reshape(1, 16, -1), device='cuda') for frame in (first, last)]
                images = [operator.adj(data) for operator, data in zip(operators, raw)]
                maps, support = native._espirit(images[0], torch.cuda.current_device())
                support_pixels.append(support)
                difference = sense_combine(images[1], maps)-sense_combine(images[0], maps)
                combined = fft2c(difference).reshape(320, 320)
                if not torch.isfinite(combined).all():
                    raise FloatingPointError(f'Nonfinite paired input: {case["exam_id"]}/R{reduction}')
                values[z] = combined.cpu().numpy()
            if not any(support_pixels):
                raise RuntimeError(f'No retained-baseline calibration support: {case["exam_id"]}/R{reduction}')
            metadata['physical_map_support_pixels'] = support_pixels
            output.attrs.update(complete=True, preprocessing_version=VERSION,
                                metadata_json=json.dumps(metadata, sort_keys=True))
        temporary.replace(path)
    finally:
        for operator in operators:
            operator.clear_plan()
        cp.get_default_memory_pool().free_all_blocks()
    return path
