"""Physical-coil combined inputs for the three-model angular CV campaign.

Uses the same retained measurements, physical ESPIRiT combination and scaling
as revision preparation, while storing only the field these models consume.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from src.native_spokes.common import sha256, digest
from src.native_spokes.acquisition import VERSION, condition_key, normalization_indices, operators
from src.native_spokes.sampling import POLICY, selected_indices

ROOT = Path(__file__).resolve().parents[2]


def revision():
    spec = importlib.util.spec_from_file_location('combined_revision_reference', ROOT/'datasets/fastMRI/native.py')
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def source_hashes():
    names = ('datasets/fastMRI/prepare_native_combined.py', 'datasets/fastMRI/native.py', 'src/native_spokes/physics.py',
             'src/native_spokes/acquisition.py', 'src/native_spokes/sampling.py')
    return {name: sha256(ROOT/name) for name in names}


def source_metadata(case, r, percent, geometry_fingerprint, version=VERSION):
    ref = revision()
    return dict(preprocessing_version=version, storage='combined-only-v1',
        sampling_policy=POLICY if version == VERSION else version,
        source_code=source_hashes(), exam_id=case['exam_id'], person_id=case['person_id'],
        raw=ref._file_signature(case['raw_path']),
        mask=ref._file_signature(case['mask_path']) if case.get('mask_path') else dict(
            has_target_lesion=0, reason=case['target_absence_reason'], source=case['label_source']),
        geometry_fingerprint=geometry_fingerprint, spoke_reduction=r, readout_percent=percent,
        calibration='Physical-coil ESPIRiT from retained precontrast samples only; shared across windows',
        normalization='Retained precontrast physical coils: up to 64 central samples, limited to retained readout',
        combination='Physical-coil SENSE combination followed by complex second-postcontrast minus precontrast subtraction')




def prepare_case(case, cache_base, conditions):
    import torch
    import cupy as cp
    from src.native_spokes.paired_spokes import VERSION as PAIRED_VERSION, paired_operators
    from src.native_spokes.physics import fft2c, sense_combine
    prep = revision(); ref = prep; cache_base = Path(cache_base)
    label_dir = cache_base/'labels'/case['exam_id']; label_dir.mkdir(parents=True, exist_ok=True)
    label_geometry = ref._prepare_label(case, label_dir)
    for r, percent in conditions:
        directory = cache_base/f'f{percent}'/case['exam_id']; directory.mkdir(parents=True, exist_ok=True)
        label = directory/'label_mask.npy'
        if not label.exists():
            label.symlink_to((label_dir/'label_mask.npy').resolve())
        path = directory/f'R{r}.h5'
        with np.load(cache_base/f'f{percent}/geometry/R{r}.npz', allow_pickle=False) as f:
            geometry = {key: f[key] for key in f.files}
        version = str(geometry['preprocessing_version'].item())
        paired = version == PAIRED_VERSION
        selected = geometry['selected_indices'] if paired else selected_indices(r)
        readout = geometry.get('readout_indices', np.arange(640, dtype=np.int64))
        if (paired and (r <= 8 or percent != 100)) or not np.array_equal(geometry['selected_indices'], selected) or len(readout) != 640*percent//100:
            raise ValueError('Combined preparation requires the agreed retained native geometry')
        source = source_metadata(case, r, percent, str(geometry['fingerprint'].item()), version); fingerprint = digest(source)
        if path.exists():
            with h5py.File(path, 'r') as f:
                if not f.attrs.get('complete') or f.attrs.get('fingerprint') != fingerprint:
                    raise ValueError(f'Different existing combined input: {path}')
            continue
        first, scale, partitions, reference_hash = prep.load_frame(case['raw_path'], selected[0], percent)
        last, _, last_partitions, _ = prep.load_frame(case['raw_path'], selected[1], percent)
        if partitions != last_partitions:
            raise ValueError('Temporal partition geometry differs')
        first /= scale; last /= scale
        physical_ops = [op.cuda() for op in (paired_operators(geometry) if paired else operators(geometry))]
        metadata = dict(**source, fingerprint=fingerprint, normalization_scale=scale,
            normalization_reference_sha256=reference_hash,
            normalization_reference_indices=normalization_indices(percent).tolist(),
            selected_original_spokes=selected.tolist(), readout_indices=readout.tolist(),
            constant_image_gain=geometry['constant_image_gain'].tolist(), image_grid=label_geometry,
            acquired_partitions=partitions, retained_spokes_per_window=len(selected[0]),
            retained_readout_samples=640*percent//100, physical_map_support_pixels=[])
        fd, temporary = tempfile.mkstemp(prefix=f'.R{r}.', suffix='.h5', dir=directory); os.close(fd)
        try:
            with h5py.File(temporary, 'w') as out, torch.inference_mode():
                stored = out.create_dataset('combined_kspace', shape=(192, 320, 320), dtype=np.complex64,
                    chunks=(1, 320, 320), compression='lzf', shuffle=True)
                for z in range(192):
                    raw = [torch.as_tensor(x[z].reshape(1, 16, -1), device='cuda') for x in (first, last)]
                    images = [op.adj(y) for op, y in zip(physical_ops, raw)]
                    maps, support = ref._espirit(images[0], torch.cuda.current_device())
                    difference = sense_combine(images[1], maps)-sense_combine(images[0], maps)
                    values = fft2c(difference).reshape(320, 320)
                    if not torch.isfinite(values).all():
                        raise FloatingPointError(f'Nonfinite combined input: {case["exam_id"]}/{condition_key(r, percent)}/{z}')
                    stored[z] = values.cpu().numpy(); metadata['physical_map_support_pixels'].append(support)
                if not any(metadata['physical_map_support_pixels']):
                    raise RuntimeError('No usable retained physical-coil calibration support')
                if source != source_metadata(case, r, percent, str(geometry['fingerprint'].item()), version):
                    raise ValueError('Source changed during combined preparation')
                out.attrs.update(complete=True, fingerprint=fingerprint, preprocessing_version=version,
                                 metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False))
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
            for op in physical_ops:
                op.clear_plan()
            cp.get_default_memory_pool().free_all_blocks()
        print(json.dumps(dict(exam_id=case['exam_id'], condition=condition_key(r, percent), complete=True)), flush=True)
