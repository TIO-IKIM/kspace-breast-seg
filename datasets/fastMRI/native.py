"""Retained physical-coil loading, ESPIRiT combination and label geometry."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
import h5py
import numpy as np
from src.native_spokes.common import atomic_json as _atomic_json
from src.native_spokes.acquisition import readout_slice, normalization_indices
N_SLICES = 192
IMAGE_SIZE = 320
ESPIRIT_SETTINGS = dict(calib_width=32, thresh=0.02, kernel_width=6, crop=0.95, max_iter=100)

def _file_signature(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    st = source.stat()
    return {"path": str(source), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}

def _espirit(coil_images, device_index: int):
    """Calibrate only the provided retained-baseline coil images."""
    import cupy as cp
    import sigpy as sp
    from sigpy.mri.app import EspiritCalib
    import torch
    from src.native_spokes.physics import fft2c

    cartesian = fft2c(coil_images).squeeze(0).contiguous()
    with cp.cuda.Device(device_index):
        calibration = cp.from_dlpack(cartesian)
        maps = EspiritCalib(calibration, device=sp.Device(device_index), show_pbar=False,
                            **ESPIRIT_SETTINGS).run()
        finite = bool(cp.isfinite(maps).all().item())
        support = int(cp.count_nonzero(cp.sum(cp.abs(maps) ** 2, axis=0) > 1e-6).item())
        if not finite:
            raise RuntimeError(f"ESPIRiT calibration failed: finite={finite}, support_pixels={support}")
        result = torch.from_dlpack(maps).unsqueeze(0).clone()
    return result, support

def _prepare_label(case: dict, case_dir: Path) -> dict[str, Any]:
    import SimpleITK as sitk

    if case.get("has_target_lesion") == 0:
        if not case.get("target_absence_reason"):
            raise ValueError("An absent target requires explicit examination-level label provenance")
        geometry = dict(shape_dhw=[N_SLICES, IMAGE_SIZE, IMAGE_SIZE], positive_voxels=0,
            annotation_kind="target absent", target_absence_reason=case["target_absence_reason"],
            source=case["label_source"], coordinates="native reconstruction voxel grid; physical spacing unspecified")
        label_path = case_dir / "label_mask.npy"
        if not label_path.exists():
            temporary = label_path.with_suffix('.npy.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, np.zeros((N_SLICES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
            temporary.replace(label_path)
        _atomic_json(case_dir / "label_geometry.json", geometry)
        return geometry
    if not case.get("mask_path"):
        raise ValueError("A target-positive examination requires its annotation")
    mask_path = Path(case["mask_path"])
    volume = sitk.ReadImage(str(mask_path))
    label = (sitk.GetArrayFromImage(volume) > 0).astype(np.uint8)
    if label.shape != (N_SLICES, IMAGE_SIZE, IMAGE_SIZE) or not label.any():
        raise ValueError(f"Invalid positive label shape/content {label.shape}: {mask_path}")
    geometry = {
        "shape_dhw": list(label.shape), "spacing_xyz": list(volume.GetSpacing()),
        "spacing_dhw": list(volume.GetSpacing())[::-1], "origin_xyz": list(volume.GetOrigin()),
        "direction_xyz": list(volume.GetDirection()), "positive_voxels": int(label.sum()),
        "source": _file_signature(mask_path),
    }
    label_path = case_dir / "label_mask.npy"
    geometry_path = case_dir / "label_geometry.json"
    if geometry_path.exists() and json.loads(geometry_path.read_text()) != geometry:
        raise RuntimeError(f"Stale label geometry: {case_dir}; choose a new cache root")
    if label_path.exists():
        saved = np.load(label_path, mmap_mode="r", allow_pickle=False)
        if saved.shape != label.shape or saved.dtype != np.uint8 or not np.array_equal(saved, label):
            raise RuntimeError(f"Label cache differs from its source: {label_path}")
        if not geometry_path.exists():
            _atomic_json(geometry_path, geometry)
        return geometry
    fd, temporary = tempfile.mkstemp(prefix=".label_mask.", suffix=".npy", dir=case_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, label, allow_pickle=False)
        os.replace(temporary, label_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    _atomic_json(geometry_path, geometry)
    return geometry

def load_frame(raw_path, selected, percent):
    import sigpy as sp
    cut = readout_slice(percent)
    with h5py.File(raw_path, 'r') as raw:
        data = raw['kspace']
        if data.shape[:4] != (2, 288, 640, 16) or not 1 <= data.shape[-1] <= 192:
            raise ValueError(f'Unexpected acquisition shape: {data.shape}')
        partitions = data.shape[-1]
        kept = data[:, selected, cut, :, :]
    values = (kept[0]+1j*kept[1]).astype(np.complex64)
    if not np.isfinite(values).all() or not np.any(values):
        raise FloatingPointError(f'Invalid retained measurements: {raw_path}')
    reference = normalization_indices(percent)
    center = np.ascontiguousarray(values[:, reference-cut.start])
    if center.shape[1] != len(reference):
        raise ValueError('Normalization reference is not contained in retained readout')
    scale = float(np.sqrt(np.mean(np.abs(center)**2, dtype=np.float64)))
    if not np.isfinite(scale) or scale <= 0:
        raise FloatingPointError('Invalid retained-center sample RMS')
    center_hash = hashlib.sha256(center.tobytes()).hexdigest()
    values = values.transpose(3, 2, 0, 1)
    padded = np.zeros((192, 16, len(selected), cut.stop-cut.start), dtype=np.complex64)
    offset = 96-partitions//2; padded[offset:offset+partitions] = values
    return sp.fft(padded, axes=(0,)).astype(np.complex64, copy=False), scale, partitions, center_hash
