"""Shared, differentiable geometry; a separate deterministic draw per patch."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F

import hashlib
import random

RECIPE = dict(horizontal_flip_probability=.5, depth_flip_probability=.5,
              rotation_probability=.5, rotation_degrees=[-15., 15.],
              zoom_probability=.5, zoom_range=[.85, 1.25],
              vertical_flip=False, quarter_turn=False, translation=False,
              noise=False, gamma=False, readout_augmentation=False,
              complex_interpolation='bilinear, same grid for real/imag',
              label_interpolation='nearest', align_corners=False, padding='zeros')
IDENTITY = dict(flips=(), angle=0., zoom=1.)


def parameters(seed, occurrence, exam, z):
    # Preserve the exact winning stream, including disabled historical draws.
    def stream(name):
        key = f'{seed}:{occurrence}:{exam}:{z}:{name}'.encode()
        return random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little'))
    discrete, continuous = stream('discrete'), stream('continuous')
    all_flips = tuple(d for d in (-1, -2, -3) if discrete.random() < .5)
    if discrete.random() < .5:
        discrete.randint(-5, 5); discrete.randint(-5, 5)
    if discrete.random() < .5:
        discrete.randrange(4)
    angle = continuous.uniform(-15., 15.) if continuous.random() < .5 else 0.
    zoom = continuous.uniform(.85, 1.25) if continuous.random() < .5 else 1.
    return dict(flips=tuple(d for d in all_flips if d in (-1, -3)), angle=angle, zoom=zoom)


def draw_batch(batch, seed, first):
    return tuple(parameters(seed, first+i, str(exam), int(batch['z_start'][i]))
                 for i, exam in enumerate(batch['exam_id']))


def _affine(value, angle, zoom, mode):
    if angle == 0 and zoom == 1:
        return value
    height, width = value.shape[-2:]
    radians = math.radians(angle)
    c, s = math.cos(radians)/zoom, math.sin(radians)/zoom
    theta = torch.tensor([[c, s, 0.], [-s, c, 0.]], device=value.device, dtype=torch.float32)
    if value.is_complex():
        flat = value.reshape(-1, height, width)
        parts = torch.stack((flat.real.float(), flat.imag.float()), dim=1)
    else:
        parts = value.reshape(-1, 1, height, width).float()
    grid = F.affine_grid(theta[None].expand(len(parts), -1, -1), parts.shape, align_corners=False)
    warped = F.grid_sample(parts, grid, mode=mode, padding_mode='zeros', align_corners=False)
    result = torch.complex(warped[:, 0], warped[:, 1]) if value.is_complex() else warped[:, 0]
    return result.reshape(value.shape).to(value.dtype)


def transform(value, parameter, inverse=False, depth=True, label=False):
    """Transform (...,D,H,W); inverse maps a learned update back, not raw data."""
    flips = tuple(d for d in parameter['flips'] if depth or d != -3)
    with torch.autocast(value.device.type, enabled=False):
        if inverse:
            value = _affine(value, -parameter['angle'], 1/parameter['zoom'], 'nearest' if label else 'bilinear')
            if flips:
                value = torch.flip(value, flips)
        else:
            if flips:
                value = torch.flip(value, flips)
            value = _affine(value, parameter['angle'], parameter['zoom'], 'nearest' if label else 'bilinear')
    return value.contiguous()


def transform_batch(value, params, **kwargs):
    if len(value) != len(params):
        raise ValueError('One shared geometric transform per patch is required')
    if all(not p['flips'] and p['angle'] == 0 and p['zoom'] == 1 for p in params):
        return value
    return torch.stack([transform(v, p, **kwargs) for v, p in zip(value, params)])


def check_geometry():
    import random
    mask = torch.zeros(4, 32, 32, dtype=torch.long); mask[1:3, 8:16, 15:24] = 1
    image = torch.complex(mask.float(), 2*mask.float())
    assert transform(image, IDENTITY) is image or torch.equal(transform(image, IDENTITY), image)
    for flips in ((-1,), (-3,), (-1, -3)):
        p = dict(IDENTITY, flips=flips)
        transformed = transform(image, p)
        target = transform(mask, p, label=True)
        if not torch.equal(transformed.real, target.float()) or not torch.equal(transform(transformed, p, inverse=True), image):
            raise RuntimeError('Flip/image/target alignment failed')
    for angle, zoom in ((-15., .85), (15., 1.25)):
        p = dict(flips=(-1, -3), angle=angle, zoom=zoom)
        x = image.clone().requires_grad_()
        transformed = transform(x, p)
        target = transform(mask, p, label=True)
        if not torch.allclose(transformed.imag, 2*transformed.real, atol=2e-6) or not set(target.unique().tolist()) <= {0, 1}:
            raise RuntimeError('Affine complex/label interpolation mismatch')
        transform(transformed, p, inverse=True).abs().square().mean().backward()
        if x.grad is None or not torch.isfinite(x.grad).all() or not x.grad.abs().sum() > 0:
            raise RuntimeError('Augmentation detached the image graph')
    state, python_state = torch.get_rng_state().clone(), random.getstate()
    batch = dict(exam_id=['case']*4, z_start=torch.tensor([0, 16, 32, 48]))
    together = draw_batch(batch, 124, 7)
    split = tuple(draw_batch(dict(exam_id=[batch['exam_id'][i]], z_start=batch['z_start'][i:i+1]), 124, 7+i)[0] for i in range(4))
    if together != split or not torch.equal(state, torch.get_rng_state()) or random.getstate() != python_state:
        raise RuntimeError('Augmentation draw stream depends on batching or global RNG')
    return dict(identity=True, complex_label_alignment=True, differentiable_inverse_update=True,
                shared_draws_across_batch_sizes=True, private_rng=True)
