"""Combined-signal image, hybrid and direct k-space segmentation models."""
from __future__ import annotations
import hashlib
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as checkpoint_function
from src.model.modules import UNet3D, UNet3D_K2Img
from src.native_spokes.physics import fft2c, ifft2c
from src.native_spokes.augmentation import RECIPE, draw_batch, transform_batch

ARMS = ('combined_magnitude', 'combined_complex', 'combined_hybrid', 'native_kspace')


def reference_phase(k, sigma):
    """Divide out the input's own smooth phase; the magnitude image is unchanged.

    sigpy's ESPIRiT maps carry the first coil's phase, a smooth field that differs per
    examination and per spoke budget. Its estimate is the phase of a Gaussian k-space
    low-pass of the input (``sigma`` in samples), so only the retained data are used.
    """
    rows = torch.arange(k.shape[-2], device=k.device, dtype=torch.float32) - k.shape[-2] // 2
    columns = torch.arange(k.shape[-1], device=k.device, dtype=torch.float32) - k.shape[-1] // 2
    window = torch.exp(-(rows[:, None]**2 + columns[None, :]**2) / (2 * sigma**2))
    phase = torch.angle(ifft2c(k * window))
    return fft2c(ifft2c(k) * torch.polar(torch.ones_like(phase), -phase))


def state_digest(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        value.update(name.encode()); value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode()); value.update(tensor.numpy().tobytes())
    return value.hexdigest()


class NativeSegmentationArm(nn.Module):
    def __init__(self, arm, checkpoint='off', phase_reference=None):
        super().__init__()
        if arm not in ARMS: raise ValueError(f'Unknown maintained model: {arm}')
        if checkpoint not in ('off', 'full'): raise ValueError('Use off/full activation checkpointing')
        self.arm = arm; self.required_fields = ('combined_kspace',)
        self.use_checkpoint = checkpoint == 'full'
        self.phase_reference = phase_reference
        self.architecture = dict(arm=arm, image_depth=3, patch_depth=24)
        if phase_reference is not None:
            self.architecture.update(phase_reference_sigma=phase_reference)
        if arm == 'native_kspace':
            self.network = UNet3D(input_shape=(1, 2, 24, 320, 320), output_shape=(1, 2, 24, 320, 320),
                                  hidden_factor=24, depth=3)
            self.architecture.pop('image_depth')
            self.architecture.update(hidden_factor=24, kspace_depth=3,
                output='one complex foreground-mask spectrum', learned_image_head=False,
                decoding='Centered iFFT2, magnitude, per-patch maximum normalization',
                strides=((2, 2, 2),) * 3)
        elif arm == 'combined_hybrid':
            self.network = UNet3D_K2Img(input_shape=(1, 2, 24, 320, 320), output_shape=(2, 1, 24, 320, 320),
                hidden_factor=20, depth=3, mid_channels=4, depth_k=2, strides_k=((2, 1, 1), (2, 1, 1)),
                up_kernel_size_k=(3, 1, 1), up_kernel_size_img=(3, 3, 3))
            self.architecture.update(hidden_factor=20, mid_channels=4, depth_k=2, strides_k=((2, 1, 1), (2, 1, 1)), input_complex_coils=1)
        else:
            parts = 1 if arm == 'combined_magnitude' else 2
            self.network = UNet3D(input_shape=(1, parts, 24, 320, 320), output_shape=(2, 1, 24, 320, 320), hidden_factor=24, depth=3)
            self.architecture.update(hidden_factor=24, image_parts=parts)

    def forward(self, batch):
        k = batch['combined_kspace'].to(torch.complex64)
        if self.phase_reference is not None:
            with torch.autocast(k.device.type, enabled=False):
                k = reference_phase(k, self.phase_reference)
        if self.arm in ('combined_hybrid', 'native_kspace'):
            x = torch.stack((k.real, k.imag), dim=1).unsqueeze(1).contiguous()
        else:
            image = ifft2c(k)
            x = (image.abs().unsqueeze(1).unsqueeze(2) if self.arm == 'combined_magnitude'
                 else torch.stack((image.real, image.imag), dim=1).unsqueeze(1).contiguous())
        if self.use_checkpoint and self.training and torch.is_grad_enabled() and x.is_cuda:
            logits = checkpoint_function(self.network, x, use_reentrant=False)
        else: logits = self.network(x)
        if self.arm == 'native_kspace':
            if self.training:
                return logits
            from src.utils.kspace_ops import kspace_to_pixel_probs
            return kspace_to_pixel_probs(logits).squeeze(2)
        return logits.squeeze(2)


class AugmentedArm(nn.Module):
    """Preserve base.network and augmentation_samples checkpoint keys."""
    def __init__(self, arm, mode, seed, phase_reference=None):
        super().__init__(); self.base = NativeSegmentationArm(arm, checkpoint=mode, phase_reference=phase_reference)
        self.seed, self.arm = seed, arm; self.required_fields = self.base.required_fields
        self.architecture = {**self.base.architecture, 'augmentation': RECIPE}
        self.register_buffer('augmentation_samples', torch.zeros((), dtype=torch.int64))

    def forward(self, batch):
        if self.training:
            params = draw_batch(batch, self.seed, int(self.augmentation_samples.item()))
            with torch.no_grad(), torch.autocast(batch['label'].device.type, enabled=False):
                if any(p['flips'] or p['angle'] != 0 or p['zoom'] != 1 for p in params):
                    batch['combined_kspace'] = fft2c(transform_batch(ifft2c(batch['combined_kspace'].to(torch.complex64)), params))
                    batch['label'] = transform_batch(batch['label'], params, label=True)
            self.augmentation_samples.add_(len(params))
        return self.base(batch)


def build(arm, geometry=None, training=None, seed=123, device='cuda'):
    mode = (training or {}).get('execution', {}).get(arm, {}).get('checkpoint', 'off')
    return AugmentedArm(arm, mode, seed, (training or {}).get('phase_reference_sigma')).to(device)
