"""MAMA-MIA replay with the original baseline and matched postcontrast spokes."""
import numpy as np


def condition_input(measurements, geometry, chunk=16):
    import torch
    from src.native_spokes.paired_spokes import paired_operators
    from src.native_spokes.physics import fft2c

    selected = geometry['selected_indices']
    retained = np.stack([measurements[t][:, selected[t]-start, :]
                         for t, start in enumerate((0, 192))])
    center = np.ascontiguousarray(retained[0, :, :, 288:352])
    scale = float(np.sqrt(np.mean(np.abs(center)**2, dtype=np.float64)))
    if not np.isfinite(scale) or scale <= 0:
        raise FloatingPointError('Invalid unchanged precontrast normalization')
    operators = [operator.cuda() for operator in paired_operators(geometry)]
    depth = measurements.shape[1]; output = np.empty((depth, 320, 320), dtype=np.complex64)
    try:
        with torch.inference_mode():
            for start in range(0, depth, chunk):
                stop = min(depth, start+chunk)
                images = [operators[t].adj(torch.as_tensor(retained[t, start:stop]/scale,
                    device='cuda').reshape(stop-start, 1, -1)) for t in range(2)]
                values = fft2c(images[1]-images[0])[:, 0]
                if not torch.isfinite(values).all():
                    raise FloatingPointError('Nonfinite paired MAMA-MIA input')
                output[start:stop] = values.cpu().numpy()
    finally:
        for operator in operators:
            operator.clear_plan()
    return output, dict(normalization_scale=scale, normalization_reference_indices=list(range(288, 352)),
                        selected_original_spokes=selected.tolist())
