"""Shared source snapshots and full-volume segmentation evaluation."""
from pathlib import Path
import json
import shutil
import numpy as np
from src.native_spokes.acquisition import EVALUATION_CONDITIONS as CONDITIONS

ROOT = Path(__file__).resolve().parents[2]
MODELS = ('combined_magnitude', 'combined_complex', 'native_kspace', 'combined_hybrid')
LABELS = dict(combined_magnitude='Mag-Seg', combined_complex='Complex-Seg',
              native_kspace='K-Seg', combined_hybrid='K2I-Seg')


def snapshot_source(destination):
    destination = Path(destination)
    for directory in ('src', 'datasets', 'experiments'):
        for source in (ROOT / directory).rglob('*.py'):
            if '__pycache__' in source.parts:
                continue
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def read(path):
    return json.loads(Path(path).read_text())


def predict_array(model, kspace, label, batch_size=4):
    import torch
    from src.native_spokes.training import patch_starts, volume_metrics
    starts = patch_starts(len(label),24,16)
    votes = np.zeros(label.shape,dtype=np.uint16); visits = np.zeros(len(label),dtype=np.uint16)
    device = next(model.parameters()).device
    with torch.no_grad():
        for first in range(0,len(starts),batch_size):
            positions = starts[first:first+batch_size]
            x = torch.as_tensor(np.stack([np.asarray(kspace[z:z+24],dtype=np.complex64) for z in positions]),device=device)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                prediction = model({'combined_kspace':x}).argmax(1).to(torch.uint8).cpu().numpy()
            for z,mask in zip(positions,prediction):
                votes[z:z+24] += mask; visits[z:z+24] += 1
    if not visits.all():
        raise ValueError('Patch voting left uncovered slices')
    result = 2*votes > visits[:,None,None]
    return result, volume_metrics(result,label)
