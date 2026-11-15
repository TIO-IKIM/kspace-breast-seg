"""Small manifest and provenance helpers for the maintained pipeline."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('combined_magnitude', 'combined_complex', 'combined_hybrid', 'native_kspace')
TRAIN_ACCELERATIONS = (1, 2, 4, 6, 8)
VALIDATION_ACCELERATIONS = (1, 4, 8)
EVAL_ACCELERATIONS = (1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 48)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): value.update(block)
    return value.hexdigest()


def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def module(path, name):
    """Load dataset-specific preparation code from the current source bundle."""
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def source_hashes():
    names = ('src/native_spokes/common.py', 'src/native_spokes/models.py', 'src/native_spokes/training.py',
        'src/native_spokes/physics.py', 'src/native_spokes/acquisition.py', 'src/native_spokes/sampling.py',
        'src/native_spokes/augmentation.py', 'src/native_spokes/paths.py',
        'src/model/modules.py', 'src/model/loss_full.py', 'src/utils/kspace_ops.py')
    return {name: sha256(ROOT / name) for name in names}


def validate_manifest(manifest):
    if manifest.get('manifest_hash') != digest({k: v for k, v in manifest.items() if k != 'manifest_hash'}):
        raise ValueError('Manifest content differs')
    if not set(manifest['arms']) <= set(ARMS): raise ValueError('Use archived entrypoints for GCC/VarNet')
    cfg = manifest['training']; budgets = cfg['validation_accelerations']
    if cfg['epochs'] < 1 or cfg['patience'] < 1 or not set(budgets) <= set(manifest['train_accelerations']):
        raise ValueError('Invalid training configuration')
    for arm in manifest['arms']:
        execution = cfg['execution'][arm]; micro = execution['microbatch']
        if micro < 1 or cfg['effective_batch'] % micro or execution['checkpoint'] not in ('off', 'full'):
            raise ValueError('Invalid microbatch/checkpoint configuration')
    cases = {case['exam_id']: case for case in manifest['cohort']}
    if len(cases) != len(manifest['cohort']): raise ValueError('Duplicate examinations')
    for fold, split in manifest['folds'].items():
        groups = [{cases[e]['person_id'] for e in split[role]} for role in ('train', 'val', 'test')]
        if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]): raise ValueError(f'Person leakage in fold {fold}')
        ids = [e for role in ('train', 'val', 'test') for e in split[role]]
        if len(ids) != len(set(ids)) or set(ids) != set(cases): raise ValueError('Incomplete examination partition')


def load_manifest(root):
    manifest = json.loads((Path(root) / 'manifest.json').read_text()); validate_manifest(manifest); return manifest
