"""Native-measurement sensitivity for the final phase-referenced hybrid."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PREP='datasets/fastMRI/native_measurement_gradients.py'
ARM='combined_hybrid'


def initialize(root, parent):
    from src.native_spokes import study
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if (root/'manifest.json').exists(): return
    registry = study.read(parent/'manifest.json')
    dataset = study.read(parent/'fastmri_breast/manifest.json')
    cases = [case for case in dataset['cohort'] if case['has_target_lesion']]
    positive = {case['exam_id'] for case in cases}
    folds = {f:{role:[e for e in split[role] if e in positive] for role in ('train','val','test')}
             for f,split in dataset['folds'].items()}
    checkpoints = {}
    for fold,item in registry['model_sources']['fastmri_breast'][ARM].items():
        checkpoint = parent/item['path']
        import torch
        selected = torch.load(checkpoint,map_location='cpu',weights_only=False)
        checkpoints[fold] = dict(path=str(checkpoint),sha256=item['sha256'],epoch=int(selected['epoch'])+1)
    for name in ('logs','cases','aggregate','profile'): (root/name).mkdir(parents=True,exist_ok=True)
    geometry = Path(registry['fastmri_breast']['cache_base'])/'f100/geometry/R4.npz'
    manifest = dict(parent=str(parent),dataset_manifest=str(parent/'fastmri_breast/manifest.json'),
        dataset_manifest_sha256=sha(parent/'fastmri_breast/manifest.json'),workspace_root=str(ROOT),
        model=ARM,checkpoints=checkpoints,cohort=cases,folds=folds,
        cache=str(Path(registry['fastmri_breast']['cache_base'])/'f100'),training=dataset['training'],
        geometry=str(geometry),geometry_sha256=sha(geometry),acceleration=4,readout_percent=100,
        patch_depth=24,patch_stride=16,patch_selection='Existing evaluation-grid patch containing the most target voxels; earliest tie',
        objective='Mean foreground-minus-background logit over fixed reference lesion voxels in the selected patch',
        calibration='Retained precontrast ESPIRiT, normalization, trajectory and density weights fixed',
        gradient_domain='Complex acquired physical-coil samples before zero padding and centered partition FFT; self phase reference included in model derivative',
        epsilon=[1e-5,3e-5,1e-4,3e-4,1e-3],random_directions=3,
        perturbation='delta_y = epsilon * norm(y) * direction / norm(direction)',
        precision='Float32 networks and complex64 physics; TF32 and autocast disabled',seed=123,
        profile_exam='fastMRI_breast_014',example_exam=registry['figures']['gradient_case'],
        interpretation='Post-hoc local sensitivity with reference-defined ROI; no sampling optimization or segmentation-improvement claim')
    write(root/'manifest.json',manifest)
    study.snapshot_source(root/'source')
    files={str(p.relative_to(root/'source')):sha(p) for p in (root/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write(root/'source_snapshot.json',dict(files=files))


def verify(root):
    root=Path(root);manifest=read(root/'manifest.json')
    if sha(manifest['dataset_manifest']) != manifest['dataset_manifest_sha256']:
        raise ValueError('Final dataset manifest changed')
    return manifest


def aggregate(root):
    from src.native_spokes.gradient_reporting import aggregate as summarize
    root=Path(root)
    summarize(root)
    write(root/'complete.json',dict(complete=True,patients=63,folds=list(range(5)),phase_reference_sigma=8,time=time.time()))


def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)

def signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)

def setup_model(root, m, fold):
    import torch
    from src.native_spokes.models import build

    torch.set_num_threads(min(8, int(os.environ.get('NATIVE_NUM_THREADS', 4))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(m['seed'])
    item = m['checkpoints'][str(fold)]
    if sha(item['path']) != item['sha256']:
        raise ValueError('Frozen checkpoint changed')
    model = build(ARM, {}, m['training'], m['seed'] + fold)
    checkpoint = torch.load(item['path'], map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    return model

def norm(tensor):
    return tensor.abs().square().sum(dtype=__import__('torch').float64).sqrt()

def score(logits, roi):
    return (logits[0, 1] - logits[0, 0])[roi].mean()

def run_case(root, m, model, fold, exam, profile=False):
    import torch
    import pandas as pd
    from src.native_spokes.physics import ifft2c
    from src.native_spokes.training import patch_starts

    output = root / ('profile' if profile else 'cases') / exam
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        r = read(output / 'complete.json')
        if r['manifest_sha256'] != sha(root / 'manifest.json'):
            raise ValueError('Existing case belongs to a different analysis')
        for name, digest in r['outputs'].items():
            if sha(output / name) != digest:
                raise ValueError(f'Existing case artifact changed: {name}')
        return r
    if not profile and (exam not in m['folds'][str(fold)]['test'] or exam in m['folds'][str(fold)]['train'] + m['folds'][str(fold)]['val']):
        raise ValueError('A patient must use its own outer-fold checkpoint')
    begin = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    case = next(c for c in m['cohort'] if c['exam_id'] == exam)
    label_path = Path(m['cache']) / exam / 'label_mask.npy'
    label = np.load(label_path, allow_pickle=False)
    starts = patch_starts(len(label), m['patch_depth'], m['patch_stride'])
    start = max(starts, key=lambda z: int(label[z:z + m['patch_depth']].sum()))
    depth = m['patch_depth']
    target = np.ascontiguousarray(label[start:start + depth] > 0)
    z = int(target.sum(axis=(1, 2)).argmax())
    roi = torch.as_tensor(target, device='cuda')
    with np.load(m['geometry'], allow_pickle=False) as handle:
        geometry = {k: handle[k] for k in handle.files}
    spec = importlib.util.spec_from_file_location('native_measurement_gradient_helper', ROOT / PREP)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    cache = Path(m['cache']) / exam / f"R{m['acceleration']}.h5"
    frontend, acquired, cached, calibration = helper.prepare_frontend(
        case, cache, geometry, start, depth, ROOT)
    values = acquired.detach().requires_grad_(True)
    kspace = frontend(values)
    # Two exact vector-Jacobian products reduce simultaneous CNN/FFT storage.
    input_kspace = kspace.detach().requires_grad_(True)
    logits = model(dict(combined_kspace=input_kspace))
    baseline = float(score(logits, roi).detach())
    probability = logits.softmax(dim=1)[0, 1].detach()
    input_gradient, = torch.autograd.grad(score(logits, roi), input_kspace)
    del logits, input_kspace
    gradient, = torch.autograd.grad(kspace, values, input_gradient)
    kspace = kspace.detach()
    del values, input_gradient
    if not torch.isfinite(gradient).all() or float(norm(gradient)) <= 0:
        raise ValueError('Missing or nonfinite native measurement gradient')
    signal_norm = float(norm(acquired))
    gradient_norm = float(norm(gradient))
    direction = gradient * (signal_norm / gradient_norm)
    predicted_slope = signal_norm * gradient_norm
    with torch.no_grad():
        cached_logits = model(dict(combined_kspace=cached))
        cached_score = float(score(cached_logits, roi))
        cached_probability = cached_logits.softmax(dim=1)[0, 1]
        probability_error = float((cached_probability - probability).abs().max())
        # ESPIRiT replay differences can be amplified by phase referencing.
        # Record the discrepancy; the gradient and its finite-difference check
        # are both evaluated at the same retained-data frontend below.
        delta_k = frontend(direction)
        # Check the actual native-data forward path against the linear shortcut.
        h = m['epsilon'][2]
        direct = frontend(acquired + h * direction)
        linear_error = float((direct - (kspace + h * delta_k)).norm() / direct.norm())
        if linear_error > 1e-4:
            raise ValueError(f'Fixed-calibration linearity failed: {linear_error}')
    del cached_logits, cached_probability, direct, cached
    sensitivity = gradient.abs().square().sum(dim=(1, 2)).sqrt().cpu().numpy()
    arrays = dict(image=ifft2c(kspace)[0, z].abs().cpu().numpy(),
                  probability=probability[z].cpu().numpy(), label=target[z].astype(np.uint8),
                  sensitivity=sensitivity, trajectory=geometry['trajectory'],
                  selected_indices=geometry['selected_indices'])
    rows = [dict(direction='baseline', random_index=-1, epsilon=0., signed_epsilon=0.,
                 score=baseline, score_change=0., predicted_change=0., relative_norm=0.)]
    for sign, name in ((1, 'gradient_increase'), (-1, 'gradient_decrease')):
        for epsilon in m['epsilon']:
            with torch.no_grad():
                perturbed = model(dict(combined_kspace=kspace + sign * epsilon * delta_k))
                value = float(score(perturbed, roi))
                if epsilon == max(m['epsilon']):
                    arrays[name] = perturbed.softmax(dim=1)[0, 1, z].cpu().numpy()
            rows.append(dict(direction=name, random_index=-1, epsilon=epsilon,
                             signed_epsilon=sign * epsilon, score=value, score_change=value - baseline,
                             predicted_change=sign * epsilon * predicted_slope,
                             relative_norm=epsilon * float(norm(direction)) / signal_norm))
    del perturbed, direction
    # The random perturbations have the same complex L2 norm over precisely the
    # same acquired samples. They are not acquisition masks or added spokes.
    generator = torch.Generator(device='cuda').manual_seed(m['seed'] + int(exam.rsplit('_', 1)[1]))
    for random_index in range(m['random_directions']):
        random = torch.randn(acquired.shape, dtype=torch.complex64, device='cuda', generator=generator)
        random *= signal_norm / float(norm(random))
        slope = float((gradient.conj() * random).real.sum(dtype=torch.float64))
        with torch.no_grad():
            random_k = frontend(random)
            for epsilon in m['epsilon']:
                perturbed = model(dict(combined_kspace=kspace + epsilon * random_k))
                value = float(score(perturbed, roi))
                rows.append(dict(direction='random', random_index=random_index, epsilon=epsilon,
                                 signed_epsilon=epsilon, score=value, score_change=value - baseline,
                                 predicted_change=epsilon * slope,
                                 relative_norm=epsilon * float(norm(random)) / signal_norm))
            # Symmetric finite difference in a random direction, in addition to
            # the much larger gradient-aligned directional derivative.
            epsilon = m['epsilon'][2]
            perturbed = model(dict(combined_kspace=kspace - epsilon * random_k))
            value = float(score(perturbed, roi))
            rows.append(dict(direction='random_negative_check', random_index=random_index, epsilon=epsilon,
                             signed_epsilon=-epsilon, score=value, score_change=value - baseline,
                             predicted_change=-epsilon * slope,
                             relative_norm=epsilon * float(norm(random)) / signal_norm))
        del random, random_k, perturbed
    del gradient
    table = pd.DataFrame(rows)
    table.insert(0, 'exam_id', exam)
    table.insert(1, 'person_id', case['person_id'])
    table.insert(2, 'fold', fold)
    if not np.isfinite(table.select_dtypes(include='number').to_numpy()).all():
        raise ValueError('Nonfinite perturbation result')
    positive = table.loc[table.direction.eq('gradient_increase')].set_index('epsilon')
    negative = table.loc[table.direction.eq('gradient_decrease')].set_index('epsilon')
    finite_difference = (positive.score - negative.score) / (2 * positive.index.to_numpy())
    relative_errors = (finite_difference - predicted_slope).abs() / predicted_slope
    # Store every step and error; this guard rejects disconnected/wrong gradients,
    # while the paper figure displays all patients without selecting good cases.
    local_checks = []
    if relative_errors.iloc[:3].min() > .05:
        # Self phase referencing can have substantial curvature at the plotted
        # perturbation sizes. Check the local derivative at smaller steps without
        # changing any plotted perturbation or selecting patients by its response.
        with torch.no_grad():
            for h in (3e-6, 1e-6, 3e-7, 1e-7):
                high = float(score(model(dict(combined_kspace=kspace+h*delta_k)),roi))
                low = float(score(model(dict(combined_kspace=kspace-h*delta_k)),roi))
                observed = (high-low)/(2*h)
                error = abs(observed-predicted_slope)/predicted_slope
                local_checks.append(dict(epsilon=h,observed_slope=observed,predicted_slope=predicted_slope,relative_error=error))
                if error <= .05: break
        if min(item['relative_error'] for item in local_checks) > .05:
            raise ValueError(f'Native gradient fails local finite differences: {local_checks}')
    del delta_k
    np.savez_compressed(output / 'example.npz', **arrays)
    table.to_csv(output / 'perturbations.csv', index=False)
    baseline_mask = probability.cpu().numpy() > .5
    tp = int(np.sum(baseline_mask & target))
    patch_dice = 2 * tp / (int(baseline_mask.sum()) + int(target.sum()))
    receipt = dict(complete=True, exam_id=exam, person_id=case['person_id'], fold=fold,
        profile_only=profile, checkpoint=m['checkpoints'][str(fold)],
        manifest_sha256=sha(root / 'manifest.json'), source_snapshot_sha256=sha(root / 'source_snapshot.json'),
        raw=signature(case['raw_path']), label_sha256=sha(label_path), cache=signature(cache),
        start=int(start), depth=depth, display_slice=start + z, lesion_voxels=int(label.sum()),
        roi_voxels=int(target.sum()), baseline_score=baseline, patch_dice=patch_dice,
        gradient_norm=gradient_norm, signal_norm=signal_norm, predicted_slope=predicted_slope,
        cached_score=cached_score, probability_max_error=probability_error,
        frontend_linearity_error=linear_error, **calibration,
        finite_difference=[dict(epsilon=float(h), observed_slope=float(finite_difference.loc[h]),
                               predicted_slope=predicted_slope, relative_error=float(relative_errors.loc[h]))
                           for h in positive.index],
        local_finite_difference_checks=local_checks,
        wall_seconds=time.monotonic() - begin,
        peak_torch_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
        outputs={name: sha(output / name) for name in ('example.npz', 'perturbations.csv')})
    execution_snapshot = ROOT/'execution_source_snapshot.json'
    if execution_snapshot.exists():
        receipt['execution_source_snapshot'] = dict(path=str(execution_snapshot), sha256=sha(execution_snapshot))
    frontend.close()
    if receipt['peak_torch_gib'] >= 39:
        raise RuntimeError('Diagnostic exceeded the worker memory budget')
    write(output / 'complete.json', receipt)
    print(json.dumps(dict(exam_id=exam, complete=True, seconds=receipt['wall_seconds'],
                          peak_gib=receipt['peak_torch_gib'], finite_difference_error=relative_errors.iloc[1])), flush=True)
    return receipt

def run(root, fold, profile=False):
    m = verify(root)
    model = setup_model(root, m, fold)
    exams = [m['profile_exam']] if profile else m['folds'][str(fold)]['test']
    if not profile and not read(root / 'profile' / m['profile_exam'] / 'complete.json')['complete']:
        raise ValueError('Completed native-gradient execution check required')
    for exam in exams:
        if not profile and (Path(root)/'cases'/exam/'complete.json').exists():
            continue
        run_case(root, m, model, fold, exam, profile)
