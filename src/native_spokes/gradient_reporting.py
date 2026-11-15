"""Verified patient aggregation and UltraPlot rendering of native gradients."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(root):
    root = Path(root)
    manifest = read(root / 'manifest.json')
    expected = {exam: int(fold) for fold, split in manifest['folds'].items() for exam in split['test']}
    if len(expected) != 63:
        raise ValueError('Expected exactly 63 outer-fold patients')
    missing = [exam for exam in expected if not (root / 'cases' / exam / 'complete.json').is_file()]
    if missing:
        profile = root / 'profile' / manifest['profile_exam'] / 'complete.json'
        status = 'The execution check is complete, but' if profile.is_file() else 'The execution check and'
        raise RuntimeError(
            f'{status} {len(missing)} of {len(expected)} patient results are missing. '
            'The execution check alone does not generate the cohort figure. '
            'If the cohort is already submitted, let it finish. Otherwise run '
            'experiments/run_final_study.py resume-gradients after the check succeeds; '
            'that queues the patient analysis and dependent figure rendering. '
            f'First missing patient: {missing[0]}.')
    recovery_path = root / 'calibration_recovery/manifest.json'
    recovery = read(recovery_path) if recovery_path.exists() else None
    if recovery is not None:
        if (recovery['base_source_snapshot_sha256'] != sha(root / 'source_snapshot.json')
                or recovery['analysis_manifest_sha256'] != sha(root / 'manifest.json')):
            raise ValueError('Calibration recovery refers to a different analysis')
        for name, digest in recovery['files'].items():
            if sha(root / 'calibration_recovery/source' / name) != digest:
                raise ValueError('Calibration recovery source changed')
    tables, cases, derivatives = [], [], []
    execution_snapshots = set()
    for exam, fold in sorted(expected.items()):
        directory = root / 'cases' / exam
        receipt = read(directory / 'complete.json')
        execution = receipt.get('execution_source_snapshot')
        if execution and execution['sha256'] not in execution_snapshots:
            snapshot = Path(execution['path'])
            if sha(snapshot) != execution['sha256']:
                raise ValueError('Gradient execution source declaration changed')
            for name, fingerprint in read(snapshot)['files'].items():
                if sha(snapshot.parent/name) != fingerprint:
                    raise ValueError('Gradient execution source changed')
            execution_snapshots.add(execution['sha256'])
        if (not receipt['complete'] or receipt['profile_only'] or receipt['fold'] != fold
                or receipt['manifest_sha256'] != sha(root / 'manifest.json')
                or receipt['source_snapshot_sha256'] != sha(root / 'source_snapshot.json')
                or receipt['checkpoint'] != manifest['checkpoints'][str(fold)]):
            raise ValueError(f'Invalid patient/checkpoint/source pairing: {exam}')
        if 'calibration_recovery' in receipt:
            entry = receipt['calibration_recovery']
            if (recovery is None or entry['manifest_sha256'] != sha(recovery_path)
                    or entry['files'] != recovery['files'] or exam not in recovery['missing'][str(fold)]):
                raise ValueError('Invalid calibration recovery provenance')
        for name, digest in receipt['outputs'].items():
            if sha(directory / name) != digest:
                raise ValueError(f'Case artifact changed: {exam}/{name}')
        table = pd.read_csv(directory / 'perturbations.csv')
        keys = ['direction', 'random_index', 'epsilon']
        n_expected = 1 + (2 + manifest['random_directions']) * len(manifest['epsilon']) + manifest['random_directions']
        if (len(table) != n_expected or table.duplicated(keys).any()
                or set(table.exam_id) != {exam} or set(table.fold) != {fold}):
            raise ValueError('Incomplete perturbation grid')
        if not np.allclose(table.relative_norm, table.epsilon, rtol=2e-6, atol=1e-12):
            raise ValueError('Perturbation norms are not matched')
        if not np.isfinite(table.select_dtypes(include='number').to_numpy()).all():
            raise ValueError('Nonfinite plotted values')
        tables.append(table)
        fields = ('exam_id', 'person_id', 'fold', 'start', 'display_slice', 'lesion_voxels', 'roi_voxels',
                  'baseline_score', 'patch_dice', 'gradient_norm', 'signal_norm', 'predicted_slope',
                  'cache_relative_error', 'probability_max_error', 'frontend_linearity_error',
                  'wall_seconds', 'peak_torch_gib')
        cases.append(dict({k: receipt[k] for k in fields},
                          calibration_recovery='calibration_recovery' in receipt))
        derivatives.extend(dict(exam_id=exam, fold=fold, **item) for item in receipt['finite_difference'])
    records = pd.concat(tables, ignore_index=True)
    cases = pd.DataFrame(cases).sort_values('person_id').reset_index(drop=True)
    if cases.person_id.nunique() != len(expected):
        raise ValueError('Each person must contribute exactly once')
    derivatives = pd.DataFrame(derivatives)
    # Average the three random directions within each patient before bootstrapping.
    points = records.loc[records.direction.isin(('gradient_increase', 'gradient_decrease', 'random'))]
    points = points.groupby(['person_id', 'direction', 'epsilon'], as_index=False).agg(
        score_change=('score_change', 'mean'), predicted_change=('predicted_change', 'mean'))
    people = cases.person_id.tolist()
    draws = np.random.default_rng(manifest['seed']).integers(0, len(people), (20_000, len(people)))
    rows = []
    for (direction, epsilon), group in points.groupby(['direction', 'epsilon']):
        if set(group.person_id) != set(people):
            raise ValueError('Incomplete paired patient aggregation')
        values = group.set_index('person_id').loc[people, 'score_change'].to_numpy()
        lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
        rows.append(dict(direction=direction, epsilon=float(epsilon), mean=float(values.mean()),
                         lo=float(lo), hi=float(hi), patients=len(values)))
    summary = pd.DataFrame(rows)
    # Pick the illustrative examination by annotated lesion volume, independent
    # of gradient strength, perturbation response or segmentation performance.
    example = cases.sort_values(['lesion_voxels', 'exam_id']).iloc[len(cases) // 2].exam_id
    destination = root / 'aggregate'
    destination.mkdir(exist_ok=True)
    for name, table in (('records', records), ('cases', cases), ('summary', summary), ('finite_differences', derivatives)):
        table.to_csv(destination / f'{name}.csv', index=False)
    np.savez_compressed(destination / 'patient_bootstrap.npz', draws=draws, person_id=people)
    check_h = manifest['epsilon'][1]
    checks = derivatives.loc[np.isclose(derivatives.epsilon, check_h)]
    result = dict(complete=True, patients=len(cases), folds=sorted(cases.fold.unique().tolist()),
        rows=len(records), example=example, example_selection='Median annotated lesion volume among all 63 patients',
        bootstrap='95% percentile intervals, 20000 shared patient resamples, conditional on frozen models',
        finite_difference_epsilon=check_h,
        finite_difference_median_relative_error=float(checks.relative_error.median()),
        finite_difference_max_relative_error=float(checks.relative_error.max()),
        cache_max_relative_error=float(cases.cache_relative_error.max()),
        probability_max_error=float(cases.probability_max_error.max()),
        peak_torch_gib=float(cases.peak_torch_gib.max()),
        calibration_recovery_cases=int(cases.calibration_recovery.sum()),
        calibration_recovery_manifest_sha256=sha(recovery_path) if recovery is not None else None,
        manifest_sha256=sha(root / 'manifest.json'), source_snapshot_sha256=sha(root / 'source_snapshot.json'),
        output_sha256={name: sha(destination / name) for name in
                       ('records.csv', 'cases.csv', 'summary.csv', 'finite_differences.csv', 'patient_bootstrap.npz')})
    (destination / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return manifest, cases, summary, derivatives, result

