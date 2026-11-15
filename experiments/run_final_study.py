#!/usr/bin/env python
"""Local execution against caller-provided prepared data and run manifests."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ('combined_magnitude', 'combined_complex', 'native_kspace', 'combined_hybrid')


def evaluate(args):
    """Evaluate one model using the existing full-volume voting implementation."""
    import numpy as np
    import torch
    from src.native_spokes.common import load_manifest
    from src.native_spokes.models import build
    from src.native_spokes.paths import cache_base
    from src.native_spokes.training import predict_volumes, write_rows

    manifest = load_manifest(args.run_root)
    configuration = manifest['training']
    cell = args.run_root / args.model / f'fold_{args.fold}'
    checkpoint = args.checkpoint or cell / 'best.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build(args.model, {}, configuration, manifest['seed'] + args.fold)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    execution = dict(configuration.get('execution', {}).get(args.model, {}))
    if (cell / 'execution.json').exists():
        execution.update(json.loads((cell / 'execution.json').read_text()))
    accelerations = args.accelerations
    if accelerations is None:
        accelerations = (manifest.get('eval_accelerations', manifest['train_accelerations'])
                         if args.readout_percent == 100 else [1])
    cases = {case['exam_id']: case for case in manifest['cohort']}
    output = args.output or cell / f'evaluation_readout_{args.readout_percent}.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for exam, reduction, prediction, metrics in predict_volumes(
            model, cache_base(manifest) / f'f{args.readout_percent}',
            manifest['folds'][str(args.fold)]['test'], accelerations, args.model,
            workers=args.workers if args.workers is not None else 0,
            batch_size=args.batch_size or execution.get('validation_batch_size', 1),
            patch_depth=configuration['patch_depth'],
            patch_stride=configuration['patch_stride']):
        case = cases[exam]
        if not case['has_target_lesion']:
            metrics['dice'] = float('nan')
        row = dict(exam_id=exam, person_id=case['person_id'],
                   has_target_lesion=case['has_target_lesion'],
                   model_variant=args.model, fold=args.fold,
                   spoke_reduction=reduction, readout_percent=args.readout_percent,
                   **metrics)
        if args.save_masks:
            directory = output.parent / output.stem / f'R{reduction}'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'{exam}.npz'
            np.savez_compressed(path, mask_packed=np.packbits(prediction.reshape(-1)),
                                shape=np.asarray(prediction.shape))
            row['prediction_path'] = str(path)
        rows.append(row)
    write_rows(output, rows)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=(
        'train', 'evaluate', 'reconstruct', 'segment', 'evaluate-pretrained',
        'gradient-init', 'gradient-run', 'gradient-aggregate'))
    parser.add_argument('--run-root', type=Path, required=True,
                        help='Prepared run directory containing manifest.json and required input metadata.')
    parser.add_argument('--cache-base', type=Path,
                        help='Prepared-data directory containing f100/, other readout conditions, and labels/.')
    parser.add_argument('--dataset', choices=('fastmri_breast', 'mama_mia'),
                        default='fastmri_breast')
    parser.add_argument('--model', choices=MODELS, default='combined_hybrid')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--accelerations', type=int, nargs='+')
    parser.add_argument('--readout-percent', type=int,
                        choices=(100, 75, 50, 25, 10, 5, 2), default=100)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--save-masks', action='store_true')
    parser.add_argument('--branch', choices=('frozen', 'finetuned'), default='finetuned')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--parent', type=Path,
                        help='Completed study registry for initializing k-space sensitivity analysis.')
    parser.add_argument('--profile', action='store_true',
                        help='Run the saved sensitivity profiling case before cohort evaluation.')
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    if args.stage in ('reconstruct', 'evaluate-pretrained') and args.cache_base is None:
        parser.error('--cache-base is required for this stage')
    if args.stage == 'gradient-init' and args.parent is None:
        parser.error('--parent is required for gradient-init')

    from src.native_spokes.paths import configure
    configure('local', args.run_root, args.dataset, args.cache_base)
    if args.stage == 'train':
        from src.native_spokes.training import train
        train(args.run_root, args.model, args.fold,
              workers=args.workers if args.workers is not None else 4)
    elif args.stage == 'evaluate':
        evaluate(args)
    elif args.stage in ('reconstruct', 'segment', 'evaluate-pretrained'):
        from src import two_stage
        if args.stage == 'reconstruct':
            two_stage.reconstruction_train(args.run_root, args.cache_base, args.fold)
        elif args.stage == 'segment':
            two_stage.segmentation_train(args.run_root, args.fold, args.branch)
        else:
            two_stage.evaluate(args.run_root, args.cache_base, args.fold, args.shard)
    else:
        from src.native_spokes import gradients
        if args.stage == 'gradient-init':
            gradients.initialize(args.run_root, args.parent.resolve())
        elif args.stage == 'gradient-aggregate':
            gradients.aggregate(args.run_root)
        else:
            fold = args.fold
            if args.profile:
                manifest = gradients.read(args.run_root / 'manifest.json')
                fold = next(int(key) for key, split in manifest['folds'].items()
                            if manifest['profile_exam'] in split['test'])
            gradients.run(args.run_root, fold, args.profile)


if __name__ == '__main__':
    main()
