"""Reconstruction pretraining and matched frozen/fine-tuned segmentation.

The shared model, augmentation, segmentation loss and trainer are reused.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.native_spokes import training as tr
from src.native_spokes.augmentation import draw_batch, transform_batch
from src.native_spokes.common import atomic_json, sha256, source_hashes
from src.native_spokes.models import build, reference_phase, state_digest
from src.native_spokes.physics import fft2c, ifft2c
from src.native_spokes.study import CONDITIONS, predict_array

ARM = 'combined_hybrid'
BRANCHES = ('frozen', 'finetuned')
VARIANTS = dict(frozen='hybrid_reconstruction_frozen',
                finetuned='hybrid_reconstruction_finetuned')
SOURCE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def execution_sources():
    return {**source_hashes(), **{str(p.relative_to(SOURCE)): sha256(p) for p in (
        SOURCE/'src/two_stage.py', SOURCE/'experiments/run_final_study.py')}}


class ReconstructionPatches(tr.PatchDataset):
    """Pair each retained-data input with R1 in the input's intensity units."""

    def __getitem__(self, index):
        item = super().__getitem__(index)
        exam, accel, start = self.records[index]
        input_scale = float(json.loads(self._file(exam, accel).attrs['metadata_json'])['normalization_scale'])
        reference = self._file(exam, 1)
        reference_scale = float(json.loads(reference.attrs['metadata_json'])['normalization_scale'])
        if min(input_scale, reference_scale) <= 0:
            raise ValueError(f'Nonpositive normalization for {exam}')
        target = np.asarray(reference['combined_kspace'][start:start+self.patch_depth], dtype=np.complex64)
        item['reference_kspace'] = torch.from_numpy(target * (reference_scale/input_scale))
        return item


class ReconstructionModel(nn.Module):
    """Original k-space U-Net, original iFFT, and a 16-parameter linear readout."""

    def __init__(self, cfg, seed, checkpoint='off'):
        super().__init__()
        # Construct both original U-Nets in their original order, then keep only
        # the frontend. This preserves the original frontend initialization.
        original = build(ARM, {}, cfg, seed, device='cpu')
        self.k_unet = original.base.network.k_unet
        self.readout = nn.Conv3d(8, 2, 1, bias=False)
        with torch.no_grad():
            self.readout.weight.zero_()
            for channel in range(4):
                self.readout.weight[0, 2*channel, 0, 0, 0] = .25
                self.readout.weight[1, 2*channel+1, 0, 0, 0] = .25
        self.seed = seed
        self.sigma = cfg['phase_reference_sigma']
        self.checkpoint = checkpoint
        self.register_buffer('augmentation_samples', torch.zeros((), dtype=torch.int64))

    def forward(self, batch):
        with torch.no_grad(), torch.autocast(batch['combined_kspace'].device.type, enabled=False):
            source = batch['combined_kspace'].to(torch.complex64)
            target = batch['reference_kspace'].to(torch.complex64)
            if self.training:
                params = draw_batch(batch, self.seed, int(self.augmentation_samples.item()))
                source = fft2c(transform_batch(ifft2c(source), params))
                target = fft2c(transform_batch(ifft2c(target), params))
                self.augmentation_samples.add_(len(params))
            # Each image uses its OWN smooth phase. The R1 target never enters
            # the input transform or either segmentation model's forward pass.
            source = reference_phase(source, self.sigma)
            target = ifft2c(reference_phase(target, self.sigma))
            source_image = ifft2c(source)
            x = torch.stack((source.real, source.imag), dim=1).unsqueeze(1).contiguous()
        if self.checkpoint == 'full' and self.training:
            from torch.utils.checkpoint import checkpoint
            features = checkpoint(self.k_unet, x, use_reentrant=False)
        else:
            features = self.k_unet(x)
        with torch.autocast(x.device.type, enabled=False):
            image = ifft2c(torch.complex(features[:, :, 0].float(), features[:, :, 1].float()))
            parts = torch.stack((image.real, image.imag), dim=2).flatten(1, 2)
        output = self.readout(parts)
        prediction = torch.complex(output[:, 0].float(), output[:, 1].float())
        return prediction, target, source_image


def fidelity_values(prediction, target, source):
    dims = tuple(range(1, target.ndim))
    energy = target.abs().square().sum(dims).clamp_min(1e-8)
    error = (prediction-target).abs().square().sum(dims)
    magnitude_error = (prediction.abs()-target.abs()).square().sum(dims)
    input_error = (source-target).abs().square().sum(dims)
    return error, energy, magnitude_error, input_error


def reconstruction_validation(model, cache, exams, cfg, microbatch):
    dataset = ReconstructionPatches(cache, exams, ARM, cfg['validation_accelerations'],
        patch_depth=cfg['patch_depth'], patch_stride=cfg['patch_stride'])
    loader = DataLoader(dataset, batch_size=microbatch, num_workers=0, pin_memory=True)
    sums = {}
    model.eval()
    try:
        for item in loader:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                prediction, target, source = model(tr.to_device(item))
                values = torch.stack(fidelity_values(prediction, target, source), dim=1).cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite reconstruction validation')
            for i, exam in enumerate(item['exam_id']):
                key = (exam, int(item['acceleration'][i]))
                sums[key] = sums.get(key, np.zeros(4, dtype=np.float64)) + values[i]
    finally:
        dataset.close()
    rows = []
    for (exam, accel), (error, energy, magnitude, initial) in sorted(sums.items()):
        rows.append(dict(exam_id=exam, acceleration=accel, nmse=error/energy,
            magnitude_nmse=magnitude/energy, input_nmse=initial/energy))
    return rows


def reconstruction_train(root, cache, fold, deadline=None):
    root, cache = Path(root), Path(cache)/'f100'
    manifest = read(root/'manifest.json')
    cfg = manifest['training']
    split = manifest['folds'][str(fold)]
    cell = root/'reconstruction'/f'fold_{fold}'
    cell.mkdir(parents=True, exist_ok=True)
    if (cell/'training_complete.json').exists():
        return read(cell/'training_complete.json')
    execution = read(cell/'execution.json')
    microbatch = execution['microbatch']
    seed = manifest['seed'] + fold
    tr.seed_all(seed)
    model = ReconstructionModel(cfg, seed, execution['checkpoint']).cuda()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in parameters if p.ndim >= 2], weight_decay=cfg['weight_decay']),
        dict(params=[p for p in parameters if p.ndim < 2], weight_decay=0)], lr=cfg['learning_rate'])
    scheduler = tr.build_scheduler(optimizer, cfg)
    sources = execution_sources()
    provenance = dict(manifest_hash=manifest['manifest_hash'], source_hashes=sources)
    atomic_json(cell/'config.json', dict(**provenance, fold=fold, training=cfg,
        execution=execution, objective=manifest['reconstruction'],
        parameter_count=sum(p.numel() for p in parameters)))
    history, best, start = [], None, 0
    if (cell/'last.pt').exists():
        saved = torch.load(cell/'last.pt', map_location='cpu', weights_only=False)
        if any(saved[k] != value for k, value in provenance.items()):
            raise ValueError('Reconstruction resume provenance differs')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best, start = saved['history'], saved['best'], saved['epoch']+1
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    for epoch in range(start, cfg['epochs']):
        longest = max([row['seconds'] for row in history] + [cfg['estimated_epoch_seconds']])
        if deadline and time.time()+max(120, 1.5*longest+60) >= deadline:
            atomic_json(cell/'continuation_required.json', dict(complete=False, epochs_completed=len(history)))
            return dict(complete=False)
        started = time.monotonic()
        dataset = ReconstructionPatches(cache, split['train'], ARM, manifest['train_accelerations'],
            epoch, cfg['patch_depth'], cfg['patch_stride'])
        positive = np.asarray(dataset.positive)
        weights = (np.where(positive, .5/positive.sum(), .5/(~positive).sum())
                   if positive.any() and not positive.all() else np.ones(len(positive)))
        samples = int(cfg.get('samples_per_epoch', len(dataset)))
        generator = torch.Generator().manual_seed(manifest['seed']+fold*1000+epoch)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), samples,
            replacement=True, generator=generator)
        loader = DataLoader(dataset, batch_size=microbatch, sampler=sampler, num_workers=4,
            pin_memory=True, persistent_workers=False)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss_sum, seen = 0., 0
        for item in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction, target, source = model(tr.to_device(item))
                error, energy, _, _ = fidelity_values(prediction, target, source)
                loss = (error/energy).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite reconstruction loss: fold {fold}, epoch {epoch}')
            count = len(error)
            group_start = (seen//cfg['effective_batch'])*cfg['effective_batch']
            group_size = min(cfg['effective_batch'], samples-group_start)
            (loss*count/group_size).backward()
            loss_sum += float(loss.detach())*count; seen += count
            if seen % cfg['effective_batch'] == 0 or seen == samples:
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        dataset.close()
        training_seconds = time.monotonic()-started
        validation, score, improved = [], None, False
        validate_now = (epoch+1) % cfg['validation_every_epochs'] == 0 or epoch+1 == cfg['epochs']
        if validate_now:
            validation = reconstruction_validation(model, cache, split['val'], cfg,
                execution['validation_batch_size'])
            score = float(np.mean([r['nmse'] for r in validation]))
            if not np.isfinite(score):
                raise FloatingPointError('Nonfinite reconstruction score')
            improved = best is None or score < best
            if improved:
                best = score
        scheduler.step()
        history.append(dict(epoch=epoch, train_nmse=loss_sum/seen, val_nmse=score,
            val_input_nmse=float(np.mean([r['input_nmse'] for r in validation])) if validation else None,
            val_magnitude_nmse=float(np.mean([r['magnitude_nmse'] for r in validation])) if validation else None,
            best_val_nmse=best, training_samples=seen, microbatch=microbatch,
            training_seconds=training_seconds, seconds=time.monotonic()-started,
            lr=scheduler.get_last_lr()[0]))
        tr.write_rows(cell/'history.csv', history)
        saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), history=history, best=best, epoch=epoch,
            fold=fold, **provenance, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        torch.save(saved, cell/'last.pt.tmp'); (cell/'last.pt.tmp').replace(cell/'last.pt')
        if improved:
            keep = {k:v for k,v in saved.items() if k not in ('optimizer','scheduler','history',
                'python_rng','numpy_rng','torch_rng','cuda_rng')}
            torch.save(keep, cell/'best.pt.tmp'); (cell/'best.pt.tmp').replace(cell/'best.pt')
            tr.write_rows(cell/'best_inner_validation.csv', validation)
        print(json.dumps(dict(stage='reconstruction', fold=fold, **history[-1]), allow_nan=False), flush=True)
    if best is None or len(history) != cfg['epochs']:
        raise RuntimeError('Reconstruction training is incomplete')
    saved = torch.load(cell/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    receipt = dict(complete=True, fold=fold, epochs_completed=len(history), best_val_nmse=best,
        best_epoch=saved['epoch']+1, checkpoint_sha256=sha256(cell/'best.pt'),
        frontend_state_sha256=state_digest(model.k_unet), **provenance)
    atomic_json(cell/'training_complete.json', receipt)
    (cell/'continuation_required.json').unlink(missing_ok=True)
    return receipt


def segmentation_train(root, fold, branch, deadline=None):
    root = Path(root)
    pretrained = root/'reconstruction'/f'fold_{fold}'
    receipt = read(pretrained/'training_complete.json')
    cell = root/branch/'fastmri_breast'/ARM/f'fold_{fold}'
    reference_hash = receipt['frontend_state_sha256']
    if sha256(pretrained/'best.pt') != receipt['checkpoint_sha256']:
        raise ValueError('Reconstruction checkpoint differs from its receipt')
    initialization = {}

    def factory(arm, geometry, training=None, seed=123, device='cuda'):
        model = build(arm, geometry, training, seed, device)
        saved = torch.load(pretrained/'best.pt', map_location='cpu', weights_only=False)
        frontend = {k.removeprefix('k_unet.'):v for k,v in saved['model'].items() if k.startswith('k_unet.')}
        model.base.network.k_unet.load_state_dict(frontend, strict=True)
        model.base.network.k_unet.requires_grad_(branch != 'frozen')
        initialization.update(frontend_state_sha256=state_digest(model.base.network.k_unet),
            image_head_state_sha256=state_digest(model.base.network.img_unet),
            full_initial_model_sha256=state_digest(model), seed=seed,
            reconstruction_checkpoint_sha256=receipt['checkpoint_sha256'],
            frozen=branch == 'frozen')
        if initialization['frontend_state_sha256'] != reference_hash:
            raise ValueError('Initialization did not preserve the selected pretrained frontend')
        existing = cell/'initialization.json'
        if existing.exists() and read(existing) != initialization:
            raise ValueError('Segmentation initialization changed on resume')
        atomic_json(existing, initialization)
        return model

    original_factory, original_sources = tr.build_network, tr.source_hashes
    tr.build_network, tr.source_hashes = factory, execution_sources
    try:
        result = tr.train(root/branch/'fastmri_breast', ARM, fold, workers=4, deadline=deadline)
    finally:
        tr.build_network, tr.source_hashes = original_factory, original_sources
    if not result['complete']:
        return result
    # One terminal check proves the supposedly fixed frontend did not move.
    saved = torch.load(cell/'best.pt', map_location='cpu', weights_only=False)
    model = build(ARM, {}, read(root/'manifest.json')['training'], device='cpu')
    model.load_state_dict(saved['model'], strict=True)
    final_hash = state_digest(model.base.network.k_unet)
    if branch == 'frozen' and final_hash != reference_hash:
        raise ValueError('Frozen frontend changed during segmentation training')
    result.update(branch=branch, reconstruction_checkpoint_sha256=receipt['checkpoint_sha256'],
        initial_frontend_state_sha256=reference_hash, selected_frontend_state_sha256=final_hash,
        initialization=read(cell/'initialization.json'))
    atomic_json(cell/'training_complete.json', result)
    atomic_json(cell/'stage_complete.json', result)
    return result


def evaluate(root, cache, fold, shard, deadline=None):
    root, cache = Path(root), Path(cache)
    manifest = read(root/'manifest.json')
    directory = root/'evaluation'/f'fold_{fold}'
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory/f'part_{shard}_complete.json'
    if receipt_path.exists():
        return
    models, fits, executions = {}, {}, {}
    for branch in BRANCHES:
        cell = root/branch/'fastmri_breast'/ARM/f'fold_{fold}'
        fits[branch] = read(cell/'stage_complete.json')
        saved = torch.load(cell/'best.pt', map_location='cpu', weights_only=False)
        models[branch] = build(ARM, {}, manifest['training'], seed=manifest['seed']+fold)
        models[branch].load_state_dict(saved['model'], strict=True); models[branch].eval()
        executions[branch] = read(cell/'execution.json')
    left = fits['frozen']['initialization']; right = fits['finetuned']['initialization']
    if any(left[k] != right[k] for k in ('full_initial_model_sha256', 'reconstruction_checkpoint_sha256')):
        raise ValueError('The two segmentation arms did not start identically')
    import pandas as pd
    path = directory/f'part_{shard}.csv'
    rows = pd.read_csv(path).to_dict('records') if path.exists() else []
    done = {(r['exam_id'],r['model_variant'],r['spoke_reduction'],r['readout_percent']) for r in rows}
    cases = {c['exam_id']:c for c in manifest['cohort']}
    for exam in sorted(manifest['folds'][str(fold)]['test'])[shard::4]:
        if deadline and time.time()+600 >= deadline:
            return
        case = cases[exam]
        label = np.load(cache/'labels'/exam/'label_mask.npy', mmap_mode='r')
        for reduction, percent in CONDITIONS:
            if all((exam, VARIANTS[b], reduction, percent) in done for b in BRANCHES):
                continue
            with h5py.File(cache/f'f{percent}'/exam/f'R{reduction}.h5', 'r') as handle:
                values = handle['combined_kspace'][:]
                metadata = json.loads(handle.attrs['metadata_json'])
            for branch in BRANCHES:
                variant = VARIANTS[branch]
                if (exam, variant, reduction, percent) in done:
                    continue
                prediction, metrics = predict_array(models[branch], values, label,
                    executions[branch]['validation_batch_size'])
                if not case['has_target_lesion']:
                    metrics['dice'] = float('nan')
                output = root/'predictions'/branch/f'fold_{fold}'/f'R{reduction}_F{percent}'/f'{exam}.npz'
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.with_suffix('.npz.tmp').open('wb') as stream:
                    np.savez_compressed(stream, mask_packed=np.packbits(prediction.reshape(-1)),
                        shape=np.asarray(prediction.shape))
                output.with_suffix('.npz.tmp').replace(output)
                rows.append(dict(dataset='fastmri_breast', model_variant=variant, branch=branch,
                    fold=fold, exam_id=exam, person_id=case['person_id'],
                    clinical_status=case['clinical_status'], has_target_lesion=case['has_target_lesion'],
                    spoke_reduction=reduction, readout_percent=percent,
                    retained_spokes_per_window=96//reduction, retained_readout_samples=640*percent//100,
                    **metrics, checkpoint_sha256=fits[branch]['checkpoint_sha256'],
                    reconstruction_checkpoint_sha256=fits[branch]['reconstruction_checkpoint_sha256'],
                    manifest_hash=manifest['manifest_hash'], normalization_scale=metadata['normalization_scale'],
                    prediction_path=str(output.relative_to(root))))
                done.add((exam, variant, reduction, percent))
        tr.write_rows(path, rows)
    expected = len(sorted(manifest['folds'][str(fold)]['test'])[shard::4])*len(CONDITIONS)*len(BRANCHES)
    if len(rows) != expected:
        raise ValueError(f'Incomplete evaluation shard: {len(rows)} / {expected}')
    atomic_json(receipt_path, dict(complete=True, fold=fold, shard=shard, rows=len(rows)))


def task_list():
    return ([dict(kind='reconstruct', fold=f) for f in range(5)]
        + [dict(kind='segment', fold=f, branch=b) for f in range(5) for b in BRANCHES]
        + [dict(kind='evaluate', fold=f, shard=s) for f in range(5) for s in range(4)])


def task_name(task):
    return f'{task["kind"]}_{task.get("branch", "both")}_f{task["fold"]}_s{task.get("shard", 0)}'


def task_receipt(root, task):
    root = Path(root); fold = task['fold']
    if task['kind'] == 'reconstruct':
        return root/'reconstruction'/f'fold_{fold}'/'training_complete.json'
    if task['kind'] == 'segment':
        return root/task['branch']/'fastmri_breast'/ARM/f'fold_{fold}'/'stage_complete.json'
    return root/'evaluation'/f'fold_{fold}'/f'part_{task["shard"]}_complete.json'


def ready(root, task):
    if task['kind'] == 'reconstruct':
        return True
    if task['kind'] == 'segment':
        return task_receipt(root, dict(kind='reconstruct', fold=task['fold'])).exists()
    return all(task_receipt(root, dict(kind='segment', fold=task['fold'], branch=b)).exists() for b in BRANCHES)
