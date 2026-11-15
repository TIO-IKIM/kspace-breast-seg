"""Matched patch training and strict-majority full-volume evaluation."""
from __future__ import annotations
import csv
import gc
import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, default_collate
from src.native_spokes.common import ARMS, atomic_json, load_manifest, sha256, source_hashes
from src.native_spokes.paths import cache_root as execution_cache_root
FIELDS = {arm: ('combined_kspace',) for arm in ARMS}


def load_geometry(cache_root, accelerations):
    return {}


def verified_preparation(run_root, manifest):
    path = Path(run_root) / 'preparation_complete.json'
    prepared = json.loads(path.read_text())
    if not prepared.get('complete') or prepared['examinations'] != len(manifest['cohort']) or prepared['conditions_per_exam'] != len(manifest['train_accelerations']):
        raise ValueError('Complete matching preprocessing is required before fitting')
    return {**prepared, 'preparation_hash': sha256(path)}


def build_network(arm, geometry, training=None, seed=123, device='cuda'):
    from src.native_spokes.models import build
    return build(arm, geometry, training, seed, device)


def build_scheduler(optimizer, configuration):
    return CosineThenConstant(optimizer, configuration['cosine_epochs'], configuration['eta_min'])

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patch_starts(depth: int, size=24, stride=16) -> list[int]:
    if depth < size:
        raise ValueError(f'Volume depth {depth} is smaller than patch depth {size}')
    return sorted(set(range(0, depth - size + 1, stride)) | {depth - size})


def load_labels(cache_root: Path, exam: str):
    return np.load(Path(cache_root) / exam / 'label_mask.npy', mmap_mode='r')


class PatchDataset(Dataset):
    def __init__(self, cache_root, exams, arm, accelerations, epoch=None,
                 patch_depth=24, patch_stride=16):
        self.cache_root = Path(cache_root)
        self.arm = arm
        self.patch_depth = patch_depth
        self.records = []
        self.positive = []
        self._files = OrderedDict()
        catalog_path = self.cache_root.parent / f'patches_d{patch_depth}_s{patch_stride}.json'
        catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else None
        for i, exam in enumerate(sorted(exams)):
            if catalog is not None:
                patches = catalog[exam]
            else:
                label = load_labels(self.cache_root, exam)
                patches = [(start, bool(np.any(label[start:start + patch_depth])))
                           for start in patch_starts(len(label), patch_depth, patch_stride)]
            budgets = accelerations if epoch is None else [accelerations[(i + epoch) % len(accelerations)]]
            for accel in budgets:
                for start, positive in patches:
                    self.records.append((exam, int(accel), start))
                    self.positive.append(positive)

    def __len__(self):
        return len(self.records)

    def _file(self, exam, accel):
        key = (exam, accel)
        if key in self._files:
            self._files.move_to_end(key)
        else:
            if len(self._files) >= 4:
                _, handle = self._files.popitem(last=False)
                handle.close()
            self._files[key] = h5py.File(self.cache_root / exam / f'R{accel}.h5', 'r')
        return self._files[key]

    def __getitem__(self, index):
        exam, accel, start = self.records[index]
        end = start + self.patch_depth
        f = self._file(exam, accel)
        batch = {name: torch.from_numpy(np.asarray(f[name][start:end], dtype=np.complex64))
                 for name in FIELDS[self.arm]}
        batch.update(label=torch.from_numpy(np.array(load_labels(self.cache_root, exam)[start:end],
                                                    dtype=np.int64)),
                     acceleration=accel, exam_id=exam, z_start=start)
        return batch

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()


def to_device(batch, device='cuda'):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def loss_function(arm=None):
    if arm == 'native_kspace':
        from src.model.loss_full import DiceFocalDetectKspaceMSELoss
        return DiceFocalDetectKspaceMSELoss(include_background=False, dice_weight=0.7,
            focal_weight=0.3, gamma=1.5, kspace_mse_weight=0.5, class_weights=[0.05, 0.95])
    from src.model.loss_full import DiceFocalDetectLossMonai
    return DiceFocalDetectLossMonai(include_background=False, lambda_dice=0.7,
                                   lambda_focal=0.3, gamma=1.5,
                                   class_weights=[0.05, 0.95])


def mean_patch_loss(criterion, logits, label_mask):
    """Preserve the microbatch-one objective for any positive/negative mixture.

    Calling the criterion once on a mixed batch averages Dice over positives
    only and would upweight them relative to individual patch accumulation.
    """
    if logits.shape[0] != label_mask.shape[0] or logits.shape[0] < 1:
        raise ValueError('Logits and labels must contain the same nonempty batch')
    if logits.shape[0] == 1:
        return criterion(logits, label_mask=label_mask)
    return torch.stack([criterion(logits[i:i + 1], label_mask=label_mask[i:i + 1])
                        for i in range(logits.shape[0])]).mean()


class CosineThenConstant:
    """Match the original cosine for 60 epochs, then hold its final rate."""

    def __init__(self, optimizer, cosine_epochs, eta_min):
        self.optimizer, self.cosine_epochs, self.eta_min = optimizer, int(cosine_epochs), float(eta_min)
        self.cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cosine_epochs, eta_min=self.eta_min)
        self.steps = 0

    def step(self):
        if self.steps < self.cosine_epochs:
            self.cosine.step()
        else:
            for group in self.optimizer.param_groups:
                group['lr'] = self.eta_min
        self.steps += 1

    def get_last_lr(self):
        return [float(group['lr']) for group in self.optimizer.param_groups]

    def state_dict(self):
        return dict(kind='cosine_then_constant', cosine_epochs=self.cosine_epochs, eta_min=self.eta_min,
                    steps=self.steps, cosine=self.cosine.state_dict())

    def load_state_dict(self, state):
        if (state.get('kind') != 'cosine_then_constant' or state['cosine_epochs'] != self.cosine_epochs
            or state['eta_min'] != self.eta_min):
            raise ValueError('Different cosine/constant schedule on resume')
        self.steps = int(state['steps']); self.cosine.load_state_dict(state['cosine'])
        if self.steps >= self.cosine_epochs and any(rate != self.eta_min for rate in self.get_last_lr()):
            raise ValueError('Resumed constant learning rate differs')


def write_rows(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def volume_metrics(prediction, label):
    target = np.asarray(label) > 0
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    denom = 2 * tp + fp + fn
    return dict(dice=2 * tp / denom if denom else 1.0, tp=tp, fp=fp, fn=fn,
                predicted_foreground_voxels=tp + fp,
                zero_overlap=int(tp == 0 and np.any(target)))


def predict_volumes(model, cache_root, exams, accelerations, arm, *, workers=0,
                    batch_size=1, patch_depth=24, patch_stride=16):
    """Yield complete-volume predictions from one prefetched validation pass.

    Patch ordering and strict-majority voting match ``predict_volume``. Only
    one volume's votes remain in memory; the loader persists across all exams
    and accelerations. Workers are spawned to avoid inheriting CUDA state.
    """
    exams, accelerations = list(exams), list(accelerations)
    if not exams or len(set(exams)) != len(exams):
        raise ValueError('Provide distinct, nonempty examinations')
    if not accelerations or len(set(accelerations)) != len(accelerations):
        raise ValueError('Provide distinct, nonempty accelerations')
    if workers < 0:
        raise ValueError('Validation worker count must be nonnegative')
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Validation batch size must be a positive integer')
    dataset = PatchDataset(cache_root, exams, arm, accelerations,
                           patch_depth=patch_depth, patch_stride=patch_stride)
    arguments = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    if workers:
        arguments.update(multiprocessing_context='spawn', prefetch_factor=1,
                         persistent_workers=False)
    loader = DataLoader(dataset, **arguments)
    iterator = None
    was_training = model.training
    model.eval()
    key = None
    count = 0
    try:
        iterator = iter(loader)
        for item in iterator:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                batch = to_device(item)
                logits = model(batch)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f'Nonfinite predictions: {item["exam_id"]}')
                predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            # A batch may cross volume boundaries. Do not retain GPU tensors or
            # an inference/autocast context while yielding a completed volume.
            del batch, logits
            if len(predictions) != len(item['exam_id']):
                raise ValueError('Prediction batch differs from validation metadata')
            for index, pred in enumerate(predictions):
                exam = item['exam_id'][index]
                accel = int(item['acceleration'][index])
                z = int(item['z_start'][index])
                if count >= len(dataset) or (exam, accel, z) != dataset.records[count]:
                    raise ValueError(f'Unexpected validation patch: {(exam, accel, z)}')
                count += 1
                if key != (exam, accel):
                    if key is not None:
                        if not coverage.all():
                            raise ValueError(f'Uncovered slices for {key}')
                        prediction = 2 * votes > coverage
                        yield (*key, prediction, volume_metrics(prediction, label))
                    key = (exam, accel)
                    label = load_labels(cache_root, exam)
                    votes = np.zeros(label.shape, dtype=np.uint8)
                    coverage = np.zeros((len(label), 1, 1), dtype=np.uint8)
                votes[z:z + patch_depth] += pred
                coverage[z:z + patch_depth] += 1
            del predictions, pred, item
        if count != len(dataset):
            raise ValueError(f'Missing {len(dataset) - count} validation patches')
        if key is None or not coverage.all():
            raise ValueError(f'Uncovered slices for {key}')
        prediction = 2 * votes > coverage
        yield (*key, prediction, volume_metrics(prediction, label))
    finally:
        if iterator is not None and hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()
        dataset.close()
        model.train(was_training)


def predict_volume(model, cache_root, exam, accel, arm, patch_depth=24, patch_stride=16):
    predictions = list(predict_volumes(model, cache_root, [exam], [accel], arm,
                                      patch_depth=patch_depth, patch_stride=patch_stride))
    _, _, prediction, metrics = predictions[0]
    return prediction, metrics


def train(run_root: Path, arm: str, fold: int, workers=4, deadline=None, session_epochs=None):
    run_root = Path(run_root)
    manifest = load_manifest(run_root)
    prepared = verified_preparation(run_root, manifest)
    if arm not in ARMS or fold not in manifest['pilot_folds']:
        raise ValueError('Requested cell outside the authorized pilot')
    cell = run_root / arm / f'fold_{fold}'
    cell.mkdir(parents=True, exist_ok=True)
    if (cell / 'training_complete.json').exists():
        done = json.loads((cell / 'training_complete.json').read_text())
        if (done['manifest_hash'] != manifest['manifest_hash'] or done['source_hashes'] != source_hashes()
            or done['preparation_hash'] != prepared['preparation_hash']):
            raise ValueError('Completed cell source/manifest differs')
        return done
    cfg = manifest['training']
    execution = dict(cfg.get('execution', {}).get(arm, {}))
    if (cell / 'execution.json').exists():
        execution.update(json.loads((cell / 'execution.json').read_text()))
    microbatch = execution.get('microbatch', cfg.get('microbatch', 1))
    cache = execution_cache_root(manifest)
    split = manifest['folds'][str(fold)]
    geometry = load_geometry(cache, manifest['train_accelerations'])
    sources = source_hashes()
    profile_path = cell / 'profile.json'
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        if not profile['complete'] or profile['manifest_hash'] != manifest['manifest_hash']:
            raise ValueError('Successful matching profile required before training')
        if profile['source_hashes'] != sources:
            raise ValueError('Sources changed since the successful profile')
    elif cfg.get('profile_required', True):
        raise ValueError('This training configuration requires a profile')
    else:
        profile = dict(estimated_epoch_seconds=cfg.get('estimated_epoch_seconds', 1800))
    seed_all(manifest['seed'] + fold)
    model_configuration = {**cfg, 'execution': {**cfg['execution'], arm: execution}}
    model = build_network(arm, geometry, model_configuration, seed=manifest['seed'] + fold)
    criterion = loss_function(arm).cuda()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['weight_decay']},
                                   {'params': no_decay, 'weight_decay': 0}], lr=cfg['learning_rate'])
    scheduler = build_scheduler(optimizer, cfg)
    atomic_json(cell / 'config.json', dict(arm=arm, fold=fold, training=cfg, execution=execution,
                manifest_hash=manifest['manifest_hash'], source_hashes=sources,
                preparation_hash=prepared['preparation_hash'],
                parameter_count=sum(p.numel() for p in model.parameters())))
    start_epoch, best, stale, history = 0, None, 0, []
    last_path = cell / 'last.pt'
    if last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=False)
        if (saved['manifest_hash'] != manifest['manifest_hash'] or saved['source_hashes'] != sources
            or saved['preparation_hash'] != prepared['preparation_hash']):
            raise ValueError('Resume checkpoint provenance differs')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start_epoch, best, stale, history = saved['epoch'] + 1, saved['best'], saved['stale'], saved['history']
        if best is not None and not np.isfinite(best):
            raise ValueError('Resume checkpoint has nonfinite validation state')
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    for epoch in range(start_epoch, cfg['epochs']):
        if stale >= cfg['patience']:
            break
        longest = max([row['seconds'] for row in history] + [profile.get('estimated_epoch_seconds', 1800)])
        if ((deadline is not None and time.time() + max(120, 1.5 * longest + 60) >= deadline)
                or (session_epochs is not None and epoch - start_epoch >= session_epochs)):
            paused = dict(complete=False, continuation_required=True, epochs_completed=len(history),
                          manifest_hash=manifest['manifest_hash'], reason='execution boundary')
            atomic_json(cell / 'continuation_required.json', paused)
            return paused
        t0 = time.monotonic()
        dataset = PatchDataset(cache, split['train'], arm, manifest['train_accelerations'], epoch,
                               cfg['patch_depth'], cfg['patch_stride'])
        pos = np.asarray(dataset.positive)
        if not pos.any() or pos.all():
            weights = np.ones(len(pos))
        else:
            weights = np.where(pos, 0.5 / pos.sum(), 0.5 / (~pos).sum())
        generator = torch.Generator().manual_seed(manifest['seed'] + fold * 1000 + epoch)
        # A fixed epoch length keeps the epoch-indexed schedule and patience in optimizer steps.
        samples = int(cfg.get('samples_per_epoch', len(dataset)))
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), samples,
                                        replacement=True, generator=generator)
        loader = DataLoader(dataset, batch_size=microbatch, sampler=sampler, num_workers=workers,
                            pin_memory=True, persistent_workers=False)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, samples_seen = 0.0, 0
        for item in loader:
            batch = to_device(item)
            batch_size = batch['label'].shape[0]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch)
                loss = mean_patch_loss(criterion, logits, batch['label'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite train loss: {arm} fold{fold} epoch{epoch}')
            group_start = (samples_seen // cfg['effective_batch']) * cfg['effective_batch']
            group_size = min(cfg['effective_batch'], samples - group_start)
            (loss * batch_size / group_size).backward()
            loss_sum += float(loss.detach()) * batch_size
            samples_seen += batch_size
            if samples_seen % cfg['effective_batch'] == 0 or samples_seen == samples:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        dataset.close()
        training_seconds = time.monotonic() - t0
        validation = []
        validate_now = ((epoch + 1) % cfg.get('validation_every_epochs', 1) == 0
                        or epoch + 1 == cfg['epochs'])
        score, improved = None, False
        validation_started = time.monotonic()
        if validate_now:
            budgets = cfg.get('validation_accelerations', manifest['train_accelerations'])
            for exam, accel, _, metrics in predict_volumes(
                    model, cache, split.get('checkpoint_val', split['val']), budgets, arm,
                    workers=execution.get('validation_workers', cfg.get('validation_workers', 0)),
                    batch_size=execution.get('validation_batch_size', 1),
                    patch_depth=cfg['patch_depth'], patch_stride=cfg['patch_stride']):
                validation.append(dict(exam_id=exam, acceleration=accel, **metrics))
            score = float(np.mean([r['dice'] for r in validation]))
            if not np.isfinite(score):
                raise FloatingPointError(f'Nonfinite validation score: {arm} fold{fold} epoch{epoch}')
            improved = best is None or score > best
            if improved:
                best, stale = score, 0
            else:
                stale += 1
        validation_seconds = time.monotonic() - validation_started if validate_now else 0.0
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=loss_sum / samples_seen, val_dice=score,
                            training_samples=samples_seen, microbatch=microbatch,
                            best_val_dice=best, stale_validation_checks=stale,
                            validated=validate_now, validation_volumes=len(validation),
                            training_seconds=training_seconds, validation_seconds=validation_seconds,
                            seconds=time.monotonic() - t0, lr=scheduler.get_last_lr()[0]))
        write_rows(cell / 'history.csv', history)
        saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     arm=arm, fold=fold, epoch=epoch, best=best, stale=stale, history=history,
                     manifest_hash=manifest['manifest_hash'], source_hashes=sources,
                     preparation_hash=prepared['preparation_hash'],
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        torch.save(saved, cell / 'last.pt.tmp')
        (cell / 'last.pt.tmp').replace(last_path)
        if improved:
            torch.save({k: v for k, v in saved.items() if k not in ('optimizer', 'scheduler', 'history',
                        'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng')}, cell / 'best.pt.tmp')
            (cell / 'best.pt.tmp').replace(cell / 'best.pt')
            write_rows(cell / 'best_inner_validation.csv', validation)
        print(json.dumps(dict(arm=arm, fold=fold, **history[-1]), allow_nan=False), flush=True)
    if best is None or not np.isfinite(best) or not (cell / 'best.pt').exists():
        raise RuntimeError('Training cannot complete without a finite validated best checkpoint')
    complete = dict(complete=True, manifest_hash=manifest['manifest_hash'], source_hashes=sources,
                    preparation_hash=prepared['preparation_hash'],
                    checkpoint_sha256=sha256(cell / 'best.pt'), best_val_dice=best,
                    epochs_completed=len(history))
    atomic_json(cell / 'training_complete.json', complete)
    (cell / 'continuation_required.json').unlink(missing_ok=True)
    return complete
