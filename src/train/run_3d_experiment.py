from __future__ import annotations

import copy
import gc
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.model.lightning import LitModel
from src.utils.visualization import PositiveSliceVisualizer

def _maybe_patient_map(dm) -> Optional[Dict[int, str]]:
    pid_to_idx = getattr(dm, "patient_id_to_index", None)
    if isinstance(pid_to_idx, dict):
        return {v: k for k, v in pid_to_idx.items()}
    return None


def _to_py(v: Any) -> Any:
    if hasattr(v, "item"):
        return v.item()
    return v


def _metrics_to_py(d: Any) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in d.items():
        vv = _to_py(v)
        if vv is None or isinstance(vv, (int, float, str, bool)):
            out[str(k)] = vv
        else:
            out[str(k)] = str(vv)
    return out


def _build_extra_checkpoints(
    ckpt_dir: str,
    extra_checkpoint_monitors: Optional[List[Dict[str, str]]],
) -> List[ModelCheckpoint]:
    """Create additional ModelCheckpoint callbacks from a list of monitor specs.

    Each spec is a dict with keys: monitor, mode, filename_prefix.
    """
    if not extra_checkpoint_monitors:
        return []
    cbs = []
    for spec in extra_checkpoint_monitors:
        mon = spec["monitor"]
        mode = spec.get("mode", "max")
        prefix = spec.get("filename_prefix", f"best_{mon}")
        cbs.append(ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{prefix}-{{epoch:02d}}-{{{mon}:.3f}}",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
            monitor=mon,
            mode=mode,
        ))
    return cbs


def run_single_3d_experiment(
    *,
    datamodule_cls: Type[pl.LightningDataModule],
    datamodule_kwargs: Dict[str, Any],
    model_name: str,
    lightning_module_cls: Type[pl.LightningModule] = LitModel,
    build_config: Callable[[Any], Dict[str, Any]],
    output_dir: str,
    run_name: str,
    project_name: str,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
    checkpoint_monitor: str = "val_dice_class_1",
    checkpoint_mode: str = "max",
    extra_checkpoint_monitors: Optional[List[Dict[str, str]]] = None,
    vis_every_n_epochs: int = 2,
    vis_num_slices: int = 6,
    vis_per_patient_limit: int = 1,
    vis_min_pos_voxels: int = 20,
) -> None:
    """Run a single train/test cycle for a 3D experiment."""
    dm = datamodule_cls(**datamodule_kwargs)
    dm.setup()

    predictions_dir = Path(output_dir) / "preds"
    figures_dir = Path(output_dir) / "figures"
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    lit_config = build_config(dm)
    lit_config.setdefault("predictions_dir", str(predictions_dir))
    lit_config.setdefault("patient_index_to_id", _maybe_patient_map(dm))

    model = lightning_module_cls(model_name, lit_config)

    logger = WandbLogger(
        project=project_name,
        name=run_name,
        config={
            "model": model_name,
            "batch_size": getattr(dm, "batch_size", None),
            "epochs": lit_config.get("epochs"),
            "lr": lit_config.get("lr"),
            "step_size": lit_config.get("step_size"),
            "gamma": lit_config.get("gamma"),
            "patch_depth": datamodule_kwargs.get("patch_depth"),
            "patch_stride": datamodule_kwargs.get("patch_stride"),
        },
        log_model=False,
    )

    ckpt_dir = os.path.join(output_dir, model_name, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"best_model-{{epoch:02d}}-{{{checkpoint_monitor}:.3f}}",
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
    )
    extra_ckpts = _build_extra_checkpoints(ckpt_dir, extra_checkpoint_monitors)

    vis_callback = PositiveSliceVisualizer(
        figures_dir=str(figures_dir),
        every_n_epochs=vis_every_n_epochs,
        num_slices=vis_num_slices,
        min_positive_voxels=vis_min_pos_voxels,
        max_batches_to_search=40,
        min_slice_gap=5,
        per_patient_limit=vis_per_patient_limit,
    )

    extra_kwargs = dict(trainer_kwargs or {})
    extra_callbacks = extra_kwargs.pop("callbacks", [])
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, vis_callback] + extra_ckpts + list(extra_callbacks),
        **extra_kwargs,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm, ckpt_path="best", weights_only=False)


def run_single_3d_experiment_hp(
    *,
    datamodule_cls: Type[pl.LightningDataModule],
    datamodule_kwargs: Dict[str, Any],
    model_name: str,
    lightning_module_cls: Type[pl.LightningModule] = LitModel,
    build_config: Callable[[Any], Dict[str, Any]],
    output_dir: str,
    run_name: str,
    project_name: str,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
    checkpoint_monitor: str = "val_dice_class_1",
    checkpoint_mode: str = "max",
    vis_every_n_epochs: int = 2,
    vis_num_slices: int = 6,
    vis_per_patient_limit: int = 1,
    vis_min_pos_voxels: int = 20,
) -> Dict[str, Any]:
    """Like run_single_3d_experiment, but returns a compact summary for HP optimization."""
    dm = datamodule_cls(**datamodule_kwargs)
    dm.setup()

    predictions_dir = Path(output_dir) / "preds"
    figures_dir = Path(output_dir) / "figures"
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    lit_config = build_config(dm)
    lit_config.setdefault("predictions_dir", str(predictions_dir))
    lit_config.setdefault("patient_index_to_id", _maybe_patient_map(dm))

    model = lightning_module_cls(model_name, lit_config)

    logger = WandbLogger(
        project=project_name,
        name=run_name,
        config={
            "model": model_name,
            "batch_size": getattr(dm, "batch_size", None),
            "epochs": lit_config.get("epochs"),
            "lr": lit_config.get("lr"),
            "step_size": lit_config.get("step_size"),
            "gamma": lit_config.get("gamma"),
            "patch_depth": datamodule_kwargs.get("patch_depth"),
            "patch_stride": datamodule_kwargs.get("patch_stride"),
        },
        log_model=False,
    )

    ckpt_dir = os.path.join(output_dir, model_name, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"best_model-{{epoch:02d}}-{{{checkpoint_monitor}:.3f}}",
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
    )

    vis_callback = PositiveSliceVisualizer(
        figures_dir=str(figures_dir),
        every_n_epochs=vis_every_n_epochs,
        num_slices=vis_num_slices,
        min_positive_voxels=vis_min_pos_voxels,
        max_batches_to_search=40,
        min_slice_gap=5,
        per_patient_limit=vis_per_patient_limit,
    )

    extra_kwargs = dict(trainer_kwargs or {})
    extra_callbacks = extra_kwargs.pop("callbacks", [])
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, vis_callback] + list(extra_callbacks),
        **extra_kwargs,
    )

    trainer.fit(model, datamodule=dm)
    test_results = trainer.test(datamodule=dm, ckpt_path="best", weights_only=False)

    exp = getattr(logger, "experiment", None)
    best_score = _to_py(getattr(checkpoint_callback, "best_model_score", None))
    best_path = getattr(checkpoint_callback, "best_model_path", None)

    test_metrics: Dict[str, Any] = {}
    if isinstance(test_results, list) and len(test_results) > 0 and isinstance(test_results[0], dict):
        test_metrics = _metrics_to_py(test_results[0])

    return {
        "run_name": run_name,
        "project_name": project_name,
        "output_dir": output_dir,
        "model_name": model_name,
        "checkpoint_monitor": checkpoint_monitor,
        "checkpoint_mode": checkpoint_mode,
        "best_model_score": best_score,
        "best_model_path": best_path,
        "test_metrics": test_metrics,
        "wandb_run_id": getattr(exp, "id", None),
        "wandb_name": getattr(exp, "name", None),
        "wandb_url": getattr(exp, "url", None),
    }


def run_cv_3d_experiment(
    *,
    datamodule_cls: Type[pl.LightningDataModule],
    datamodule_kwargs: Dict[str, Any],
    model_name: str,
    lightning_module_cls: Type[pl.LightningModule] = LitModel,
    build_config: Callable[[Any], Dict[str, Any]],
    splits: List[Dict[str, List[str]]],
    output_dir_base: str,
    run_name_base: str,
    project_name: str,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
    checkpoint_monitor: str = "val_dice_class_1",
    checkpoint_mode: str = "max",
    extra_checkpoint_monitors: Optional[List[Dict[str, str]]] = None,
    vis_every_n_epochs: int = 2,
    vis_num_slices: int = 6,
    vis_per_patient_limit: int = 1,
    vis_min_pos_voxels: int = 20,
    seed: Optional[int] = None,
) -> None:
    """Run patient-level CV using predefined patient ID splits.

    When `seed` is provided, each fold is seeded with `seed + fold_idx` before
    datamodule and model construction so weight init, data shuffling, and
    worker state are deterministic and shared across models trained on the
    same fold.
    """
    for fold_idx, split in enumerate(splits, start=1):
        print(f"--- 3D FOLD {fold_idx}/{len(splits)} ---")
        model = trainer = dm = logger = None
        try:
            if seed is not None:
                pl.seed_everything(seed + fold_idx, workers=True)

            fold_dm_kwargs = dict(datamodule_kwargs)
            fold_dm_kwargs["predefined_patient_ids"] = split

            fold_output_dir = os.path.join(output_dir_base, f"fold_{fold_idx}")
            os.makedirs(fold_output_dir, exist_ok=True)

            dm = datamodule_cls(**fold_dm_kwargs)
            dm.setup()

            lit_config = build_config(dm)
            lit_config.setdefault("predictions_dir", os.path.join(fold_output_dir, model_name, "predictions"))
            lit_config.setdefault("patient_index_to_id", _maybe_patient_map(dm))

            model = lightning_module_cls(model_name, lit_config)

            logger = WandbLogger(
                project=project_name,
                name=f"{run_name_base}-fold-{fold_idx}",
                group=run_name_base,
                config={
                    "model": model_name,
                    "batch_size": getattr(dm, "batch_size", None),
                    "epochs": lit_config.get("epochs"),
                    "lr": lit_config.get("lr"),
                    "step_size": lit_config.get("step_size"),
                    "gamma": lit_config.get("gamma"),
                    "patch_depth": fold_dm_kwargs.get("patch_depth"),
                    "patch_stride": fold_dm_kwargs.get("patch_stride"),
                    "fold": fold_idx,
                },
                log_model=False,
                reinit=True,
            )

            ckpt_dir = os.path.join(fold_output_dir, model_name, "checkpoints")
            checkpoint_callback = ModelCheckpoint(
                dirpath=ckpt_dir,
                filename=f"best_model-{{epoch:02d}}-{{{checkpoint_monitor}:.3f}}",
                save_top_k=1,
                save_weights_only=True,
                verbose=True,
                monitor=checkpoint_monitor,
                mode=checkpoint_mode,
            )
            extra_ckpts = _build_extra_checkpoints(ckpt_dir, extra_checkpoint_monitors)

            vis_callback = PositiveSliceVisualizer(
                figures_dir=os.path.join(fold_output_dir, "figures"),
                every_n_epochs=vis_every_n_epochs,
                num_slices=vis_num_slices,
                min_positive_voxels=vis_min_pos_voxels,
                max_batches_to_search=40,
                min_slice_gap=5,
                per_patient_limit=vis_per_patient_limit,
            )

            extra_kwargs = dict(trainer_kwargs or {})
            extra_callbacks = extra_kwargs.pop("callbacks", [])
            # Important: callback instances (e.g. EarlyStopping) keep internal state
            # and Lightning restores callback state from checkpoints. If we reuse the
            # same callback objects across folds, later folds can stop early.
            extra_callbacks = [copy.deepcopy(cb) for cb in extra_callbacks]
            fold_extra_ckpts = [copy.deepcopy(cb) for cb in extra_ckpts]
            trainer = pl.Trainer(
                logger=logger,
                callbacks=[checkpoint_callback, vis_callback] + fold_extra_ckpts + list(extra_callbacks),
                **extra_kwargs,
            )

            print(f"Starting training for fold {fold_idx}...")
            trainer.fit(model, datamodule=dm)
            print(f"Training finished for fold {fold_idx}.")

            print(f"Starting testing for fold {fold_idx}...")
            trainer.test(datamodule=dm, ckpt_path="best", weights_only=False)
            print(f"Testing finished for fold {fold_idx}.")
        except Exception as e:
            print(f"!!! Fold {fold_idx} failed: {e}")
            traceback.print_exc()
        finally:
            try:
                wandb.finish()
            except Exception:
                pass
            # Free GPU memory before next fold
            del model, trainer, dm, logger
            gc.collect()
            torch.cuda.empty_cache()


