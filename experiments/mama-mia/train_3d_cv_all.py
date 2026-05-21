"""Train all 3D models on the MAMA-MIA dataset.

Mirrors the fastMRI-breast full-dataset recipe (see
`experiments/fastmri-breast/train_3d_cv_all_full.py`): conditional Dice
(positive patches only) + Focal (all patches), cosine LR, same augmentation /
patching / capacity.

MAMA-MIA contains only positive patients, so we use the base `LitModel` (not
`LitModelFullDataset`) and monitor `val_vol_dice_class_1`. With every patient
having `nref > 0`, this equals `val_vol_dice_class_1_pos` numerically — the
`_pos` filter in `LitModelFullDataset` is a no-op here, and AUROC would be
NaN (single-class). Keeping the base module also preserves patch-level
`val_avg_dice` / `val_dice_class_1` logging that the Full subclass drops.
"""
import os
# Import numpy/sklearn BEFORE torch on aarch64 (GH200) to avoid a libgomp /
# OpenBLAS double-registration that aborts with "free(): invalid pointer"
# during static init.
import numpy  # noqa: F401
from sklearn.model_selection import KFold
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping

import sys
sys.path.append("/home/l721f/code/kspace-pred-net")

from src.data.mamamia_3d import MamaMIA3DKSpaceDataModule
from src.model.loss_full import DiceFocalDetectLossMonai, DiceFocalDetectKspaceMSELoss
from src.train.run_3d_experiment import run_cv_3d_experiment


MODELS = [
    # "UNet3D",
    # "UNet3D_iFFT_mag",
    # "UNet3D_iFFT_complex",
    "UNet3D_K2Img_noHWdown",
]


def get_patient_splits(dataset_dir: str, n_folds: int = 5, seed: int = 123):
    patient_ids = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in kf.split(patient_ids):
        splits.append({
            "train": [patient_ids[i] for i in train_idx],
            "val": [patient_ids[i] for i in val_idx],
        })
    return splits


def train(model_name: str):
    torch.set_float32_matmul_precision("medium")

    batch_size = 16
    epochs = 60
    learning_rate = 3e-4
    eta_min = 1e-6
    patch_depth = 24
    patch_stride = 16

    dataset_dir = "/home/l721f/data/mama-mia/syn_slices"
    output_dir = f"/home/l721f/code/kspace-pred-net/results/mama-mia/3d_cv_{model_name}/"

    splits = get_patient_splits(dataset_dir, n_folds=5, seed=123)

    if model_name in ["UNet3D_K2Img", "UNet3D_K2Img_noHWdown"]:
        hidden_factor = 20
        mid_channels = 4
    else:
        hidden_factor = 24
        mid_channels = None

    depth = 3
    depth_k = 2

    # Matches fastmri full recipe. MamaMIA's augment datamodule already
    # hard-codes {(2,0.04),(4,0.08)} with RandomMaskFunc, so no accel_specs
    # kwarg is exposed — the undersampling distribution is the same.
    augment_prob = 0.3
    grad_clip = 1.0

    dm_kwargs = dict(
        batch_size=batch_size,
        dataset_dir=dataset_dir,
        patch_depth=patch_depth,
        patch_stride=patch_stride,
        train_val_ratio=0.8,
        oversample_positives_factor=1,
        train_pos_fraction=0.5,
        augment_fastmri_mask_prob=augment_prob,
    )

    def build_config(dm):
        c_in_k, v_in, d_k, h_k, w_k = dm.input_shape

        if model_name == "UNet3D_iFFT_mag":
            input_shape = (c_in_k, 1, d_k, h_k, w_k)
            output_shape = (2, 1, d_k, h_k, w_k)
        elif model_name == "UNet3D_iFFT_complex":
            input_shape = (c_in_k, 2, d_k, h_k, w_k)
            output_shape = (2, 1, d_k, h_k, w_k)
        elif model_name == "UNet3D":
            input_shape = dm.input_shape
            output_shape = (1, v_in, d_k, h_k, w_k)
        else:
            input_shape = dm.input_shape
            output_shape = (2, 1, d_k, h_k, w_k)

        if model_name == "UNet3D":
            criterion = DiceFocalDetectKspaceMSELoss(
                include_background=False,
                dice_weight=0.7,
                focal_weight=0.3,
                gamma=1.5,
                kspace_mse_weight=0.5,
                class_weights=[0.05, 0.95],
            )
        else:
            criterion = DiceFocalDetectLossMonai(
                include_background=False,
                lambda_dice=0.7,
                lambda_focal=0.3,
                gamma=1.5,
                class_weights=[0.05, 0.95],
            )

        config = {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "input_domain": dm.input_domain,
            "label_domain": dm.label_domain,
            "epochs": epochs,
            "lr": learning_rate,
            "optimizer_class": optim.AdamW,
            "scheduler_class": optim.lr_scheduler.CosineAnnealingLR,
            "scheduler_kwargs": {
                "T_max": epochs,
                "eta_min": eta_min,
            },
            "criterion": criterion,
            "hidden_factor": hidden_factor,
            "depth": depth,
        }

        if mid_channels is not None:
            config["mid_channels"] = mid_channels
        if model_name in ["UNet3D_K2Img", "UNet3D_K2Img_noHWdown"]:
            config["up_kernel_size_img"] = (3, 3, 3)
            config["depth_k"] = depth_k
            if model_name == "UNet3D_K2Img_noHWdown":
                config["strides_k"] = tuple((2, 1, 1) for _ in range(depth_k))
                config["up_kernel_size_k"] = (3, 1, 1)

        return config

    model_class_name = {
        "UNet3D": "UNet3D",
        "UNet3D_iFFT_mag": "UNet3D_iFFT",
        "UNet3D_iFFT_complex": "UNet3D_iFFT",
        "UNet3D_K2Img": "UNet3D_K2Img",
        "UNet3D_K2Img_noHWdown": "UNet3D_K2Img",
    }[model_name]

    trainer_kwargs = dict(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy="auto",
        precision="bf16-mixed",
        log_every_n_steps=10,
        gradient_clip_val=grad_clip,
        limit_val_batches=1.0,
        callbacks=[EarlyStopping(
            monitor="val_vol_dice_class_1",
            patience=15,
            mode="max",
        )],
    )

    run_cv_3d_experiment(
        datamodule_cls=MamaMIA3DKSpaceDataModule,
        datamodule_kwargs=dm_kwargs,
        model_name=model_class_name,
        build_config=build_config,
        splits=splits,
        output_dir_base=output_dir,
        run_name_base=f"{model_name}-3d-cv",
        project_name="kspace-pred-net-mama-mia",
        trainer_kwargs=trainer_kwargs,
        checkpoint_monitor="val_vol_dice_class_1",
        seed=123,
    )


if __name__ == "__main__":
    for model in MODELS:
        print(f"\n{'='*60}\nTraining {model}\n{'='*60}")
        train(model)
