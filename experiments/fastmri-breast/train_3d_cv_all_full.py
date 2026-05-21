"""Train all 3D models on the FULL fastMRI-breast dataset (300 patients).

Uses stratified CV, conditional Dice (positive-only) + Focal (all samples),
positive patient oversampling, and reports AUROC + pos-only Dice.
"""
import csv
import os
# Import numpy/sklearn BEFORE torch on aarch64 (GH200) to avoid a libgomp /
# OpenBLAS double-registration that aborts with "free(): invalid pointer"
# during static init. See hpo_k2img_noHWdown.py for the working order.
import numpy  # noqa: F401
from sklearn.model_selection import StratifiedKFold
import torch
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping

import sys
sys.path.append("/home/l721f/code/kspace-pred-net")

from src.data.fastmri_3d_full import FastMRIBreast3DKSpaceFullDataModule
from src.model.loss_full import DiceFocalDetectLossMonai, DiceFocalDetectKspaceMSELoss
from src.model.lightning_full import LitModelFullDataset
from src.train.run_3d_experiment import run_cv_3d_experiment


MODELS = [
    "UNet3D",
    "UNet3D_iFFT_mag",
    "UNet3D_iFFT_complex",
    "UNet3D_K2Img_noHWdown",
]


def get_patient_splits(dataset_dir: str, n_folds: int = 5, seed: int = 123):
    """Generate n-fold patient-level splits, stratified by has_lesion."""
    csv_path = os.path.join(dataset_dir, "patient_labels.csv")
    patient_ids = []
    labels = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_ids.append(row["patient_id"])
            labels.append(int(row["has_lesion"]))

    # Only keep patients that actually exist in dataset_dir.
    existing = set(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )
    filtered = [(pid, lab) for pid, lab in zip(patient_ids, labels) if pid in existing]
    patient_ids = [x[0] for x in filtered]
    labels = [x[1] for x in filtered]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in skf.split(patient_ids, labels):
        splits.append({
            "train": [patient_ids[i] for i in train_idx],
            "val": [patient_ids[i] for i in val_idx],
        })
    return splits


def train(model_name: str):
    torch.set_float32_matmul_precision("medium")

    batch_size = 16 # 16 is standard
    epochs = 60
    learning_rate = 3e-4
    eta_min = 1e-6
    patch_depth = 24
    patch_stride = 16 # 8 is standard

    dataset_dir = "/home/l721f/data/fastmri-breast/training_data_full"
    output_dir = f"/home/l721f/code/kspace-pred-net/results/fastmri-breast/3d_cv_{model_name}_full/"

    splits = get_patient_splits(dataset_dir, n_folds=5, seed=123)

    if model_name in ["UNet3D_K2Img", "UNet3D_K2Img_noHWdown"]:
        hidden_factor = 20
        mid_channels = 4
    else:
        hidden_factor = 24
        mid_channels = None

    depth = 3
    depth_k = 2

    # baseline augmentation config from hpo_aug_robustness.py:
    # augment_prob=0.3 with train accels {2x,4x} sampled via RandomMaskFunc.
    augment_prob = 0.3
    augment_accel_specs = [
        (2, 0.04),
        (4, 0.08),
    ]
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
        augment_accel_specs=augment_accel_specs,
        patient_oversample_factor=2,
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
            monitor="val_vol_dice_class_1_pos",
            patience=15,
            mode="max",
        )],
    )

    run_cv_3d_experiment(
        datamodule_cls=FastMRIBreast3DKSpaceFullDataModule,
        datamodule_kwargs=dm_kwargs,
        model_name=model_class_name,
        lightning_module_cls=LitModelFullDataset,
        build_config=build_config,
        splits=splits,
        output_dir_base=output_dir,
        run_name_base=f"{model_name}-3d-cv-full",
        project_name="kspace-pred-net-fastmri-breast-cv-full",
        trainer_kwargs=trainer_kwargs,
        checkpoint_monitor="val_vol_dice_class_1_pos",
        seed=123,
    )


if __name__ == "__main__":
    for model in MODELS:
        print(f"\n{'='*60}\nTraining {model}\n{'='*60}")
        train(model)
