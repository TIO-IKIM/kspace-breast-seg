"""LitModel subclass for full-dataset training with AUROC and positive-only Dice."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from src.model.lightning import LitModel


class LitModelFullDataset(LitModel):
    """Extends LitModel with:

    - Per-patient AUROC using predicted positive volume from overlap-voted
      reconstruction (tp + fp from case_stats).
    - Positive-only volume-level Dice (reported alongside the standard all-patient Dice).
    """

    # ---- volume state with detection bookkeeping ----

    @staticmethod
    def _new_vol_state() -> Dict[str, Any]:
        state = LitModel._new_vol_state()
        state["detection_gt"] = []         # per-patient has_lesion labels
        state["detection_scores"] = []     # per-patient detection scores
        return state

    # ---- override finalize to record detection from predicted volume ----

    def _finalize_volume(self, state: Dict[str, Any], write_preds: bool = False) -> None:
        pid = state["pid"]

        super()._finalize_volume(state, write_preds=write_preds)

        # Detection score = predicted positive volume (tp + fp) from the
        # overlap-voted reconstruction.  Positive patients have a cluster of
        # predicted-foreground voxels; negatives have near-zero (specificity ~1).
        if pid is not None and int(pid) in state["case_stats"]:
            tp, fp, fn, nref = state["case_stats"][int(pid)]
            has_lesion = 1 if nref > 0 else 0
            detect_score = float(tp + fp)
            state["detection_gt"].append(has_lesion)
            state["detection_scores"].append(detect_score)

    # ---- epoch end: skip patch-level Dice, compute AUROC and positive-only Dice ----

    def on_validation_epoch_end(self) -> None:
        # Skip parent's patch-level Dice (val_avg_dice, val_dice_class_1) —
        # only report volume-level metrics.
        self.val_dice_metric.reset()

        self._finalize_volume(self._val_vol_state)
        if self._val_vol_state["case_stats"]:
            vol_mean = self._compute_vol_dice_mean(self._val_vol_state["case_stats"])
            self.log("val_vol_dice_class_1", vol_mean, sync_dist=True)

        self._log_full_metrics(self._val_vol_state, prefix="val")

    def on_test_epoch_end(self) -> None:
        self._finalize_volume(self._test_vol_state, write_preds=True)
        if self._test_vol_state["case_stats"]:
            case_mean = self._compute_vol_dice_mean(self._test_vol_state["case_stats"])
            self.log("test_dice_class_1", case_mean, sync_dist=True)

        self._log_full_metrics(self._test_vol_state, prefix="test")

    def _log_full_metrics(self, state: Dict[str, Any], prefix: str) -> None:
        case_stats = state.get("case_stats", {})

        # --- Positive-only volume Dice ---
        pos_dice_vals = []
        for pid, (tp, fp, fn, nref) in case_stats.items():
            if nref <= 0:
                continue
            denom = 2 * tp + fp + fn
            if denom == 0:
                pos_dice_vals.append(float("nan"))
            else:
                pos_dice_vals.append((2.0 * tp) / denom)

        if pos_dice_vals:
            arr = np.array(pos_dice_vals, dtype=np.float32)
            pos_dice = float(np.nanmean(arr))
        else:
            pos_dice = 0.0
        self.log(f"{prefix}_vol_dice_class_1_pos", pos_dice, sync_dist=True)

        # --- AUROC ---
        gt = state.get("detection_gt", [])
        scores = state.get("detection_scores", [])

        if len(gt) >= 2 and len(set(gt)) >= 2:
            try:
                from sklearn.metrics import roc_auc_score
                auroc = float(roc_auc_score(gt, scores))
            except Exception:
                auroc = float("nan")
        else:
            auroc = float("nan")
        self.log(f"{prefix}_auroc", auroc, sync_dist=True)
