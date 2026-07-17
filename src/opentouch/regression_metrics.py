"""Metrics for the pose-transition regression task.

Per-joint and fingertip error, NOT aggregate MSE: the 21 keypoints are
retargeted from 7 Rokoko sensors, so many joints are kinematic
interpolations of the sensor-driven ones rather than independent
measurements. An aggregate MSE would be diluted by joints that carry no
independent signal, hiding exactly the effect (or absence of one) this task
exists to measure. Fingertip columns (FINGERTIP_COLUMNS in pose_regression.py)
are reported separately because they carry the most genuine motion signal.

The copy baseline (predict zero delta) is reported explicitly as the sanity
floor every model must beat: predicting exactly zero delta scores exactly
the mean squared displacement of the targets -- see
compute_copy_baseline_metrics and
tests/test_pose_regression.py::test_copy_baseline_equals_mean_squared_displacement.

Motion filtering is a REPORTING split, not a data filter. Many windows have
near-zero motion, where zero-delta is nearly correct and no model can be
distinguished from the copy baseline. compute_regression_metrics always
returns "all_*" metrics over every sample, and additionally returns
"moving_*" metrics over the subset whose median fingertip displacement over
k is >= motion_threshold, when a threshold is given. Nothing is ever
silently dropped from training or from the "all" numbers.

compute_dual_target_metrics is the eval-time report: it scores the SAME raw
model prediction against BOTH target spaces (world_delta and
articulation_delta) regardless of which one the model was trained on, plus
the wrist-translation error on its own. A model trained on
articulation_delta will (correctly) score badly against world_delta targets
-- it was never trained to predict arm translation -- and that bad number is
the honest report, not a bug to paper over. The moving-subset mask is always
derived from ARTICULATION displacement (never raw world-space displacement,
which is ~76% wrist/arm translation at this dataset's scale -- see
pose_regression.py's module docstring) so "moving" means the same set of
samples in both the world_delta view and the articulation_delta view.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from opentouch.pose_regression import FINGERTIP_COLUMNS, NUM_KEYPOINTS, WRIST_INDEX


def per_joint_squared_error(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    """(B,21,3),(B,21,3) -> (21,) mean squared L2 displacement error per joint."""
    if pred_delta.shape != target_delta.shape:
        raise ValueError(f"shape mismatch: {tuple(pred_delta.shape)} vs {tuple(target_delta.shape)}")
    sq_err = (pred_delta - target_delta).pow(2).sum(dim=-1)  # (B,21) squared L2 per joint
    return sq_err.mean(dim=0)  # (21,)


def fingertip_displacement(pose_delta: torch.Tensor) -> torch.Tensor:
    """(B,21,3) -> (B,) median L2 displacement across the 5 fingertip joints.

    Used both as the "is this window moving" motion signal and as an input
    to fingertip error reporting. Median (not mean) over the 5 fingertips so
    one outlier finger doesn't flag an otherwise-static window as moving.
    """
    tip_deltas = pose_delta[:, FINGERTIP_COLUMNS, :]  # (B,5,3)
    tip_dist = tip_deltas.norm(dim=-1)  # (B,5)
    return tip_dist.median(dim=-1).values  # (B,)


def _summarize(pred_delta: torch.Tensor, target_delta: torch.Tensor, prefix: str) -> Dict[str, float]:
    per_joint = per_joint_squared_error(pred_delta, target_delta)  # (21,)
    metrics: Dict[str, float] = {
        f"{prefix}mse_all_joints": per_joint.mean().item(),
        f"{prefix}mse_fingertips": per_joint[list(FINGERTIP_COLUMNS)].mean().item(),
    }
    for j in range(NUM_KEYPOINTS):
        metrics[f"{prefix}mse_joint_{j}"] = per_joint[j].item()
    return metrics


def compute_copy_baseline_metrics(
    target_delta: torch.Tensor, prefix: str = "copy_baseline_",
) -> Dict[str, float]:
    """The zero-delta ("copy the current pose forward") sanity floor.

    No model involved -- this is a pure statistic of the targets. Predicting
    exactly zero delta scores exactly mean(target_delta**2) per joint, since
    (0 - target)**2 == target**2.
    """
    zero_pred = torch.zeros_like(target_delta)
    return _summarize(zero_pred, target_delta, prefix)


def _metrics_given_mask(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    moving_mask: Optional[torch.Tensor],
    motion_threshold: Optional[float],
) -> Dict[str, float]:
    """Shared by compute_regression_metrics (mask derived from its own
    target) and compute_dual_target_metrics (mask derived from articulation
    displacement and reused for both target views)."""
    metrics: Dict[str, float] = {"num_samples": float(pred_delta.shape[0])}
    metrics.update(_summarize(pred_delta, target_delta, prefix="all_"))
    metrics.update(compute_copy_baseline_metrics(target_delta, prefix="all_copy_baseline_"))

    if moving_mask is not None:
        num_moving = int(moving_mask.sum().item())
        metrics["motion_threshold"] = float(motion_threshold)
        metrics["num_moving_samples"] = float(num_moving)
        metrics["moving_fraction"] = num_moving / pred_delta.shape[0]

        if num_moving > 0:
            moving_pred = pred_delta[moving_mask]
            moving_target = target_delta[moving_mask]
            metrics.update(_summarize(moving_pred, moving_target, prefix="moving_"))
            metrics.update(compute_copy_baseline_metrics(moving_target, prefix="moving_copy_baseline_"))

    return metrics


def compute_regression_metrics(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    motion_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Per-joint/fingertip MSE + copy baseline, on ALL samples and, if
    motion_threshold is given, on the "moving" subset (median fingertip
    displacement of target_delta over k >= motion_threshold). Both are
    always computed and returned side by side.

    Single-target-space convenience function (used directly by tests and by
    callers that only care about one delta space at a time). For the
    eval-time report against BOTH world_delta and articulation_delta from
    the same prediction, use compute_dual_target_metrics instead -- that one
    derives the moving mask from articulation displacement specifically,
    not from whichever target_delta happens to be passed here.
    """
    if pred_delta.shape[0] == 0:
        raise ValueError("compute_regression_metrics received an empty batch")

    moving_mask = None
    if motion_threshold is not None:
        displacement = fingertip_displacement(target_delta)  # (B,)
        moving_mask = displacement >= motion_threshold

    return _metrics_given_mask(pred_delta, target_delta, moving_mask, motion_threshold)


def compute_dual_target_metrics(
    pred_delta: torch.Tensor,
    world_delta: torch.Tensor,
    articulation_delta: torch.Tensor,
    motion_threshold: Optional[float] = None,
) -> Dict[str, object]:
    """The eval-time report: {"world": {...}, "articulation": {...},
    "wrist_translation_mse": float}. Scores the SAME raw pred_delta against
    both target spaces regardless of which one the model was trained to
    predict -- see module docstring for why a "bad" score in the untrained
    space is the correct, honest outcome rather than something to hide.

    The moving-subset mask (if motion_threshold is given) is computed ONCE
    from articulation_delta's fingertip displacement and reused for both
    the "world" and "articulation" views, so "moving" refers to the same set
    of samples in both -- never two different subsets silently compared.
    """
    if pred_delta.shape[0] == 0:
        raise ValueError("compute_dual_target_metrics received an empty batch")

    moving_mask = None
    if motion_threshold is not None:
        displacement = fingertip_displacement(articulation_delta)  # (B,)
        moving_mask = displacement >= motion_threshold

    wrist_translation_mse = F.mse_loss(
        pred_delta[:, WRIST_INDEX, :], world_delta[:, WRIST_INDEX, :],
    ).item()

    return {
        "world": _metrics_given_mask(pred_delta, world_delta, moving_mask, motion_threshold),
        "articulation": _metrics_given_mask(pred_delta, articulation_delta, moving_mask, motion_threshold),
        "wrist_translation_mse": wrist_translation_mse,
    }
