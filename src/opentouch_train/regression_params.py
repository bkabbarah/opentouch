"""CLI argument parser for the pose-transition regression task.

Deliberately a separate parser from opentouch_train.params.parse_args (the
retrieval pipeline's), not an extension of it: --task-type here is a
checkpoint/logging label ("pose_regression"), NOT a
opentouch_train.data.TASK_ALIASES key, and this pipeline does not touch the
retrieval task-type machinery (parse_task, TASK_ALIASES) at all.
"""

from __future__ import annotations

import argparse


def parse_regression_args(args):
    parser = argparse.ArgumentParser(
        description="OpenTouch pose-transition regression: does tactile predict "
                     "pose deltas beyond what pose alone predicts?"
    )

    parser.add_argument(
        "--train-data", type=str, required=True,
        help="Path to preprocessed HF dataset directory.",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=20,
        help="Window length T. data.py's windowing is untouched by this task; "
             "T=20 is the validated default (OpenTouch window ablation Table 5) "
             "and --horizon-k must stay < this value.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="Seed for the clip-level train/val/test split -- shared with the "
             "retrieval and classification pipelines' --split-seed so splits "
             "are directly comparable.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Model init / training seed.")

    parser.add_argument(
        "--task-type", type=str, default="pose_regression", choices=["pose_regression"],
        help="Checkpoint/logging label only -- not a retrieval TASK_ALIASES key.",
    )
    parser.add_argument(
        "--horizon-k", type=int, required=True, choices=[1, 2, 4, 8, 16],
        help="Predict the pose delta this many frames ahead (33-266ms at 30Hz). "
             "Must be < --sequence-length.",
    )
    parser.add_argument(
        "--motion-threshold", type=float, default=None,
        help="Median fingertip ARTICULATION displacement over k above which a "
             "sample counts as 'moving' for the moving-subset metrics. Default: "
             "computed once from the train split as the 25th percentile of that "
             "same statistic (opentouch_train.regression_data.compute_motion_threshold), "
             "logged, and recorded in checkpoint metadata -- pass this flag to "
             "override with an explicit value instead. Metrics are always also "
             "reported on ALL samples regardless -- see opentouch.regression_metrics.",
    )
    parser.add_argument(
        "--target-mode", type=str, default="articulation_delta",
        choices=["world_delta", "articulation_delta"],
        help="What the model is trained to predict. 'articulation_delta' (default) "
             "removes the wrist's own translation from the target, since ~76%% of "
             "the raw world_delta is whole-hand/arm translation tactile has no "
             "reason to predict (see opentouch.pose_regression module docstring). "
             "'world_delta' trains on the raw pose[t+k]-pose[t]. Eval always "
             "reports metrics for BOTH modes regardless of this flag.",
    )
    parser.add_argument(
        "--pose-only", action="store_true", default=False,
        help="Baseline mode: predict the pose delta from pose at t alone. No "
             "tactile tensor is ever built or passed to the model in this mode "
             "(PoseTransitionRegressor(use_tactile=False) asserts on it). This "
             "is the bar the tactile+pose model must beat.",
    )
    parser.add_argument(
        "--shuffle-tactile", action="store_true", default=False,
        help="Capacity-matched control: architecturally identical tactile+pose "
             "model (exact same param count, hard-asserted), but each sample's "
             "tactile window is drawn from a different window in the same split "
             "(fixed derangement, deterministic given --split-seed -- see "
             "opentouch_train.regression_data._make_derangement), so tactile is "
             "real and present but carries no information about THIS sample's "
             "motion. Tests whether tactile CONTENT matters or just the extra "
             "capacity -- mutually exclusive with --pose-only.",
    )

    parser.add_argument(
        "--tactile-emb-dim", type=int, default=64,
        help="Tactile encoder embedding dim (unused when --pose-only).",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Regression head hidden width.")

    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per GPU.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers per GPU.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument(
        "--warmup", type=float, default=0.05,
        help="Warmup: fraction of total steps if < 1, or absolute step count if >= 1.",
    )
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "const"])
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--skip-scheduler", action="store_true", default=False)
    parser.add_argument("--precision", type=str, default="amp", help="Floating point precision.")

    parser.add_argument("--logs", type=str, default="./logs/")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--log-every-n-steps", type=int, default=100)
    parser.add_argument("--val-frequency", type=int, default=5, help="Eval every N epochs.")
    parser.add_argument("--save-frequency", type=int, default=20)
    parser.add_argument("--save-most-recent", action="store_true", default=True)
    parser.add_argument("--delete-previous-checkpoint", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dist-url", default=None, type=str)
    parser.add_argument("--dist-backend", default=None, type=str)
    parser.add_argument("--no-set-device-rank", action="store_true", default=False)
    parser.add_argument("--use-bn-sync", action="store_true", default=False)
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--wandb-project-name", type=str, default="opentouch-pose-regression")
    parser.add_argument("--wandb-notes", type=str, default="")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--log-local", action="store_true", default=False)

    parsed = parser.parse_args(args)

    if parsed.horizon_k >= parsed.sequence_length:
        raise ValueError(
            f"--horizon-k ({parsed.horizon_k}) must be < --sequence-length "
            f"({parsed.sequence_length}) so t+k stays inside the window"
        )
    if parsed.motion_threshold is not None and parsed.motion_threshold < 0:
        raise ValueError(f"--motion-threshold must be >= 0, got {parsed.motion_threshold}")
    if parsed.pose_only and parsed.shuffle_tactile:
        raise ValueError(
            "--pose-only and --shuffle-tactile are mutually exclusive modes: "
            "pose-only has no tactile at all, shuffle-tactile has real-but-"
            "mispaired tactile. Pick one."
        )

    return parsed
