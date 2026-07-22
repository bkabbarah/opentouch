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
        "--causal", dest="causal", action="store_true", default=True,
        help="Tactile embedding for sample (window, t) is computed ONLY from frames "
             "<= t (a fixed-length causal window ending at t, edge-padded at the start "
             "-- see --causal-window and opentouch_train.regression_data's "
             "'CAUSAL TACTILE WINDOW' module docstring section). Default: True. Fixes a "
             "temporal-leakage bug where the encoder saw the entire T-frame window, "
             "including frames in (t, t+horizon_k] that the target world_delta = "
             "pose[t+horizon_k]-pose[t] is computed from.",
    )
    parser.add_argument(
        "--noncausal", dest="causal", action="store_false",
        help="Disable --causal: reproduces the PRE-FIX behavior (full T-frame window "
             "fed to the encoder regardless of t) for measuring the leak's magnitude, "
             "not for new runs.",
    )
    parser.add_argument(
        "--causal-window", type=int, default=None,
        help="Fixed length of the causal tactile window ending at t (frames before the "
             "start of the clip are edge-padded by repeating frame 0). Default: "
             "--sequence-length, matching the length the tactile encoder is trained on. "
             "Ignored when --noncausal.",
    )
    parser.add_argument(
        "--min-history", type=int, default=10,
        help="Exclude sample (window, t) unless it has >= this many REAL (non-padded) "
             "causal frames, i.e. min(t+1, causal_window) >= --min-history -- at small t "
             "the causal window is mostly edge-padding (t=0 is a single static frame "
             "repeated causal_window times), and a near-chance result there is "
             "uninterpretable ('no tactile signal' vs. 'no tactile history was given' look "
             "identical). Ignored when --noncausal. Since real history is capped at "
             "sequence_length - horizon_k, this requires sequence_length >= horizon_k + "
             "min_history (with the default causal_window) or every sample is excluded -- "
             "PoseTransitionDataset raises immediately with that formula if so, rather than "
             "silently building an empty dataset. Pass 1 to effectively disable filtering "
             "(every t has >= 1 real frame, so nothing is excluded).",
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
    if parsed.causal_window is None:
        parsed.causal_window = parsed.sequence_length
    if parsed.causal:
        # Fast-fail formula, mirrors PoseTransitionDataset.__init__'s own
        # check exactly (see regression_data.py's "MIN-HISTORY FILTERING"
        # docstring section) -- catches an infeasible k/sequence_length/
        # min_history combo before any data is even loaded, not just before
        # training starts.
        max_real_history = min(parsed.sequence_length - parsed.horizon_k, parsed.causal_window)
        if max_real_history < parsed.min_history:
            raise ValueError(
                f"--causal with --min-history={parsed.min_history} leaves ZERO valid samples "
                f"for --horizon-k={parsed.horizon_k} --sequence-length={parsed.sequence_length} "
                f"--causal-window={parsed.causal_window}: the longest real (non-padded) causal "
                f"history any t can reach is min(sequence_length-horizon_k, causal_window)="
                f"{max_real_history} < --min-history={parsed.min_history}. With the default "
                "causal_window (== sequence_length), this requires sequence_length >= "
                f"horizon_k + min_history ({parsed.sequence_length} >= {parsed.horizon_k} + "
                f"{parsed.min_history} = {parsed.horizon_k + parsed.min_history}). Increase "
                "--sequence-length, reduce --horizon-k, or lower --min-history."
            )

    return parsed
