"""Standalone evaluation for pose-transition regression checkpoints.

Mirrors opentouch_train/eval.py's _read_checkpoint_meta pattern: checkpoints
saved by regression_main.py carry task_type/horizon_k/target_mode/pose_only/
shuffle_tactile/motion_threshold/tactile_emb_dim/hidden_dim/sequence_length/
split_seed, so this script reconstructs the EXACT PoseTransitionRegressor +
PoseTransitionDataset config the checkpoint was trained with, rather than
guessing defaults that could silently mismatch the checkpoint's architecture
or (for --shuffle-tactile) silently evaluate against a DIFFERENT
pose/tactile pairing than training used.

Always reports metrics for BOTH target spaces (world_delta and
articulation_delta) plus the wrist-translation error, regardless of which
target_mode the checkpoint was trained with -- see
opentouch.regression_metrics.compute_dual_target_metrics.

Usage::
    python -m opentouch_train.regression_eval \
        --checkpoint logs/.../checkpoints/epoch_100.pt \
        --data preprocessed_data/train_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from opentouch.pose_regression import decompose_world_delta, PoseTransitionRegressor
from opentouch.regression_metrics import compute_dual_target_metrics
from opentouch_train.data import VideoTactilePoseDataset
from opentouch_train.regression_data import PoseTransitionDataset, regression_collate_fn

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_REQUIRED_META_FIELDS = (
    "horizon_k", "target_mode", "pose_only", "shuffle_tactile",
    "tactile_emb_dim", "hidden_dim", "sequence_length", "split_seed",
)


def _read_params_file(params_file: Path) -> dict:
    params = {}
    if not params_file.exists():
        return params
    for line in params_file.read_text().splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            params[key.strip()] = value.strip()
    return params


def _read_checkpoint_meta(path) -> dict:
    """Read pose-transition-regression metadata from checkpoint, falling
    back to params.txt in the log dir, exactly like eval.py does for
    task_type/model/tactile_encoder_type.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = {
        "task_type": ckpt.get("task_type"),
        "epoch": ckpt.get("epoch"),
        "horizon_k": ckpt.get("horizon_k"),
        "target_mode": ckpt.get("target_mode"),
        "pose_only": ckpt.get("pose_only"),
        "shuffle_tactile": ckpt.get("shuffle_tactile"),
        "motion_threshold": ckpt.get("motion_threshold"),
        "tactile_emb_dim": ckpt.get("tactile_emb_dim"),
        "hidden_dim": ckpt.get("hidden_dim"),
        "sequence_length": ckpt.get("sequence_length"),
        "split_seed": ckpt.get("split_seed"),
    }

    if any(meta[k] is None for k in _REQUIRED_META_FIELDS):
        params_file = Path(path).resolve().parent.parent / "params.txt"
        params = _read_params_file(params_file)

        def _coerce(key, caster):
            if meta[key] is None and key in params:
                meta[key] = caster(params[key])

        _coerce("task_type", str)
        _coerce("horizon_k", int)
        _coerce("target_mode", str)
        _coerce("pose_only", lambda v: v == "True")
        _coerce("shuffle_tactile", lambda v: v == "True")
        _coerce("motion_threshold", float)
        _coerce("tactile_emb_dim", int)
        _coerce("hidden_dim", int)
        _coerce("sequence_length", int)
        _coerce("split_seed", int)

    missing = [k for k in _REQUIRED_META_FIELDS if meta[k] is None]
    if missing:
        raise ValueError(
            f"Checkpoint '{path}' is missing required pose-regression metadata "
            f"{missing} (checked both the checkpoint dict and params.txt). This "
            "checkpoint cannot be safely restored -- pass the missing values "
            "explicitly via CLI flags if you are certain of them. Note: for "
            "--shuffle-tactile checkpoints, restoring the wrong split_seed "
            "would silently evaluate against a DIFFERENT pose/tactile pairing "
            "than the one trained on."
        )
    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a pose-transition regression checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    p.add_argument("--data", required=True, help="Path to preprocessed HF dataset.")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--horizon-k", type=int, default=None, help="Override (auto-detected from checkpoint).")
    p.add_argument("--motion-threshold", type=float, default=None, help="Override (auto-detected from checkpoint).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-ratio", type=float, default=0.1, help="Must match training.")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Must match training.")
    p.add_argument(
        "--split-seed", type=int, default=None,
        help="Override (auto-detected from checkpoint). Must match training exactly "
             "for --shuffle-tactile checkpoints, or the pose/tactile pairing changes.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None, help="Optional path to save metrics JSON.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    meta = _read_checkpoint_meta(args.checkpoint)

    horizon_k = args.horizon_k if args.horizon_k is not None else meta["horizon_k"]
    motion_threshold = (
        args.motion_threshold if args.motion_threshold is not None else meta["motion_threshold"]
    )
    split_seed = args.split_seed if args.split_seed is not None else meta["split_seed"]
    pose_only = meta["pose_only"]
    shuffle_tactile = meta["shuffle_tactile"]
    target_mode = meta["target_mode"]

    log.info(
        f"Checkpoint: {args.checkpoint}  Epoch: {meta.get('epoch', '?')}  "
        f"pose_only: {pose_only}  shuffle_tactile: {shuffle_tactile}  "
        f"target_mode: {target_mode}  horizon_k: {horizon_k}  "
        f"tactile_emb_dim: {meta['tactile_emb_dim']}  hidden_dim: {meta['hidden_dim']}"
    )

    device = torch.device(args.device)

    model = PoseTransitionRegressor(
        use_tactile=not pose_only,
        tactile_emb_dim=meta["tactile_emb_dim"],
        hidden_dim=meta["hidden_dim"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    if next(iter(state_dict)).startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    # strict=True: an architecture mismatch (e.g. pose_only restored wrong)
    # must raise, not silently evaluate randomly-initialized weights.
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    base_dataset = VideoTactilePoseDataset(
        hf_dataset_path=args.data,
        split=args.split,
        sequence_length=meta["sequence_length"],
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=split_seed,
        include_tactile=not pose_only,
        include_visual=False,
        include_pose=True,
    )
    dataset = PoseTransitionDataset(
        base_dataset, horizon_k, shuffle_tactile=shuffle_tactile, shuffle_seed=split_seed,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        collate_fn=regression_collate_fn,
    )
    log.info(f"Split: {args.split}  samples: {len(dataset)}  batches: {len(dataloader)}")

    all_pred, all_world, all_articulation = [], [], []
    with torch.inference_mode():
        for batch in dataloader:
            pose_t = batch["pose_t"].to(device)
            world_delta = batch["world_delta"].to(device)
            _, articulation_delta = decompose_world_delta(world_delta)
            tactile_pressure = batch["tactile_pressure"].to(device) if not pose_only else None
            pred_delta = model(pose_t, tactile_pressure)
            all_pred.append(pred_delta.float().cpu())
            all_world.append(world_delta.float().cpu())
            all_articulation.append(articulation_delta.float().cpu())

    all_pred_t = torch.cat(all_pred)
    all_world_t = torch.cat(all_world)
    all_articulation_t = torch.cat(all_articulation)
    dual_metrics = compute_dual_target_metrics(
        all_pred_t, all_world_t, all_articulation_t, motion_threshold=motion_threshold,
    )

    print(f"\n{'='*60}")
    print(f"  Checkpoint     : {args.checkpoint}")
    mode_label = "pose-only baseline" if pose_only else ("shuffled-tactile control" if shuffle_tactile else "tactile+pose")
    print(f"  Mode           : {mode_label}")
    print(f"  Trained target : {target_mode}")
    print(f"  Horizon k      : {horizon_k}")
    print(f"  Split          : {args.split}  ({int(dual_metrics['world']['num_samples'])} samples)")
    print(f"  Wrist translation MSE : {dual_metrics['wrist_translation_mse']:.6f}")
    print(f"{'='*60}")

    for space in ("world", "articulation"):
        m = dual_metrics[space]
        marker = "  <-- trained on this" if space == target_mode.replace("_delta", "") else ""
        print(f"\n  [{space}]{marker}")
        print(f"    ALL     mse_all_joints : {m['all_mse_all_joints']:.6f}   "
              f"mse_fingertips : {m['all_mse_fingertips']:.6f}   "
              f"copy_baseline(fingertips) : {m['all_copy_baseline_mse_fingertips']:.6f}")
        if "moving_mse_fingertips" in m:
            print(
                f"    MOVING  ({int(m['num_moving_samples'])}/{int(m['num_samples'])}="
                f"{m['moving_fraction']:.1%}, threshold={m['motion_threshold']})   "
                f"mse_all_joints : {m['moving_mse_all_joints']:.6f}   "
                f"mse_fingertips : {m['moving_mse_fingertips']:.6f}   "
                f"copy_baseline(fingertips) : {m['moving_copy_baseline_mse_fingertips']:.6f}"
            )
    print(f"\n{'='*60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(dual_metrics, f, indent=2)
        log.info(f"Saved metrics to {args.output}")

    return dual_metrics


if __name__ == "__main__":
    main()
