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

from opentouch.pose_regression import (
    POSE_DIM,
    PoseTransitionRegressor,
    decompose_world_delta,
    grip_aperture_delta,
)
from opentouch.regression_metrics import compute_dual_target_metrics, fingertip_displacement
from opentouch_train.regression_train import _aperture_metrics, _onset_metrics, onset_target_and_mask
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
        "tactile_reduce": ckpt.get("tactile_reduce", "none"),
        "pose_only": ckpt.get("pose_only"),
        "shuffle_tactile": ckpt.get("shuffle_tactile"),
        "motion_threshold": ckpt.get("motion_threshold"),
        "tactile_emb_dim": ckpt.get("tactile_emb_dim"),
        "hidden_dim": ckpt.get("hidden_dim"),
        "sequence_length": ckpt.get("sequence_length"),
        "split_seed": ckpt.get("split_seed"),
        "git_commit": ckpt.get("git_commit"),
        "git_dirty": ckpt.get("git_dirty"),
        "causal": ckpt.get("causal"),
        "causal_window": ckpt.get("causal_window"),
        "min_history": ckpt.get("min_history"),
        "split_group_by": ckpt.get("split_group_by"),
        "tactile_correction_input": ckpt.get("tactile_correction_input"),
        "fusion": ckpt.get("fusion"),
    }

    if any(
        meta[k] is None for k in
        _REQUIRED_META_FIELDS
        + ("git_commit", "git_dirty", "causal", "causal_window",
           "split_group_by", "tactile_correction_input")
    ):
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
        _coerce("git_commit", str)
        _coerce("git_dirty", lambda v: v == "True")
        _coerce("causal", lambda v: v == "True")
        _coerce("causal_window", int)
        _coerce("min_history", lambda v: None if v == "None" else int(v))
        _coerce("split_group_by", str)
        _coerce("tactile_correction_input", str)
        _coerce("fusion", str)

    if meta["fusion"] is None:
        meta["fusion"] = "gate"   # predates the flag; every such run used the gate

    if meta["tactile_correction_input"] is None:
        # Predates the flag; the default at the time was the pose+tactile
        # correction input. Recorded explicitly because it sets the width of
        # tactile_head, so guessing wrong is a load_state_dict size mismatch,
        # not a silent wrong answer -- loud, but still worth stating.
        meta["tactile_correction_input"] = "pose_tactile"

    if meta["split_group_by"] is None:
        # Checkpoints written before split_group_by was recorded were all
        # trained on the clip-level split, because that was the only option
        # the regression pipeline had. Unlike the retrieval side -- where the
        # same silent fallback scored a scene-trained model at 72.09 mAP
        # against a gallery of its own training participants -- this default
        # is a correct reconstruction rather than a guess, but it is stated
        # rather than assumed.
        log.warning(
            f"Checkpoint '{path}' has no 'split_group_by' (predates the field) -- "
            "defaulting to 'clip', which is what the pipeline could only have done "
            "at the time. Pass --split-group-by explicitly to override."
        )
        meta["split_group_by"] = "clip"

    if meta["causal"] is None:
        # Predates the causal-tactile-window fix entirely (checkpoint and
        # params.txt both lack it) -- every such checkpoint was ACTUALLY
        # trained on the leaky pre-fix path (full T-window fed regardless of
        # t), so causal=False is the correct reconstruction, not a guess --
        # same "predates this field" convention as opentouch_train.eval's
        # tactile_encoder_type default.
        log.warning(
            f"Checkpoint '{path}' has no 'causal' field (predates the causal-tactile-window "
            "fix) -- defaulting to causal=False, matching what this checkpoint was ACTUALLY "
            "trained with."
        )
        meta["causal"] = False
    if meta["causal"] and meta["causal_window"] is None:
        log.warning(
            f"Checkpoint '{path}' has causal=True but no 'causal_window' -- defaulting to "
            f"sequence_length={meta['sequence_length']} (the causal fix's own default)."
        )
        meta["causal_window"] = meta["sequence_length"]

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
        "--split-group-by", default=None, choices=["clip", "scene"],
        help="Override the checkpoint's recorded split geometry. Leave unset -- a "
             "scene-trained model evaluated on a clip split sees participants it "
             "trained on, and reports an inflated number that looks plausible.",
    )
    p.add_argument(
        "--split-seed", type=int, default=None,
        help="Override (auto-detected from checkpoint). Must match training exactly "
             "for --shuffle-tactile checkpoints, or the pose/tactile pairing changes.",
    )
    p.add_argument(
        "--causal", dest="causal", action="store_true", default=None,
        help="Override (auto-detected from checkpoint; defaults to False for checkpoints "
             "that predate the causal-tactile-window fix).",
    )
    p.add_argument("--noncausal", dest="causal", action="store_false", help="See --causal.")
    p.add_argument(
        "--causal-window", type=int, default=None, help="Override (auto-detected from checkpoint).",
    )
    p.add_argument(
        "--min-history", type=int, default=None, help="Override (auto-detected from checkpoint).",
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
    causal = args.causal if args.causal is not None else meta["causal"]
    causal_window = args.causal_window if args.causal_window is not None else meta["causal_window"]
    min_history = args.min_history if args.min_history is not None else meta["min_history"]
    split_group_by = args.split_group_by or meta["split_group_by"]
    if args.split_group_by and args.split_group_by != meta["split_group_by"]:
        log.warning(
            "--split-group-by=%s OVERRIDES the checkpoint's %s -- the eval set will "
            "not be the split this model was trained against.",
            args.split_group_by, meta["split_group_by"],
        )
    pose_only = meta["pose_only"]
    shuffle_tactile = meta["shuffle_tactile"]
    target_mode = meta["target_mode"]

    log.info(
        f"Checkpoint: {args.checkpoint}  Epoch: {meta.get('epoch', '?')}  "
        f"pose_only: {pose_only}  shuffle_tactile: {shuffle_tactile}  "
        f"target_mode: {target_mode}  horizon_k: {horizon_k}  "
        f"tactile_emb_dim: {meta['tactile_emb_dim']}  hidden_dim: {meta['hidden_dim']}"
    )
    log.info(
        f"causal: {causal}  causal_window: {causal_window}  min_history: {min_history}"
    )
    log.info(f"split_group_by: {split_group_by}  split_seed: {split_seed}")
    log.info(
        f"git_commit: {meta.get('git_commit', '?')}  git_dirty: {meta.get('git_dirty', '?')}"
    )

    device = torch.device(args.device)

    model = PoseTransitionRegressor(
        use_tactile=not pose_only,
        tactile_emb_dim=meta["tactile_emb_dim"],
        hidden_dim=meta["hidden_dim"],
        tactile_correction_input=meta["tactile_correction_input"],
        fusion=meta["fusion"],
        tactile_reduce=meta.get("tactile_reduce", "none"),
        output_dim=1 if meta["target_mode"] in ("grip_aperture", "motion_onset") else POSE_DIM,
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
        split_group_by=split_group_by,
        include_tactile=not pose_only,
        include_visual=False,
        include_pose=True,
    )
    dataset = PoseTransitionDataset(
        base_dataset, horizon_k, shuffle_tactile=shuffle_tactile, shuffle_seed=split_seed,
        causal=causal, causal_window=causal_window, min_history=min_history,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        collate_fn=regression_collate_fn,
    )
    log.info(f"Split: {args.split}  samples: {len(dataset)}  batches: {len(dataloader)}")

    all_pred, all_world, all_articulation = [], [], []
    all_scalar_target = []
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
            if meta["target_mode"] == "grip_aperture":
                all_scalar_target.append(
                    grip_aperture_delta(pose_t, world_delta).float().cpu())
            elif meta["target_mode"] == "motion_onset":
                t_, m_ = onset_target_and_mask(
                    batch["past_delta"].to(device), batch["past_valid"],
                    world_delta, motion_threshold,
                )
                all_scalar_target.append(torch.stack(
                    [t_.reshape(-1).float().cpu(), m_.reshape(-1).float().cpu()], dim=-1))

    all_pred_t = torch.cat(all_pred)
    all_world_t = torch.cat(all_world)
    all_articulation_t = torch.cat(all_articulation)

    # Scalar targets do not go through the per-joint dual-target machinery at
    # all: the prediction is (B,1), not (B,21,3).
    if meta["target_mode"] == "grip_aperture":
        target = torch.cat(all_scalar_target)
        moving = fingertip_displacement(all_articulation_t) >= motion_threshold
        m = _aperture_metrics(all_pred_t, target, moving)
        print("")
        print(f"  Checkpoint : {args.checkpoint}")
        print(f"  Split      : {args.split}   n={len(all_pred_t)}  moving={int(moving.sum())}")
        print("  R2_vs_zero: %.6f  AUC_sign: %.6f"
              % (m["moving_r2_vs_zero"], m["moving_auc_sign"]))
        if args.output:
            with open(args.output, "w") as fh:
                json.dump({"split": args.split, "target_mode": "grip_aperture", **m}, fh, indent=2)
        return m

    if meta["target_mode"] == "motion_onset":
        stacked = torch.cat(all_scalar_target)
        target, mask = stacked[:, :1], stacked[:, 1] > 0.5
        m = _onset_metrics(all_pred_t, target, mask)
        print("")
        print(f"  Checkpoint : {args.checkpoint}")
        print(f"  Split      : {args.split}   still n={int(m.get('n_still', 0))}")
        print("  base_rate: %.6f  AUC: %.6f" % (m.get("base_rate", float("nan")),
                                                m.get("auc", float("nan"))))
        if args.output:
            with open(args.output, "w") as fh:
                json.dump({"split": args.split, "target_mode": "motion_onset", **m}, fh, indent=2)
        return m

    dual_metrics = compute_dual_target_metrics(
        all_pred_t, all_world_t, all_articulation_t, motion_threshold=motion_threshold,
    )

    print(f"\n{'='*60}")
    print(f"  Checkpoint     : {args.checkpoint}")
    mode_label = "pose-only baseline" if pose_only else ("shuffled-tactile control" if shuffle_tactile else "tactile+pose")
    print(f"  Mode           : {mode_label}")
    print(f"  Trained target : {target_mode}")
    print(f"  Horizon k      : {horizon_k}")
    print(f"  Causal         : {causal}  (window={causal_window}, min_history={min_history})")
    print(f"  Git commit     : {meta.get('git_commit', '?')}  (dirty: {meta.get('git_dirty', '?')})")
    print(f"  Split          : {args.split}  ({int(dual_metrics['world']['num_samples'])} samples)")
    print(f"  Wrist translation MSE : {dual_metrics['wrist_translation_mse']:.6f}")
    if not pose_only:
        gate_value = model.gate.detach().float().mean().item()
        print(f"  Residual-fusion gate  : {gate_value:.6f}  (trains away from 0 only if tactile helps)")
        dual_metrics["gate_value"] = gate_value
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
