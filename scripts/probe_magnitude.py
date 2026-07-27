"""Does touch carry MAGNITUDE information, or only direction?

WHY THIS EXISTS. Everything established so far is a sign result: touch
predicts which way finger articulation is about to go. The abstract claims
direction only and deliberately makes no magnitude claim. But the expensive
open question is whether to rebuild the end-to-end forecasting arm, and that
arm is trained on MSE, which is dominated by magnitude. If touch carries no
magnitude information, the rebuild is very likely to reproduce the existing
null and cost two days to do it.

This answers that question in minutes instead, using frozen features that are
already validated. Same encoders, same causal windowing, same shuffle control,
same sample set as the direction probe. The only change is the readout: ridge
regression onto the articulation delta VECTOR rather than logistic regression
onto its sign.

WHAT A POSITIVE LOOKS LIKE. Adding touch to the pose condition reduces
held-out error, with a clip-clustered interval excluding zero, and beats its
dimensionality-matched shuffled twin. That would motivate the end-to-end
rebuild.

WHAT A NEGATIVE MEANS. Touch's contribution is direction-only. That is a
clean, publishable scope statement, it is consistent with the current
abstract, and it says the end-to-end rebuild would be chasing a signal that
is not in the features. Either way the answer is worth having before spending
GPU-days.

Errors are reported against a predict-zero baseline, the same reference the
end-to-end regression used, so R2 here is directly comparable in spirit to
"beats copy-zero" there.
"""

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from opentouch.articulation_frames import (
    ARTICULATING_JOINTS,
    all_target_variants,
    hand_frame_degenerate,
)
from opentouch.pose_regression import COORD_DIM, NUM_KEYPOINTS, WRIST_INDEX
from opentouch.regression_metrics import fingertip_displacement
from opentouch_train.data import VideoTactilePoseDataset, _load_and_split_dataset
from opentouch_train.regression_data import (
    PoseTransitionDataset,
    _causal_frame_indices,
    _make_derangement,
    compute_motion_threshold,
)
from tactile_direction_probe import (
    encode_causal,
    encode_pose_causal,
    load_pose_encoder,
    load_tactile_encoder,
    min_history_mask,
    standardize,
)
from tactile_subset_probe import cluster_ids_for_windows

# Same lags as probe_rawpose, so the raw-kinematics condition is identical
# across the sign and magnitude analyses.
LAGS = [0, 1, 2, 3, 5, 8, 12, 19]
FINGERTIPS = (4, 8, 12, 16, 20)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--horizon-k", type=int, default=8)
parser.add_argument("--sequence-length", type=int, default=36)
parser.add_argument("--causal-window", type=int, default=20)
parser.add_argument("--min-history", type=int, default=10)
parser.add_argument("--split-seed", type=int, default=42)
parser.add_argument("--split-group-by", default="clip", choices=["clip", "scene"])
parser.add_argument("--eval-split", default="val", choices=["val", "test"])
parser.add_argument("--emb-dim", type=int, default=64)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--n-boot", type=int, default=1000)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--output", required=True)
args = parser.parse_args()
device = torch.device(args.device)

tactile_encoder = load_tactile_encoder(args.checkpoint, args.emb_dim, device)
pose_encoder = load_pose_encoder(args.checkpoint, args.emb_dim, device)
splits = _load_and_split_dataset(args.data, 0.1, 0.1, args.split_seed, args.split_group_by)


def prepare(split_name, threshold):
    base = VideoTactilePoseDataset(
        split=split_name, _preloaded=splits[split_name], hf_dataset_path=args.data,
        sequence_length=args.sequence_length, include_tactile=True,
        include_visual=False, include_pose=True,
    )
    dataset = PoseTransitionDataset(base, args.horizon_k, shuffle_tactile=False, causal=False)
    if threshold is None:
        threshold = compute_motion_threshold(dataset, percentile=25.0)

    pose = dataset._pose
    n_windows = pose.shape[0]
    valid_t = dataset.valid_t_per_window
    window_idx = np.repeat(np.arange(n_windows), valid_t)
    t_values = np.tile(np.arange(valid_t), n_windows)

    keep = min_history_mask(t_values, args.causal_window, args.min_history)
    window_idx, t_values = window_idx[keep], t_values[keep]
    pose_t = pose[:, :valid_t].reshape(-1, NUM_KEYPOINTS, COORD_DIM)[keep]
    pose_future = pose[:, args.horizon_k:].reshape(-1, NUM_KEYPOINTS, COORD_DIM)[keep]

    good = (~hand_frame_degenerate(pose_t)).numpy()
    window_idx, t_values = window_idx[good], t_values[good]
    pose_t, pose_future = pose_t[good], pose_future[good]

    targets = all_target_variants(pose_t, pose_future)
    moving = (
        fingertip_displacement(targets["wrist_translation_removed"]) >= threshold
    ).numpy()

    causal_frame_idx = _causal_frame_indices(valid_t, args.causal_window)
    permutation = _make_derangement(n_windows, seed=args.split_seed)

    windows_t = torch.as_tensor(window_idx)
    lagged = []
    for lag in LAGS:
        frame_idx = np.clip(t_values - lag, 0, None)
        gathered = dataset._pose[windows_t, torch.as_tensor(frame_idx)]
        gathered = gathered - gathered[:, WRIST_INDEX : WRIST_INDEX + 1, :]
        lagged.append(gathered.reshape(gathered.shape[0], -1).numpy())

    prepared = {
        "target": targets["rigid_removed_handframe"].numpy(),
        "moving": moving,
        "pose_raw": np.hstack(lagged),
        "tactile": encode_causal(
            tactile_encoder, dataset._tactile, window_idx, t_values,
            causal_frame_idx, args.batch_size, device,
        ).numpy(),
        "shuffled": encode_causal(
            tactile_encoder, dataset._tactile, permutation[window_idx], t_values,
            causal_frame_idx, args.batch_size, device,
        ).numpy(),
        "pose_emb": encode_pose_causal(
            pose_encoder, dataset._pose, window_idx, t_values,
            causal_frame_idx, args.batch_size, device,
        ).numpy(),
        "clusters": cluster_ids_for_windows(dataset, window_idx, "clip"),
    }
    print(
        "[%s] samples=%d moving=%d" % (split_name, len(window_idx), int(moving.sum())),
        flush=True,
    )
    return prepared, threshold


train, motion_threshold = prepare("train", None)
evaluation, _ = prepare(args.eval_split, motion_threshold)


def flatten_target(target, joints):
    """(N,21,3) -> (N, len(joints)*3), dropping the wrist row which is zero."""
    return target[:, list(joints), :].reshape(target.shape[0], -1)


CONDITIONS = {
    "pose_emb": (train["pose_emb"], evaluation["pose_emb"]),
    "pose_emb_plus_tactile": (
        np.hstack([train["pose_emb"], train["tactile"]]),
        np.hstack([evaluation["pose_emb"], evaluation["tactile"]]),
    ),
    "pose_emb_plus_shuffled": (
        np.hstack([train["pose_emb"], train["shuffled"]]),
        np.hstack([evaluation["pose_emb"], evaluation["shuffled"]]),
    ),
    "pose_raw": (train["pose_raw"], evaluation["pose_raw"]),
    "pose_raw_plus_tactile": (
        np.hstack([train["pose_raw"], train["tactile"]]),
        np.hstack([evaluation["pose_raw"], evaluation["tactile"]]),
    ),
    "pose_raw_plus_shuffled": (
        np.hstack([train["pose_raw"], train["shuffled"]]),
        np.hstack([evaluation["pose_raw"], evaluation["shuffled"]]),
    ),
    "tactile_only": (train["tactile"], evaluation["tactile"]),
}

from sklearn.linear_model import RidgeCV  # noqa: E402

ALPHAS = np.logspace(-2, 4, 13)
moving_mask = evaluation["moving"]
clusters = evaluation["clusters"][moving_mask]

report = {
    "horizon_k": args.horizon_k,
    "eval_split": args.eval_split,
    "split_group_by": args.split_group_by,
    "n_train": int(train["target"].shape[0]),
    "n_eval_moving": int(moving_mask.sum()),
    "n_clusters": int(len(np.unique(clusters))),
    "joint_sets": {},
}

for set_name, joints in (("all_joints", ARTICULATING_JOINTS), ("fingertips", FINGERTIPS)):
    y_train = flatten_target(train["target"], joints)
    y_eval = flatten_target(evaluation["target"], joints)[moving_mask]

    # Predict-zero reference, the same baseline the end-to-end regression used.
    zero_mse = float(np.mean(y_eval ** 2))

    per_sample_error = {}
    entry = {"copy_zero_mse": zero_mse, "conditions": {}}
    for name, (x_train, x_eval) in CONDITIONS.items():
        scaled_train, scaled_eval = standardize(x_train, x_eval)
        model = RidgeCV(alphas=ALPHAS)
        model.fit(scaled_train, y_train)
        predictions = model.predict(scaled_eval[moving_mask])
        squared = (predictions - y_eval) ** 2
        per_sample_error[name] = squared.mean(axis=1)
        mse = float(squared.mean())
        entry["conditions"][name] = {
            "mse": mse,
            "r2_vs_zero": float(1.0 - mse / zero_mse),
            "alpha": float(getattr(model, "alpha_", float("nan"))),
            "n_features": int(x_train.shape[1]),
        }
        print(
            "  %-24s %-11s mse %.6e   R2 vs zero %+.4f   (d=%d)"
            % (set_name, name, mse, entry["conditions"][name]["r2_vs_zero"], x_train.shape[1]),
            flush=True,
        )

    # Clip-clustered paired bootstrap on the ERROR REDUCTION from adding touch.
    # Positive delta means touch lowers error.
    def bootstrap_reduction(with_touch, without_touch):
        unique = np.unique(clusters)
        rows_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
        rng = np.random.default_rng(args.split_seed)
        observed = float(per_sample_error[without_touch].mean() - per_sample_error[with_touch].mean())
        draws = []
        for _ in range(args.n_boot):
            drawn = rng.choice(unique, size=len(unique), replace=True)
            rows = np.concatenate([rows_by_cluster[c] for c in drawn])
            draws.append(
                per_sample_error[without_touch][rows].mean()
                - per_sample_error[with_touch][rows].mean()
            )
        draws = np.asarray(draws)
        return {
            "error_reduction": observed,
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)),
            "relative_pct": float(100.0 * observed / per_sample_error[without_touch].mean()),
        }

    entry["touch_effect"] = {
        "emb_vs_pose": bootstrap_reduction("pose_emb_plus_tactile", "pose_emb"),
        "emb_vs_shuffled": bootstrap_reduction("pose_emb_plus_tactile", "pose_emb_plus_shuffled"),
        "raw_vs_pose": bootstrap_reduction("pose_raw_plus_tactile", "pose_raw"),
        "raw_vs_shuffled": bootstrap_reduction("pose_raw_plus_tactile", "pose_raw_plus_shuffled"),
    }
    for label, values in entry["touch_effect"].items():
        verdict = "HELPS" if values["ci_low"] > 0 else ("HURTS" if values["ci_high"] < 0 else "no effect")
        print(
            "  %-24s %-16s reduction %+.3e [%+.3e, %+.3e]  %+.2f%%   %s"
            % (set_name, label, values["error_reduction"], values["ci_low"],
               values["ci_high"], values["relative_pct"], verdict),
            flush=True,
        )
    report["joint_sets"][set_name] = entry
    print(flush=True)

with open(args.output, "w") as handle:
    json.dump(report, handle, indent=2)
print("wrote " + args.output)
