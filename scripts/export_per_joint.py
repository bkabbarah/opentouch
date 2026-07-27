"""Per-joint, per-axis AUC table for the paper.

probe_rigid.py and probe_rigid_ci.py both aggregate to per-axis means before
writing, so the per-joint numbers a results table needs were never saved for
the corrected target. (The older published-target probe DID save them, which
is why they exist there and not here.) This recomputes them and writes both a
JSON and a CSV.

For every (joint, axis) it reports the AUC of each condition, the marginal
contribution of touch over each pose baseline, and a clip-clustered paired
bootstrap interval on that marginal. Identical machinery to the aggregate
runs -- same encoders, same causal windowing, same derangement, same sample
set -- so the per-axis means of this table reproduce the published numbers
and can be checked against them.
"""

import argparse
import csv
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from opentouch.articulation_frames import (
    AXIS_NAMES_HAND,
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
    JOINT_LABELS,
    encode_causal,
    encode_pose_causal,
    load_pose_encoder,
    load_tactile_encoder,
    min_history_mask,
    standardize,
)
from tactile_subset_probe import clip_clustered_bootstrap, cluster_ids_for_windows

LAGS = [0, 1, 2, 3, 5, 8, 12, 19]
MAX_ITER = 5000

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
parser.add_argument(
    "--include-raw-pose", action="store_true",
    help="Also fit the 504-dim raw-kinematics conditions. Roughly triples runtime.",
)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--output-prefix", required=True)
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

    prepared = {
        "target": targets["rigid_removed_handframe"],
        "moving": moving,
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
    if args.include_raw_pose:
        windows_t = torch.as_tensor(window_idx)
        frames = []
        for lag in LAGS:
            frame_idx = np.clip(t_values - lag, 0, None)
            gathered = dataset._pose[windows_t, torch.as_tensor(frame_idx)]
            gathered = gathered - gathered[:, WRIST_INDEX : WRIST_INDEX + 1, :]
            frames.append(gathered.reshape(gathered.shape[0], -1).numpy())
        # Differences rather than stacked frames: same span, far better
        # conditioned. See probe_rawpose.py.
        prepared["pose_raw"] = np.hstack([frames[0]] + [frames[0] - f for f in frames[1:]])
    print(
        "[%s] samples=%d moving=%d" % (split_name, len(window_idx), int(moving.sum())),
        flush=True,
    )
    return prepared, threshold


train, motion_threshold = prepare("train", None)
evaluation, _ = prepare(args.eval_split, motion_threshold)

CONDITIONS = {
    "tactile": (train["tactile"], evaluation["tactile"]),
    "shuffled_tactile": (train["shuffled"], evaluation["shuffled"]),
    "pose_emb": (train["pose_emb"], evaluation["pose_emb"]),
    "pose_emb_plus_tactile": (
        np.hstack([train["pose_emb"], train["tactile"]]),
        np.hstack([evaluation["pose_emb"], evaluation["tactile"]]),
    ),
    "pose_emb_plus_shuffled": (
        np.hstack([train["pose_emb"], train["shuffled"]]),
        np.hstack([evaluation["pose_emb"], evaluation["shuffled"]]),
    ),
}
if args.include_raw_pose:
    CONDITIONS["pose_raw"] = (train["pose_raw"], evaluation["pose_raw"])
    CONDITIONS["pose_raw_plus_tactile"] = (
        np.hstack([train["pose_raw"], train["tactile"]]),
        np.hstack([evaluation["pose_raw"], evaluation["tactile"]]),
    )

scaled = {name: standardize(x, y) for name, (x, y) in CONDITIONS.items()}
moving_mask = evaluation["moving"]
eval_features = {name: y[moving_mask] for name, (_, y) in scaled.items()}
clusters = evaluation["clusters"][moving_mask]

import warnings  # noqa: E402

from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

non_converged = {}
rows = []

for joint in range(NUM_KEYPOINTS):
    if joint == WRIST_INDEX:
        continue
    for axis_index, axis_name in enumerate(AXIS_NAMES_HAND):
        y_train = (train["target"][:, joint, axis_index] > 0).numpy().astype(int)
        y_eval = (evaluation["target"][:, joint, axis_index] > 0).numpy().astype(int)[moving_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_eval)) < 2:
            continue

        scores, aucs = {}, {}
        for name in CONDITIONS:
            model = LogisticRegression(max_iter=MAX_ITER, random_state=42)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(scaled[name][0], y_train)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    non_converged[name] = non_converged.get(name, 0) + 1
            scores[name] = model.predict_proba(eval_features[name])[:, 1]
            aucs[name] = roc_auc_score(y_eval, scores[name])

        row = {
            "joint": JOINT_LABELS[joint],
            "axis": axis_name,
            "n_eval": int(len(y_eval)),
            "positive_rate": float(y_eval.mean()),
        }
        row.update({"auc_%s" % k: float(v) for k, v in aucs.items()})

        vs_pose = clip_clustered_bootstrap(
            y_eval, scores["pose_emb_plus_tactile"], scores["pose_emb"],
            clusters, args.n_boot, 42,
        )
        vs_shuffled = clip_clustered_bootstrap(
            y_eval, scores["pose_emb_plus_tactile"], scores["pose_emb_plus_shuffled"],
            clusters, args.n_boot, 42,
        )
        row.update({
            "marginal_vs_pose": vs_pose["delta"],
            "marginal_vs_pose_ci_low": vs_pose["ci_low"],
            "marginal_vs_pose_ci_high": vs_pose["ci_high"],
            "marginal_vs_shuffled": vs_shuffled["delta"],
            "marginal_vs_shuffled_ci_low": vs_shuffled["ci_low"],
            "marginal_vs_shuffled_ci_high": vs_shuffled["ci_high"],
        })
        if args.include_raw_pose:
            vs_raw = clip_clustered_bootstrap(
                y_eval, scores["pose_raw_plus_tactile"], scores["pose_raw"],
                clusters, args.n_boot, 42,
            )
            row.update({
                "marginal_vs_raw_pose": vs_raw["delta"],
                "marginal_vs_raw_pose_ci_low": vs_raw["ci_low"],
                "marginal_vs_raw_pose_ci_high": vs_raw["ci_high"],
            })
        rows.append(row)
        print(
            "  %-12s %-7s tactile %.4f  pose %.4f  pose+tac %.4f  marginal %+.4f [%+.4f,%+.4f]"
            % (row["joint"], axis_name, aucs["tactile"], aucs["pose_emb"],
               aucs["pose_emb_plus_tactile"], vs_pose["delta"],
               vs_pose["ci_low"], vs_pose["ci_high"]),
            flush=True,
        )

# Per-axis means, so this table can be checked against the published aggregates.
summary = {}
for axis_name in AXIS_NAMES_HAND:
    axis_rows = [r for r in rows if r["axis"] == axis_name]
    if not axis_rows:
        continue
    summary[axis_name] = {
        key: float(np.mean([r[key] for r in axis_rows]))
        for key in axis_rows[0]
        if key.startswith(("auc_", "marginal_"))
    }
    summary[axis_name]["n_joints_marginal_ci_excludes_zero"] = int(
        sum(1 for r in axis_rows if r["marginal_vs_pose_ci_low"] > 0)
    )
    summary[axis_name]["n_joints"] = len(axis_rows)

report = {
    "horizon_k": args.horizon_k,
    "eval_split": args.eval_split,
    "split_group_by": args.split_group_by,
    "checkpoint": args.checkpoint,
    "target": "rigid_removed_handframe",
    "n_eval_moving": int(moving_mask.sum()),
    "n_clusters": int(len(np.unique(clusters))),
    "n_boot": args.n_boot,
    "non_converged_fits": non_converged,
    "per_axis_means": summary,
    "per_joint": rows,
}
with open(args.output_prefix + ".json", "w") as handle:
    json.dump(report, handle, indent=2)
with open(args.output_prefix + ".csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\n=== per-axis means (check these against the published aggregates) ===", flush=True)
for axis_name, values in summary.items():
    print("  %-7s tactile %.4f  pose %.4f  pose+tac %.4f  marginal %+.4f  (%d/%d joints CI>0)" % (
        axis_name, values["auc_tactile"], values["auc_pose_emb"],
        values["auc_pose_emb_plus_tactile"], values["marginal_vs_pose"],
        values["n_joints_marginal_ci_excludes_zero"], values["n_joints"]))
if non_converged:
    print("\n!! non-converged fits: %s -- treat as provisional" % non_converged)
else:
    print("\nAll fits converged.")
print("wrote %s.json and %s.csv" % (args.output_prefix, args.output_prefix))
