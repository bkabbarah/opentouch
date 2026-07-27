"""Does touch add information over the FULL kinematic state, not just over a
learned pose embedding?

THE THREAT THIS ADDRESSES. Every result so far compares touch against
`pose_emb_causal`, a frozen 64-dim biGRU embedding taken from the retrieval
checkpoint. That embedding is demonstrably lossy: raw single-frame pose beats
it at long horizons. So "touch adds information a pose encoder does not
supply" could mean either

  (a) touch carries contact information genuinely absent from hand kinematics, or
  (b) touch is recovering kinematic information the 64-dim bottleneck discarded.

The shuffled control cannot separate these. A sample's touch reading is taken
at the same instant as its pose and is therefore correlated with it, while a
deranged touch reading comes from a different moment with different pose. The
shuffle only rules out "any 64 extra numbers help".

THE TEST. Give the probe the raw wrist-centred pose at several causal lags,
flattened. A linear model over lagged positions can form any linear function
of position, velocity and acceleration, so this is the full kinematic state,
uncompressed, with no bottleneck. If touch STILL adds on top of that, reading
(b) is dead and the claim is about contact information.

Wrist-centred rather than raw world coordinates, matching what PoseEncoder
does internally: absolute position would let the probe identify the clip.
"""

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from opentouch.articulation_frames import all_target_variants, hand_frame_degenerate
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
from tactile_subset_probe import clip_clustered_bootstrap, cluster_ids_for_windows

# Lags into the causal window, measured back from t.
#
# Fed as [p(t), p(t)-p(t-1), p(t)-p(t-2), ...] rather than as raw stacked
# frames. That is an invertible linear map of the stacked frames, so the two
# span exactly the same function class and a converged solution is identical
# -- but stacked lagged positions are near-duplicates of each other, which
# leaves lbfgs badly conditioned. The first version of this script hit the
# iteration limit on 177 of 240 fits, which made the raw-pose baseline look
# artificially weak and inflated the apparent tactile gain. Differences are
# small and far less collinear, so the same information optimises cleanly.
LAGS = [0, 1, 2, 3, 5, 8, 12, 19]
MAX_ITER = 5000
AXES = ("radial", "spread", "curl")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--horizon-k", type=int, default=8)
parser.add_argument("--sequence-length", type=int, default=36)
parser.add_argument("--causal-window", type=int, default=20)
parser.add_argument("--min-history", type=int, default=10)
parser.add_argument("--split-seed", type=int, default=42)
parser.add_argument("--split-group-by", default="clip", choices=["clip", "scene"])
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

    # Raw kinematic history: wrist-centred pose at each lag. Every index is
    # clamped at 0 and never exceeds t, so no future frame can enter.
    windows_t = torch.as_tensor(window_idx)
    frames = []
    for lag in LAGS:
        frame_idx = np.clip(t_values - lag, 0, None)
        gathered = dataset._pose[windows_t, torch.as_tensor(frame_idx)]
        gathered = gathered - gathered[:, WRIST_INDEX : WRIST_INDEX + 1, :]
        frames.append(gathered.reshape(gathered.shape[0], -1).numpy())
    # frames[0] is p(t); the rest become displacements from it.
    pose_raw = np.hstack([frames[0]] + [frames[0] - f for f in frames[1:]])

    prepared = {
        "target": targets["rigid_removed_handframe"],
        "moving": moving,
        "pose_raw": pose_raw,
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
        "[%s] samples=%d moving=%d raw_pose_dims=%d"
        % (split_name, len(window_idx), int(moving.sum()), pose_raw.shape[1]),
        flush=True,
    )
    return prepared, threshold


train, motion_threshold = prepare("train", None)
val, _ = prepare("val", motion_threshold)

CONDITIONS = {
    "pose_emb": (train["pose_emb"], val["pose_emb"]),
    "pose_raw": (train["pose_raw"], val["pose_raw"]),
    "pose_raw_plus_tactile": (
        np.hstack([train["pose_raw"], train["tactile"]]),
        np.hstack([val["pose_raw"], val["tactile"]]),
    ),
    "pose_raw_plus_shuffled": (
        np.hstack([train["pose_raw"], train["shuffled"]]),
        np.hstack([val["pose_raw"], val["shuffled"]]),
    ),
}
scaled = {name: standardize(x, y) for name, (x, y) in CONDITIONS.items()}
moving_mask = val["moving"]
eval_features = {name: y[moving_mask] for name, (_, y) in scaled.items()}
clusters = val["clusters"][moving_mask]

import warnings  # noqa: E402

from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# Counted and reported, never swallowed: a non-converged baseline looks
# artificially weak and would inflate the measured tactile contribution.
non_converged = {}

report = {
    "horizon_k": args.horizon_k,
    "lags": LAGS,
    "split_group_by": args.split_group_by,
    "n_val_moving": int(moving_mask.sum()),
    "n_clusters": int(len(np.unique(clusters))),
    "axes": {},
}

for axis_index, axis_name in enumerate(AXES):
    aucs = {name: [] for name in CONDITIONS}
    comparisons = {"vs_raw_pose": [], "vs_shuffled": []}
    for joint in range(NUM_KEYPOINTS):
        if joint == WRIST_INDEX:
            continue
        y_train = (train["target"][:, joint, axis_index] > 0).numpy().astype(int)
        y_val = (val["target"][:, joint, axis_index] > 0).numpy().astype(int)[moving_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue
        scores = {}
        for name in CONDITIONS:
            model = LogisticRegression(max_iter=MAX_ITER, random_state=42)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(scaled[name][0], y_train)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    non_converged[name] = non_converged.get(name, 0) + 1
            scores[name] = model.predict_proba(eval_features[name])[:, 1]
            aucs[name].append(roc_auc_score(y_val, scores[name]))
        comparisons["vs_raw_pose"].append(clip_clustered_bootstrap(
            y_val, scores["pose_raw_plus_tactile"], scores["pose_raw"],
            clusters, args.n_boot, 42,
        ))
        comparisons["vs_shuffled"].append(clip_clustered_bootstrap(
            y_val, scores["pose_raw_plus_tactile"], scores["pose_raw_plus_shuffled"],
            clusters, args.n_boot, 42,
        ))

    entry = {"auc_mean": {n: float(np.mean(v)) for n, v in aucs.items() if v}}
    for comparison, values in comparisons.items():
        entry[comparison] = {
            "mean_delta": float(np.mean([v["delta"] for v in values])),
            "mean_ci_low": float(np.mean([v["ci_low"] for v in values])),
            "mean_ci_high": float(np.mean([v["ci_high"] for v in values])),
            "n_joints_ci_excludes_zero": int(sum(1 for v in values if v["ci_low"] > 0)),
            "n_joints": len(values),
        }
    report["axes"][axis_name] = entry

    means = entry["auc_mean"]
    print("\n### %s" % axis_name, flush=True)
    print(
        "    pose_emb %.4f   pose_RAW %.4f   raw+tactile %.4f   raw+shuffled %.4f"
        % (means["pose_emb"], means["pose_raw"],
           means["pose_raw_plus_tactile"], means["pose_raw_plus_shuffled"]),
        flush=True,
    )
    for comparison in ("vs_raw_pose", "vs_shuffled"):
        e = entry[comparison]
        print(
            "    %-13s %+.4f [%+.4f, %+.4f]  (%d/%d joints CI>0)"
            % (comparison, e["mean_delta"], e["mean_ci_low"], e["mean_ci_high"],
               e["n_joints_ci_excludes_zero"], e["n_joints"]),
            flush=True,
        )

report["non_converged_fits"] = non_converged
total_bad = sum(non_converged.values())
if total_bad:
    print("\n!! %d fits hit the iteration limit: %s" % (total_bad, non_converged), flush=True)
    print("!! Treat these numbers as provisional; a non-converged baseline inflates "
          "the apparent tactile gain.", flush=True)
else:
    print("\nAll fits converged.", flush=True)

with open(args.output, "w") as handle:
    json.dump(report, handle, indent=2)
print("\nwrote " + args.output)
