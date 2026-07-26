"""Clip-clustered confidence intervals for tactile's marginal contribution.

probe_rigid.py reports point AUCs. Samples are heavily autocorrelated (adjacent
t share nearly all their causal frames; windows within a clip share
participant, scene and glove calibration), so a per-sample interval would be
several times too narrow. This resamples whole CLIPS, paired, so the shared
difficulty of a clip cancels in the difference.

Reports, per axis, the marginal AUC of (pose_emb + tactile) over pose_emb, and
over the dimensionality-matched shuffled twin, with 95% intervals.
"""
import argparse, json, sys
import numpy as np, torch

sys.path.insert(0, "scripts")
from opentouch.articulation_frames import all_target_variants, hand_frame_degenerate
from opentouch.pose_regression import COORD_DIM, NUM_KEYPOINTS, WRIST_INDEX
from opentouch.regression_metrics import fingertip_displacement
from opentouch_train.data import VideoTactilePoseDataset, _load_and_split_dataset
from opentouch_train.regression_data import (
    PoseTransitionDataset, _causal_frame_indices, _make_derangement, compute_motion_threshold)
from tactile_direction_probe import (
    encode_causal, encode_pose_causal, load_pose_encoder, load_tactile_encoder,
    min_history_mask, standardize)
from tactile_subset_probe import clip_clustered_bootstrap, cluster_ids_for_windows

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--data", required=True)
p.add_argument("--horizon-k", type=int, default=8)
p.add_argument("--sequence-length", type=int, default=36)
p.add_argument("--causal-window", type=int, default=20)
p.add_argument("--min-history", type=int, default=10)
p.add_argument("--split-seed", type=int, default=42)
p.add_argument("--emb-dim", type=int, default=64)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--n-boot", type=int, default=1000)
p.add_argument("--cluster-unit", default="clip", choices=["clip", "scene"])
p.add_argument("--target-variant", default="rigid_removed_handframe")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--output", required=True)
a = p.parse_args()
dev = torch.device(a.device)

AXES = {"rigid_removed_handframe": ("long", "flex", "normal"),
        "wrist_translation_removed": ("x", "y", "z")}[a.target_variant]

tac = load_tactile_encoder(a.checkpoint, a.emb_dim, dev)
pen = load_pose_encoder(a.checkpoint, a.emb_dim, dev)
splits = _load_and_split_dataset(a.data, 0.1, 0.1, a.split_seed)

def prep(name, threshold):
    base = VideoTactilePoseDataset(split=name, _preloaded=splits[name], hf_dataset_path=a.data,
        sequence_length=a.sequence_length, include_tactile=True, include_visual=False, include_pose=True)
    ds = PoseTransitionDataset(base, a.horizon_k, shuffle_tactile=False, causal=False)
    if threshold is None:
        threshold = compute_motion_threshold(ds, percentile=25.0)
    pose = ds._pose; n = pose.shape[0]; vt = ds.valid_t_per_window
    widx = np.repeat(np.arange(n), vt); tv = np.tile(np.arange(vt), n)
    keep = min_history_mask(tv, a.causal_window, a.min_history)
    widx, tv = widx[keep], tv[keep]
    pt = pose[:, :vt].reshape(-1, NUM_KEYPOINTS, COORD_DIM)[keep]
    pf = pose[:, a.horizon_k:].reshape(-1, NUM_KEYPOINTS, COORD_DIM)[keep]
    good = (~hand_frame_degenerate(pt)).numpy()
    widx, tv, pt, pf = widx[good], tv[good], pt[good], pf[good]
    tgts = all_target_variants(pt, pf)
    moving = (fingertip_displacement(tgts["wrist_translation_removed"]) >= threshold).numpy()
    cfi = _causal_frame_indices(vt, a.causal_window); perm = _make_derangement(n, seed=a.split_seed)
    out = dict(
        tgt=tgts[a.target_variant], moving=moving,
        tactile=encode_causal(tac, ds._tactile, widx, tv, cfi, a.batch_size, dev).numpy(),
        shuf=encode_causal(tac, ds._tactile, perm[widx], tv, cfi, a.batch_size, dev).numpy(),
        pose_emb=encode_pose_causal(pen, ds._pose, widx, tv, cfi, a.batch_size, dev).numpy(),
        clusters=cluster_ids_for_windows(ds, widx, a.cluster_unit))
    print(f"[{name}] samples={len(widx)} moving={int(moving.sum())} clusters={len(np.unique(out['clusters']))}", flush=True)
    return out, threshold

tr, thr = prep("train", None)
va, _ = prep("val", thr)

CONDS = {"pose": (tr["pose_emb"], va["pose_emb"]),
         "tactile": (np.hstack([tr["pose_emb"], tr["tactile"]]), np.hstack([va["pose_emb"], va["tactile"]])),
         "shuffled": (np.hstack([tr["pose_emb"], tr["shuf"]]), np.hstack([va["pose_emb"], va["shuf"]]))}
STD = {n: standardize(x, y) for n, (x, y) in CONDS.items()}
mov = va["moving"]
MOV = {n: y[mov] for n, (_, y) in STD.items()}
clusters = va["clusters"][mov]

from sklearn.linear_model import LogisticRegression
report = {"horizon_k": a.horizon_k, "target_variant": a.target_variant,
          "cluster_unit": a.cluster_unit, "n_boot": a.n_boot,
          "n_val_moving": int(mov.sum()), "n_clusters": int(len(np.unique(clusters))), "axes": {}}

for ai, ax in enumerate(AXES):
    per_joint = {"vs_pose": [], "vs_shuffled": []}
    for j in range(NUM_KEYPOINTS):
        if j == WRIST_INDEX: continue
        ytr = (tr["tgt"][:, j, ai] > 0).numpy().astype(int)
        yva = (va["tgt"][:, j, ai] > 0).numpy().astype(int)[mov]
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2: continue
        sc = {}
        for n in CONDS:
            clf = LogisticRegression(max_iter=2000, random_state=42).fit(STD[n][0], ytr)
            sc[n] = clf.predict_proba(MOV[n])[:, 1]
        per_joint["vs_pose"].append(clip_clustered_bootstrap(yva, sc["tactile"], sc["pose"], clusters, a.n_boot, 42))
        per_joint["vs_shuffled"].append(clip_clustered_bootstrap(yva, sc["tactile"], sc["shuffled"], clusters, a.n_boot, 42))
    entry = {}
    for comp, vals in per_joint.items():
        entry[comp] = {"mean_delta": float(np.mean([v["delta"] for v in vals])),
                       "mean_ci_low": float(np.mean([v["ci_low"] for v in vals])),
                       "mean_ci_high": float(np.mean([v["ci_high"] for v in vals])),
                       "n_joints_ci_excludes_zero": int(sum(1 for v in vals if v["ci_low"] > 0)),
                       "n_joints": len(vals)}
    report["axes"][ax] = entry
    print(f"  {ax}: vs_pose {entry['vs_pose']['mean_delta']:+.4f} "
          f"[{entry['vs_pose']['mean_ci_low']:+.4f},{entry['vs_pose']['mean_ci_high']:+.4f}] "
          f"({entry['vs_pose']['n_joints_ci_excludes_zero']}/{entry['vs_pose']['n_joints']} joints CI>0)  |  "
          f"vs_shuffled {entry['vs_shuffled']['mean_delta']:+.4f} "
          f"[{entry['vs_shuffled']['mean_ci_low']:+.4f},{entry['vs_shuffled']['mean_ci_high']:+.4f}] "
          f"({entry['vs_shuffled']['n_joints_ci_excludes_zero']}/{entry['vs_shuffled']['n_joints']})", flush=True)

json.dump(report, open(a.output, "w"), indent=2)
print("wrote " + a.output)
