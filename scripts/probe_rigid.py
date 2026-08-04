"""Re-run the causal direction probe against the RIGID-ROTATION-REMOVED,
palm-framed target, holding everything else identical to
tactile_direction_probe.py.

The published probe decodes the sign of `wrist_translation_removed` on world
axes. The rigid diagnostic shows the best rigid rotation about the wrist
explains ~95% of that target's energy at the median sample, so the published
result cannot distinguish "tactile predicts finger articulation" from "tactile
predicts whole-hand re-orientation". This swaps ONLY the target and reports
both side by side.
"""
import argparse, json, logging, sys
import numpy as np, torch

sys.path.insert(0, "scripts")
from opentouch.articulation_frames import all_target_variants, hand_frame_degenerate
from opentouch.pose_regression import COORD_DIM, NUM_KEYPOINTS, WRIST_INDEX
from opentouch.regression_metrics import fingertip_displacement
from opentouch_train.data import VideoTactilePoseDataset, _load_and_split_dataset
from opentouch_train.regression_data import (
    PoseTransitionDataset, _causal_frame_indices, _make_derangement, compute_motion_threshold)
from tactile_direction_probe import (
    JOINT_LABELS, encode_causal, encode_pose_causal, fit_and_eval_probe,
    load_pose_encoder, load_tactile_encoder, min_history_mask, standardize)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--data", required=True)
p.add_argument("--horizon-k", type=int, default=8)
p.add_argument("--sequence-length", type=int, default=36)
p.add_argument("--causal-window", type=int, default=20)
p.add_argument("--min-history", type=int, default=10)
p.add_argument("--split-seed", type=int, default=42)
p.add_argument("--split-group-by", default="clip", choices=["clip", "scene"],
               help="Unit held disjoint between the probe's train and eval clips. "
                    "'scene' holds participants disjoint, which tests whether a decoded "
                    "signal generalizes to a NEW person rather than to new clips of "
                    "people the probe already saw.")
p.add_argument("--emb-dim", type=int, default=64)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--eval-split", default="val", choices=["val", "test"],
               help="Split the probe is scored on. The probe is always FIT on train. "
                    "The test split has been touched by no probe in this project, so "
                    "scoring on it is a genuine out-of-sample confirmation rather than "
                    "a repeat of the set every earlier decision was made against.")
p.add_argument("--all-variants", action="store_true",
               help="Also probe the two intermediate targets (rotation removed but world "
                    "axes; rotation kept but palm axes), which separates the frame change "
                    "from the rotation removal instead of confounding them.")
p.add_argument("--encoder-seed", type=int, default=None,
               help="Seed the RANDOM tactile encoder's initialisation ONLY. Without it "
                    "the encoder is built from ambient RNG state, so every --random-"
                    "tactile-encoder run used a different, unrecorded encoder and none "
                    "of them can be regenerated. Held separate from --split-seed on "
                    "purpose: varying the encoder must not also move the data split or "
                    "the derangement, or the contrast confounds encoder variance with "
                    "split variance. Pairing a full run and a scalar run on the SAME "
                    "encoder-seed removes encoder-init variance from the comparison "
                    "entirely.")
p.add_argument("--tactile-reduce", default="none", choices=["none", "scalar"],
               help="'scalar' replaces each tactile frame with its spatial mean, "
                    "broadcast back over the taxel grid: per-frame TOTAL pressure is "
                    "preserved exactly and every spatial pattern is destroyed, while "
                    "temporal structure and the encoder are untouched. HANDOFF 2.34 "
                    "found this costs only ~12% of the tactile benefit on grip "
                    "aperture, and 2.35 confirms via STAG that the reduction really "
                    "does destroy spatial information. Aperture is rotation-invariant "
                    "by construction, so this probe -- whose target is NOT -- is the "
                    "test of whether that null generalises.")
p.add_argument("--random-tactile-encoder", action="store_true",
               help="Replace the pretrained tactile encoder with a RANDOMLY INITIALISED "
                    "frozen one, leaving the pose encoder pretrained. The control for "
                    "whether this probe measures the LEARNED tactile representation or "
                    "merely raw tactile signal pushed through a fixed projection. On the "
                    "forecasting side the analogous control recovered ~71% of the effect "
                    "and actually beat the pretrained encoder (HANDOFF 2.19d), so the "
                    "same question is open here. If marginal AUC survives at full "
                    "strength with random weights, the project's hypothesis -- that an "
                    "encoder representing temporal structure is what unlocks predictive "
                    "tactile information -- is not supported by this probe.")
p.add_argument("--output", required=True)
a = p.parse_args()
dev = torch.device(a.device)

if a.random_tactile_encoder:
    from opentouch.tactile_encoder import CNNetEmbedding
    if a.encoder_seed is not None:
        torch.manual_seed(a.encoder_seed)
    tac = CNNetEmbedding(emb_dim=a.emb_dim).to(dev).eval()
    for _p in tac.parameters():
        _p.requires_grad_(False)
    log.warning(
        "RANDOM-TACTILE CONTROL: the tactile encoder is randomly initialised and frozen "
        "-- no pretrained weights. The pose encoder is still the pretrained one, so any "
        "marginal AUC here is what raw tactile signal buys through a fixed random "
        "projection, with the learned tactile representation removed."
    )
else:
    tac = load_tactile_encoder(a.checkpoint, a.emb_dim, dev)
pen = load_pose_encoder(a.checkpoint, a.emb_dim, dev)
splits = _load_and_split_dataset(a.data, 0.1, 0.1, a.split_seed, a.split_group_by)

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
    tactile = ds._tactile
    if a.tactile_reduce == "scalar":
        # Same reduction as PoseTransitionRegressor._reduce_tactile. Applied to
        # the shared tensor so the real and deranged encodings are treated
        # identically -- reducing only one of them would confound the ablation
        # with the derangement control.
        tactile = tactile.mean(dim=(-2, -1), keepdim=True).expand_as(tactile).contiguous()
    tc = encode_causal(tac, tactile, widx, tv, cfi, a.batch_size, dev).numpy()
    ts = encode_causal(tac, tactile, perm[widx], tv, cfi, a.batch_size, dev).numpy()
    pe = encode_pose_causal(pen, ds._pose, widx, tv, cfi, a.batch_size, dev).numpy()
    log.info("[%s] samples=%d moving=%d" % (name, len(widx), moving.sum()))
    return dict(tgts=tgts, moving=moving, tactile=tc, shuf=ts, pose_emb=pe), threshold

tr, thr = prep("train", None)
va, _ = prep(a.eval_split, thr)

CONDS = {
    "tactile": (tr["tactile"], va["tactile"]),
    "shuffled_tactile": (tr["shuf"], va["shuf"]),
    "pose_emb_causal": (tr["pose_emb"], va["pose_emb"]),
    "pose_emb_plus_tactile": (np.hstack([tr["pose_emb"], tr["tactile"]]),
                              np.hstack([va["pose_emb"], va["tactile"]])),
    "pose_emb_plus_shuffled": (np.hstack([tr["pose_emb"], tr["shuf"]]),
                               np.hstack([va["pose_emb"], va["shuf"]])),
}
STD = {n: standardize(x, y) for n, (x, y) in CONDS.items()}
MOV = {n: y[va["moving"]] for n, (_, y) in STD.items()}

# Full 2x2 decomposition. The headline comparison changes TWO things at
# once (remove whole-hand rotation, AND re-express in palm axes), so both
# intermediates are computed to attribute the gain to one, the other, or
# their interaction:
#
#                        world axes                 palm axes
#   rotation kept   wrist_translation_removed   ..._handframe
#   rotation gone   rigid_removed               rigid_removed_handframe
ALL_AXES = {
    "wrist_translation_removed": ("x", "y", "z"),
    "wrist_translation_removed_handframe": ("long", "flex", "normal"),
    "rigid_removed": ("x", "y", "z"),
    "rigid_removed_handframe": ("long", "flex", "normal"),
}
AXES = ALL_AXES if a.all_variants else {
    k: v for k, v in ALL_AXES.items()
    if k in ("wrist_translation_removed", "rigid_removed_handframe")
}
out = {"horizon_k": a.horizon_k, "sequence_length": a.sequence_length,
       "split_group_by": a.split_group_by, "checkpoint": a.checkpoint,
       "random_tactile_encoder": a.random_tactile_encoder,
       "tactile_reduce": a.tactile_reduce,
       "encoder_seed": a.encoder_seed,
       "eval_split": a.eval_split,
       "n_val_moving": int(va["moving"].sum()), "results": {}}

for variant, axis_names in AXES.items():
    out["results"][variant] = {}
    for cond in CONDS:
        per_axis = {ax: [] for ax in axis_names}
        for j in range(NUM_KEYPOINTS):
            if j == WRIST_INDEX: continue
            for ai, ax in enumerate(axis_names):
                ytr = (tr["tgts"][variant][:, j, ai] > 0).numpy().astype(int)
                yva = (va["tgts"][variant][:, j, ai] > 0).numpy().astype(int)[va["moving"]]
                r = fit_and_eval_probe(STD[cond][0], ytr, MOV[cond], yva)
                if r and r["auc"] == r["auc"]: per_axis[ax].append(r["auc"])
        out["results"][variant][cond] = {ax: float(np.mean(v)) for ax, v in per_axis.items() if v}
        log.info("  %-28s %-24s %s" % (variant, cond,
            {ax: round(float(np.mean(v)), 4) for ax, v in per_axis.items() if v}))

for variant, axis_names in AXES.items():
    r = out["results"][variant]
    best = max(axis_names, key=lambda ax: r["tactile"][ax])
    marg = r["pose_emb_plus_tactile"][best] - r["pose_emb_causal"][best]
    shuf = r["pose_emb_plus_shuffled"][best] - r["pose_emb_causal"][best]
    out["results"][variant]["_summary"] = {
        "best_axis": best, "tactile_alone": r["tactile"][best],
        "shuffled_alone": r["shuffled_tactile"][best],
        "marginal_over_matched_pose": marg, "shuffled_marginal": shuf}
    print("\n### %s  best axis=%s" % (variant, best))
    print("    tactile alone %.4f   shuffled alone %.4f" % (r["tactile"][best], r["shuffled_tactile"][best]))
    print("    marginal over matched-temporal pose: %+.4f  (shuffled %+.4f)" % (marg, shuf))

json.dump(out, open(a.output, "w"), indent=2)
print("\nwrote " + a.output)
