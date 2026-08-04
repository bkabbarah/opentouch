"""Do models trained on the conventional target LEARN the rotation shortcut?

THE GAP THIS CLOSES. §2.25/§2.28/§2.30 measure that the wrist-relative
"articulation" target is 82-96% whole-hand rotation. That is a fact about the
DATA. It does not yet show that anyone exploits it. A reviewer can fairly say
"so the target is confounded -- maybe models still learn articulation anyway".

THE TEST. Take a model trained on the conventional target
(`--target-mode articulation_delta`), run inference, and decompose ITS OWN
PREDICTIONS with exactly the same rigid_fraction used on the data:

    ground truth share  = rigid_fraction(pose_t, pose_t + articulation_delta)
    prediction   share  = rigid_fraction(pose_t, pose_t + predicted_delta)

If the model's output is as rigid as the data -- or more so -- then it is
predicting whole-hand rotation and being scored as though that were hand
motion. That converts a measurement into a mechanism: the target rewards
rotation-tracking, so models rotation-track.

WHY "OR MORE SO" IS THE INTERESTING CASE. MSE-optimal prediction under
uncertainty collapses toward the conditional mean (§2.22). The rotational part
of the delta is the smooth, low-entropy, predictable part; true articulation is
the high-entropy part. So a model minimising MSE should concentrate even harder
on rotation than the data does. A prediction share ABOVE the ground-truth share
is the signature of the shortcut, not merely evidence consistent with it.

    python scripts/shortcut_analysis.py --checkpoint logs/<run>/checkpoints/epoch_N.pt \
        --data <dataset> --split val --out results_shortcut_<run>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from opentouch.articulation_frames import (  # noqa: E402
    all_target_variants, hand_frame_degenerate, rigid_fraction,
)
from opentouch.pose_regression import (  # noqa: E402
    POSE_DIM, PoseTransitionRegressor, decompose_world_delta,
)
from opentouch.regression_metrics import fingertip_displacement  # noqa: E402
from opentouch_train.data import VideoTactilePoseDataset  # noqa: E402
from opentouch_train.regression_data import PoseTransitionDataset, regression_collate_fn  # noqa: E402
from opentouch_train.regression_eval import _read_checkpoint_meta  # noqa: E402


def energy(delta):
    return delta.flatten(1).pow(2).sum(dim=1)


def summarize(share, label):
    return {
        "label": label,
        "n": int(share.numel()),
        "median_rigid_share": float(share.median()),
        "mean_rigid_share": float(share.mean()),
        "frac_above_90pct": float((share > 0.9).float().mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = _read_checkpoint_meta(args.checkpoint)
    if meta["target_mode"] != "articulation_delta":
        raise SystemExit(
            "This analysis only means anything for models trained on the CONVENTIONAL "
            "target. This checkpoint has target_mode=%s. A model trained on "
            "rigid_articulation has already had the rotation removed from its target, "
            "so it cannot exhibit the shortcut." % meta["target_mode"]
        )

    device = torch.device(args.device)
    model = PoseTransitionRegressor(
        use_tactile=not meta["pose_only"],
        tactile_emb_dim=meta["tactile_emb_dim"],
        hidden_dim=meta["hidden_dim"],
        tactile_correction_input=meta["tactile_correction_input"],
        fusion=meta["fusion"],
        output_dim=POSE_DIM,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    if next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    base = VideoTactilePoseDataset(
        hf_dataset_path=args.data, split=args.split,
        sequence_length=meta["sequence_length"], val_ratio=0.1, test_ratio=0.1,
        random_seed=meta["split_seed"], split_group_by=meta["split_group_by"],
        include_tactile=not meta["pose_only"], include_visual=False, include_pose=True,
    )
    ds = PoseTransitionDataset(
        base, meta["horizon_k"], shuffle_tactile=meta["shuffle_tactile"],
        shuffle_seed=meta["split_seed"], causal=meta["causal"],
        causal_window=meta["causal_window"], min_history=meta["min_history"],
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=regression_collate_fn)

    P, G, R = [], [], []
    with torch.inference_mode():
        for batch in dl:
            pose_t = batch["pose_t"].to(device)
            world_delta = batch["world_delta"].to(device)
            _, art = decompose_world_delta(world_delta)
            tac = batch["tactile_pressure"].to(device) if not meta["pose_only"] else None
            pred = model(pose_t, tac).reshape(art.shape)
            P.append(pose_t.float().cpu()); G.append(art.float().cpu()); R.append(pred.float().cpu())

    pose_t = torch.cat(P); gt = torch.cat(G); pred = torch.cat(R)
    # The target's wrist row is exactly zero by construction; hold the
    # prediction to the same convention so the two shares are comparable.
    pred = pred - pred[:, :1, :]

    good = ~hand_frame_degenerate(pose_t)
    moving = fingertip_displacement(gt) >= meta["motion_threshold"]
    keep = good & moving
    pose_t, gt, pred = pose_t[keep], gt[keep], pred[keep]

    gt_share = rigid_fraction(pose_t, pose_t + gt)
    pred_share = rigid_fraction(pose_t, pose_t + pred)

    # How much motion does the model emit at all, and how much of the
    # NON-rigid part does it recover? A model that predicts pure rotation has
    # a near-zero non-rigid residual regardless of its rigid share.
    gt_nonrigid = energy(all_target_variants(pose_t, pose_t + gt)["rigid_removed"]).sqrt()
    pred_nonrigid = energy(all_target_variants(pose_t, pose_t + pred)["rigid_removed"]).sqrt()

    # THE NUMBER THAT MATTERS. Split the model's squared error into the part
    # attributable to the rigid (rotational) component of the target and the
    # part attributable to the non-rigid (articulation) component.
    #
    # MSE is energy-weighted, so the MEAN rigid share governs it, not the
    # median. If the model emits almost no rotation, the target's rotational
    # energy passes straight into the error as a FLOOR that every method pays
    # equally -- which compresses all method differences into whatever is
    # left. That is a quantitative explanation for §2.22's observation that
    # copy-zero is nearly unbeatable and everything fights over 8-9%.
    gt_var = all_target_variants(pose_t, pose_t + gt)
    err = pred - gt
    total_sq = energy(err)
    rigid_gt = gt - gt_var["rigid_removed"]        # the rotational part of the target
    nonrigid_gt = gt_var["rigid_removed"]
    # Error the model would still incur if it predicted the non-rigid part
    # perfectly and continued to emit nothing for the rotational part.
    floor_sq = energy(rigid_gt)
    addressable = 1.0 - float(floor_sq.sum() / total_sq.sum()) if total_sq.sum() > 0 else float("nan")

    report = {
        "checkpoint": args.checkpoint, "split": args.split,
        "target_mode": meta["target_mode"], "horizon_k": meta["horizon_k"],
        "pose_only": meta["pose_only"], "shuffle_tactile": meta["shuffle_tactile"],
        "n_moving": int(keep.sum()),
        "ground_truth": summarize(gt_share, "ground truth delta"),
        "prediction": summarize(pred_share, "model prediction"),
        "shortcut_gap_median": float(pred_share.median() - gt_share.median()),
        "nonrigid_magnitude_ratio": float(pred_nonrigid.median() / gt_nonrigid.median()),
        "predicted_motion_magnitude_ratio": float(
            energy(pred).sqrt().median() / energy(gt).sqrt().median()),
        "mean_rigid_share_energy_weighted": float(
            (energy(rigid_gt).sum() / energy(gt).sum())),
        "frac_of_MSE_from_unpredicted_rotation": float(floor_sq.sum() / total_sq.sum()),
        "addressable_fraction_of_MSE": addressable,
    }

    print("\n=== DOES THE MODEL LEARN THE ROTATION SHORTCUT? ===")
    print("checkpoint: %s" % args.checkpoint)
    print("k=%s  pose_only=%s  shuffle=%s  n(moving)=%d\n"
          % (meta["horizon_k"], meta["pose_only"], meta["shuffle_tactile"], report["n_moving"]))
    print("%-24s %14s %14s %14s" % ("", "median share", "mean share", ">90% rigid"))
    for key in ("ground_truth", "prediction"):
        e = report[key]
        print("%-24s %13.1f%% %13.1f%% %13.1f%%"
              % (e["label"], 100 * e["median_rigid_share"],
                 100 * e["mean_rigid_share"], 100 * e["frac_above_90pct"]))
    print("\nshortcut gap (prediction - ground truth): %+.1f points"
          % (100 * report["shortcut_gap_median"]))
    print("non-rigid magnitude the model emits, vs ground truth: %.1f%%"
          % (100 * report["nonrigid_magnitude_ratio"]))
    print("total predicted motion magnitude, vs ground truth:    %.1f%%"
          % (100 * report["predicted_motion_magnitude_ratio"]))
    print("\nA POSITIVE gap means the model's output is MORE rotation-dominated than")
    print("the data it was trained on -- the signature of the shortcut. A non-rigid")
    print("magnitude far below 100% means it is barely predicting articulation at all.")

    print("\n=== WHERE THE MSE ACTUALLY GOES ===")
    print("rotational share of target energy (energy-weighted): %.1f%%"
          % (100 * report["mean_rigid_share_energy_weighted"]))
    print("share of this model's MSE that is UNPREDICTED ROTATION: %.1f%%"
          % (100 * report["frac_of_MSE_from_unpredicted_rotation"]))
    print("=> at most %.1f%% of the MSE is addressable by predicting articulation better."
          % (100 * report["addressable_fraction_of_MSE"]))
    print("Every method pays the rotational floor equally, so method differences are")
    print("compressed into what is left. Compare with §2.22: pose-only beats copy-zero")
    print("by 8-9%, which is the size of the addressable slice, not a modelling failure.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
