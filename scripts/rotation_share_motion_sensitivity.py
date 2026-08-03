"""Is the rotation share an artifact of nearly-still frames?

THE WORRY. scripts/rotation_share.py applies no minimum-motion threshold: it
excludes degenerate palm geometry and nothing else. For a hand that barely
moves between t and t+k, the reported fraction

    1 - ||residual||^2 / ||wrist_translation_removed||^2

is a ratio of two tiny numbers, both dominated by annotation noise rather than
by motion. If those samples were driving the headline median, the claim would
be an artifact of stillness and would collapse under any sensible filter.

THE PREDICTION being tested. Annotation noise is roughly isotropic and
independent per joint, so it is badly explained by a single whole-hand
rotation. Still frames should therefore score LOW, not high, and including
them should bias the reported share DOWN -- making the published number
conservative rather than inflated. That is a prediction, not a measurement,
which is why this script exists.

WHAT IT REPORTS. Per-sample rigid share against per-sample motion magnitude
(the L2 norm of the wrist-translation-removed delta, i.e. the square root of
the denominator), summarised two ways:

  - median rigid share within each motion decile
  - median rigid share among samples above a motion percentile floor, which is
    exactly the filter a reviewer would ask you to apply

USAGE
    python scripts/rotation_share_motion_sensitivity.py --npy dexycb_poses.npy --name DexYCB
    python scripts/rotation_share_motion_sensitivity.py --opentouch-data <path> --name OpenTouch
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opentouch.articulation_frames import (  # noqa: E402
    all_target_variants, hand_frame_degenerate, rigid_fraction,
)
from rotation_share import _as_tensor  # noqa: E402

PERCENTILE_FLOORS = [0, 10, 25, 50, 75, 90]


def pairs_for_horizon(seqs, k, max_pairs=200_000, seed=0):
    """Same pairing rotation_share.py uses: consecutive (t, t+k) within each
    sequence, degenerate palms dropped, subsampled identically."""
    cur, fut = [], []
    for s in seqs:
        if s.shape[0] <= k:
            continue
        cur.append(s[:-k])
        fut.append(s[k:])
    if not cur:
        return None, None
    pose_t = torch.cat(cur)
    pose_future = torch.cat(fut)

    good = ~hand_frame_degenerate(pose_t)
    pose_t, pose_future = pose_t[good], pose_future[good]

    if pose_t.shape[0] > max_pairs:
        rng = np.random.default_rng(seed)
        sel = torch.as_tensor(np.sort(rng.choice(pose_t.shape[0], max_pairs, replace=False)))
        pose_t, pose_future = pose_t[sel], pose_future[sel]
    return pose_t, pose_future


def analyse(seqs, horizons, name):
    report = {"name": name, "n_sequences": len(seqs), "horizons": {}}
    for k in horizons:
        pose_t, pose_future = pairs_for_horizon(seqs, k)
        if pose_t is None:
            continue

        share = rigid_fraction(pose_t, pose_future).numpy()
        # Motion magnitude is the square root of rigid_fraction's denominator:
        # the total wrist-translation-removed displacement of the 21 joints.
        total = all_target_variants(pose_t, pose_future)["wrist_translation_removed"]
        motion = total.flatten(1).pow(2).sum(dim=1).sqrt().numpy()

        entry = {
            "n_pairs": int(share.size),
            "median_all": float(np.median(share)),
            "mean_all": float(share.mean()),
            "by_decile": [],
            "by_floor": {},
        }

        edges = np.percentile(motion, np.arange(0, 101, 10))
        for i in range(10):
            lo, hi = edges[i], edges[i + 1]
            m = (motion >= lo) & (motion <= hi) if i == 9 else (motion >= lo) & (motion < hi)
            if m.sum() == 0:
                continue
            entry["by_decile"].append({
                "decile": i + 1,
                "n": int(m.sum()),
                "motion_lo": float(lo),
                "motion_hi": float(hi),
                "median_rigid_share": float(np.median(share[m])),
            })

        for p in PERCENTILE_FLOORS:
            thr = np.percentile(motion, p)
            m = motion >= thr
            entry["by_floor"][str(p)] = {
                "n": int(m.sum()),
                "median_rigid_share": float(np.median(share[m])),
                "mean_rigid_share": float(share[m].mean()),
            }
        report["horizons"][str(k)] = entry
    return report


def _print(report):
    print("\n=== MOTION SENSITIVITY: %s (%d sequences) ==="
          % (report["name"], report["n_sequences"]))
    for k, e in report["horizons"].items():
        print("\nk=%s   n=%d   median(all)=%.1f%%   mean(all)=%.1f%%"
              % (k, e["n_pairs"], 100 * e["median_all"], 100 * e["mean_all"]))
        print("  motion decile (1=stillest, 10=largest motion):")
        print("    %-8s %8s %14s   %s" % ("decile", "n", "median share", "motion range"))
        for d in e["by_decile"]:
            print("    %-8d %8d %13.1f%%   [%.4f, %.4f]"
                  % (d["decile"], d["n"], 100 * d["median_rigid_share"],
                     d["motion_lo"], d["motion_hi"]))
        print("  keeping only samples above a motion percentile floor:")
        print("    %-10s %8s %14s %13s" % ("floor", "n kept", "median share", "mean share"))
        for p in PERCENTILE_FLOORS:
            v = e["by_floor"][str(p)]
            print("    p%-9d %8d %13.1f%% %12.1f%%"
                  % (p, v["n"], 100 * v["median_rigid_share"], 100 * v["mean_rigid_share"]))


def load_opentouch(data, sequence_length, split_seed, splits):
    from opentouch_train.data import VideoTactilePoseDataset, _load_and_split_dataset
    parts = _load_and_split_dataset(data, 0.1, 0.1, split_seed)
    seqs = []
    for split in splits:
        if split not in parts:
            continue
        base = VideoTactilePoseDataset(
            split=split, _preloaded=parts[split], hf_dataset_path=data,
            sequence_length=sequence_length,
            include_tactile=False, include_visual=False, include_pose=True,
        )
        for idx in range(len(base)):
            pose = base[idx]["hand_landmarks"]
            if pose.dim() == 4:
                pose = pose.squeeze(1)
            seqs.append(pose.float())
    return seqs


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npy")
    src.add_argument("--opentouch-data")
    ap.add_argument("--name", default=None)
    ap.add_argument("--horizons", type=int, nargs="+", default=[2, 8])
    ap.add_argument("--sequence-length", type=int, default=36)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.npy:
        seqs = _as_tensor(np.load(args.npy, allow_pickle=True))
        name = args.name or os.path.basename(args.npy)
    else:
        seqs = load_opentouch(args.opentouch_data, args.sequence_length,
                              args.split_seed, args.splits)
        name = args.name or "OpenTouch"
        print("Loaded %d windows" % len(seqs))

    report = analyse(seqs, args.horizons, name)
    _print(report)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
