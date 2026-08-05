"""Dataset-agnostic rotation-share diagnostic.

THE CLAIM THIS MEASURES. A wrist-relative "articulation delta" -- the standard
target for hand-motion prediction -- removes the wrist's TRANSLATION but not
its ROTATION. This reports the fraction of that target's energy the BEST-FIT
rigid rotation about the wrist explains. Because the rotation is fit per
sample (a maximum over rotations), the share is an UPPER BOUND on rotation
content -- equivalently, the residual is a lower bound on true articulation.
Coordinated finger motion that happens to resemble a rigid rotation is counted
as rotation. On OpenTouch the median share is ~96%, which is why tactile
appeared redundant with pose kinematics until the target was corrected.

WHY THIS FILE EXISTS SEPARATELY FROM scripts/rigid_diag.py. That script is
welded to VideoTactilePoseDataset and this project's split machinery, so it
can only ever speak about OpenTouch. The claim is not about OpenTouch. This
takes a plain array of hand-pose sequences and needs no tactile, no encoder,
no checkpoint, no split -- so it runs on any dataset that ships hand-joint
annotations, and it is the piece worth releasing.

INPUT CONTRACT
    poses : (N, T, 21, 3) float array of hand joint positions over time,
            or a list of (T_i, 21, 3) arrays if sequences differ in length.
    Joint 0 must be the wrist. Ordering of the other 20 only has to be
    consistent within the array -- the diagnostic is permutation-invariant
    across non-wrist joints because it works on the point cloud as a whole.

    Sequences must be TEMPORAL and at a known frame rate. Single-frame
    datasets (e.g. FreiHAND) cannot be used: there is no delta to decompose.

USAGE
    from scripts.rotation_share import rotation_share_report
    report = rotation_share_report(poses, horizons=[2, 4, 8, 16], fps=30)

    python scripts/rotation_share.py --npy poses.npy --fps 30 --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from opentouch.articulation_frames import hand_frame_degenerate, rigid_fraction


def _as_tensor(poses):
    # Variable-length sequences survive np.save/np.load as an OBJECT array, not
    # as a list -- so `np.load(...)` on a ragged dump lands here, not in the
    # list branch below. Without this, the documented "list of (T_i,21,3)"
    # input works in-process but fails the moment it round-trips through a
    # .npy file, which is exactly how the CLI is used.
    if isinstance(poses, np.ndarray) and poses.dtype == object:
        poses = list(poses)
    if isinstance(poses, (list, tuple)):
        seqs = []
        for i, p in enumerate(poses):
            t = torch.as_tensor(np.asarray(p, dtype=np.float32), dtype=torch.float32)
            if t.dim() != 3 or tuple(t.shape[-2:]) != (21, 3):
                raise ValueError(
                    f"sequence {i} must be (T,21,3) with joint 0 = wrist, got "
                    f"{tuple(t.shape)}"
                )
            seqs.append(t)
        return seqs
    arr = torch.as_tensor(np.asarray(poses), dtype=torch.float32)
    if arr.dim() != 4 or arr.shape[-2:] != (21, 3):
        raise ValueError(
            f"expected (N,T,21,3), got {tuple(arr.shape)}. If joints are ordered "
            "differently, reindex so joint 0 is the wrist; the other 20 need only "
            "be internally consistent."
        )
    return [arr[i] for i in range(arr.shape[0])]


def rotation_share_report(poses, horizons=(2, 4, 8, 16), fps=None, name=None,
                          groups=None, max_pairs_per_horizon=200_000, seed=0):
    """Rotation share of the wrist-relative delta, per horizon.

    `groups` optionally labels each sequence (subject, activity, scene ...);
    when given, the report also breaks the median down by group, which is how
    you tell "this holds across the dataset" from "one activity drives it".
    """
    seqs = _as_tensor(poses)
    if groups is not None and len(groups) != len(seqs):
        raise ValueError(f"groups has {len(groups)} entries for {len(seqs)} sequences")

    rng = np.random.default_rng(seed)
    report = {"name": name, "fps": fps, "n_sequences": len(seqs), "horizons": {}}

    for k in horizons:
        cur, fut, grp = [], [], []
        for i, s in enumerate(seqs):
            if s.shape[0] <= k:
                continue
            cur.append(s[:-k])
            fut.append(s[k:])
            if groups is not None:
                grp.extend([groups[i]] * (s.shape[0] - k))
        if not cur:
            continue
        pose_t = torch.cat(cur)
        pose_future = torch.cat(fut)
        grp = np.asarray(grp) if groups is not None else None

        # Non-finite coordinates would abort the BATCHED Kabsch SVD for the
        # whole run with an opaque LinAlgError -- and hand_frame_degenerate
        # does not catch NaN (every comparison on NaN is False, so the sample
        # would be KEPT). This project's loaders filter non-finite frames, but
        # this file is the piece meant for third-party arrays, so the input
        # contract cannot assume that. Dropped and counted, never trusted.
        finite = torch.isfinite(pose_t).flatten(1).all(dim=1) & \
                 torch.isfinite(pose_future).flatten(1).all(dim=1)
        n_nonfinite = int((~finite).sum())
        if n_nonfinite:
            pose_t, pose_future = pose_t[finite], pose_future[finite]
            if grp is not None:
                grp = grp[finite.numpy()]
        if pose_t.shape[0] == 0:
            continue

        # Degenerate hands (collinear/collapsed joints) have no well-defined
        # palm frame; excluded rather than allowed to emit NaNs.
        good = ~hand_frame_degenerate(pose_t)
        pose_t, pose_future = pose_t[good], pose_future[good]
        if grp is not None:
            grp = grp[good.numpy()]

        if pose_t.shape[0] > max_pairs_per_horizon:
            sel = rng.choice(pose_t.shape[0], max_pairs_per_horizon, replace=False)
            sel_t = torch.as_tensor(np.sort(sel))
            pose_t, pose_future = pose_t[sel_t], pose_future[sel_t]
            if grp is not None:
                grp = grp[np.sort(sel)]

        share = rigid_fraction(pose_t, pose_future).numpy()
        entry = {
            "n_pairs": int(share.size),
            "n_nonfinite_dropped": n_nonfinite,
            "median_rigid_share": float(np.median(share)),
            "mean_rigid_share": float(share.mean()),
            "frac_above_80pct": float((share > 0.8).mean()),
            "frac_above_90pct": float((share > 0.9).mean()),
            "ms_ahead": (1000.0 * k / fps) if fps else None,
        }
        if grp is not None:
            entry["by_group"] = {
                str(g): {"n": int((grp == g).sum()),
                         "median_rigid_share": float(np.median(share[grp == g]))}
                for g in sorted(set(grp.tolist()))
            }
        report["horizons"][str(k)] = entry
    return report


def _print(report):
    print("\n=== ROTATION SHARE OF THE WRIST-RELATIVE DELTA ===")
    print("dataset: %s   sequences: %d   fps: %s"
          % (report.get("name") or "(unnamed)", report["n_sequences"], report.get("fps")))
    print("\nHow much of the standard 'articulation' target is actually whole-hand rotation.\n")
    print("%-5s %10s %14s %12s %12s" % ("k", "n pairs", "median share", ">80% rigid", ">90% rigid"))
    for k, e in report["horizons"].items():
        print("%-5s %10d %13.1f%% %11.1f%% %11.1f%%"
              % (k, e["n_pairs"], 100 * e["median_rigid_share"],
                 100 * e["frac_above_80pct"], 100 * e["frac_above_90pct"]))
    for k, e in report["horizons"].items():
        if "by_group" in e:
            print("\n  k=%s by group (median rigid share):" % k)
            for g, v in sorted(e["by_group"].items(), key=lambda x: -x[1]["median_rigid_share"]):
                print("    %-34s n=%-8d %.1f%%" % (g[:34], v["n"], 100 * v["median_rigid_share"]))
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True, help="(N,T,21,3) array of hand poses, joint 0 = wrist")
    ap.add_argument("--groups-npy", default=None, help="optional (N,) labels per sequence")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    poses = np.load(args.npy, allow_pickle=True)
    groups = np.load(args.groups_npy, allow_pickle=True).tolist() if args.groups_npy else None
    report = rotation_share_report(poses, horizons=args.horizons, fps=args.fps,
                                   name=args.name or os.path.basename(args.npy), groups=groups)
    _print(report)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
