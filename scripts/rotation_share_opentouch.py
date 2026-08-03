"""Run the dataset-agnostic rotation diagnostic on OpenTouch, split by SCENE.

Two purposes.

VALIDATION. scripts/rotation_share.py is meant to run on any dataset, so it
must first reproduce this project's own published figure -- ~95.7% median rigid
share at k=8 (HANDOFF 2.7) -- on the data that figure came from. If the
agnostic path disagrees with scripts/rigid_diag.py, the agnostic path is wrong
and nothing it says about another dataset can be trusted.

GENERALIZATION, the cheap kind. The claim "the standard target is mostly
wrist rotation" is currently one number over one dataset. Breaking it down by
scene -- 26 recording sessions spanning grocery aisles, an office, a fab lab
and eating -- shows whether it is a property of hand motion or an artifact of
one activity. That is not a substitute for a second dataset, but it costs
nothing and it is the difference between "we measured this once" and "this
holds across every activity we have".

    python scripts/rotation_share_opentouch.py --data <dataset> --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opentouch_train.data import VideoTactilePoseDataset, _load_and_split_dataset
from rotation_share import rotation_share_report, _print


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="All three by default: this is a property of the data, "
                         "not a model evaluation, so there is nothing to hold out.")
    ap.add_argument("--sequence-length", type=int, default=36)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default="results_rotation_share_opentouch.json")
    args = ap.parse_args()

    parts = _load_and_split_dataset(args.data, 0.1, 0.1, args.split_seed)

    # One "sequence" per window. Windows within a clip overlap, which inflates
    # the PAIR count but not the median -- and the median is what is reported.
    by_scene = defaultdict(list)
    for split in args.splits:
        if split not in parts:
            continue
        base = VideoTactilePoseDataset(
            split=split, _preloaded=parts[split], hf_dataset_path=args.data,
            sequence_length=args.sequence_length,
            include_tactile=False, include_visual=False, include_pose=True,
        )
        for idx in range(len(base)):
            sample = base[idx]
            pose = sample["hand_landmarks"]
            if pose.dim() == 4:          # (T,1,21,3) -> (T,21,3)
                pose = pose.squeeze(1)
            by_scene[sample["scene"]].append(pose.float())

    scenes = sorted(by_scene)
    seqs, groups = [], []
    for sc in scenes:
        seqs.extend(by_scene[sc])
        groups.extend([sc] * len(by_scene[sc]))
    print("Loaded %d windows across %d scenes" % (len(seqs), len(scenes)))

    report = rotation_share_report(
        seqs, horizons=args.horizons, fps=args.fps,
        name="OpenTouch (%s)" % "+".join(args.splits), groups=groups,
    )
    _print(report)

    k8 = report["horizons"].get("8", {}).get("median_rigid_share")
    if k8 is not None:
        print("\nVALIDATION against HANDOFF 2.7 (rigid_diag.py reported 0.957 at k=8):")
        print("  agnostic path gives %.3f  ->  %s"
              % (k8, "AGREES" if abs(k8 - 0.957) < 0.02 else "DISAGREES -- do not trust cross-dataset runs"))

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
