"""How much of the 'articulation delta' was actually whole-hand ROTATION?

Loads the val split's pose windows, builds the k-step delta the probe uses,
and reports (a) the share of that delta's energy the best rigid rotation about
the wrist explains, and (b) where the per-axis motion energy sits under the
current world-axis target versus the rigid-removed palm-frame target.

No checkpoint, no encoder -- pose only.
"""
import argparse, json
import numpy as np, torch

from opentouch.articulation_frames import (
    ARTICULATING_JOINTS, all_target_variants, hand_frame_degenerate, rigid_fraction,
)
from opentouch_train.data import VideoTactilePoseDataset, _load_and_split_dataset
from opentouch_train.regression_data import PoseTransitionDataset

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--split", default="val")
p.add_argument("--sequence-length", type=int, default=20)
p.add_argument("--split-seed", type=int, default=42)
p.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8, 16])
p.add_argument("--output", default=None)
a = p.parse_args()

splits = _load_and_split_dataset(a.data, 0.1, 0.1, a.split_seed)
base = VideoTactilePoseDataset(
    split=a.split, _preloaded=splits[a.split], hf_dataset_path=a.data,
    sequence_length=a.sequence_length, include_tactile=False,
    include_visual=False, include_pose=True,
)
report = {"split": a.split, "sequence_length": a.sequence_length, "horizons": {}}

for k in a.horizons:
    ds = PoseTransitionDataset(base, k, shuffle_tactile=False, causal=False)
    pose = ds._pose
    valid_t = ds.valid_t_per_window
    pose_t = pose[:, :valid_t].reshape(-1, 21, 3)
    pose_future = pose[:, k:].reshape(-1, 21, 3)

    good = ~hand_frame_degenerate(pose_t)
    pose_t, pose_future = pose_t[good], pose_future[good]

    share = rigid_fraction(pose_t, pose_future)
    v = all_target_variants(pose_t, pose_future)

    def axis_energy(delta):
        e = delta[:, ARTICULATING_JOINTS].pow(2).sum(dim=1).mean(dim=0)
        return (e / e.sum()).tolist()

    cur = axis_energy(v["wrist_translation_removed"])
    new = axis_energy(v["rigid_removed_handframe"])
    entry = {
        "n_samples": int(pose_t.shape[0]),
        "rigid_fraction_median": float(share.median()),
        "rigid_fraction_mean": float(share.mean()),
        "rigid_fraction_p25": float(share.quantile(0.25)),
        "rigid_fraction_p75": float(share.quantile(0.75)),
        "frac_samples_over_50pct_rigid": float((share > 0.5).float().mean()),
        "frac_samples_over_80pct_rigid": float((share > 0.8).float().mean()),
        "axis_energy_current_target_xyz": cur,
        "axis_energy_rigid_removed_handframe_long_flex_normal": new,
    }
    report["horizons"][str(k)] = entry
    print(f"\n=== k={k}  n={entry['n_samples']}")
    print(f"  rigid-explained share of 'articulation' energy: "
          f"median {entry['rigid_fraction_median']:.3f}  mean {entry['rigid_fraction_mean']:.3f}  "
          f"IQR [{entry['rigid_fraction_p25']:.3f}, {entry['rigid_fraction_p75']:.3f}]")
    print(f"  samples >50% rigid: {entry['frac_samples_over_50pct_rigid']:.1%}   "
          f">80% rigid: {entry['frac_samples_over_80pct_rigid']:.1%}")
    print(f"  energy share, CURRENT target (world x,y,z):      "
          f"{cur[0]:.3f} / {cur[1]:.3f} / {cur[2]:.3f}")
    print(f"  energy share, rigid-removed palm (long,flex,nrm): "
          f"{new[0]:.3f} / {new[1]:.3f} / {new[2]:.3f}")

if a.output:
    json.dump(report, open(a.output, "w"), indent=2)
    print(f"\nwrote {a.output}")
