"""Do the palm-frame axis LABELS mean what they are named?

articulation_frames builds an orthonormal, hand-relative, causally-computed
basis and calls its rows (long, flex, normal). The orthonormality, the
world-frame invariance and the causality are all unit-tested. What is NOT
tested is the anatomical claim in the names: that `flex` really is the
direction fingers curl toward the palm, and that `long` really runs along the
palm.

Those names rest on the 21-keypoint layout (wrist=0, then five MCP/PIP/DIP/TIP
blocks, so index MCP=5, middle MCP=9, pinky MCP=17), which the codebase
asserts but never checks against the data. This script checks it against the
data, three ways:

  1. GEOMETRY. For a closed hand, each fingertip should sit on the palm side
     of its MCP. Project (tip - mcp) onto `flex` and check the sign flips the
     way curling predicts as the hand closes.
  2. MOTION. When a hand closes (fingertips approach the wrist), the residual
     articulation should carry a systematically negative `flex` component if
     `flex` points away from the palm, positive if toward it. Either is fine;
     what matters is that the relationship is STRONG, since a mislabeled or
     scrambled axis would show no consistent relation.
  3. SPREAD. Finger abduction (fingers fanning apart) should live mainly in
     the plane spanned by `long` and `normal`, not along `flex`.

A wrong keypoint layout would break all three at once.
"""
import argparse, json
import numpy as np, torch

from opentouch.articulation_frames import (
    all_target_variants, hand_frame_basis, hand_frame_degenerate, to_hand_frame)
from opentouch.pose_regression import WRIST_INDEX
from opentouch_train.data import VideoTactilePoseDataset, _load_and_split_dataset
from opentouch_train.regression_data import PoseTransitionDataset

TIPS = [4, 8, 12, 16, 20]
MCPS = [1, 5, 9, 13, 17]

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--split", default="val")
p.add_argument("--sequence-length", type=int, default=20)
p.add_argument("--horizon-k", type=int, default=8)
p.add_argument("--split-seed", type=int, default=42)
p.add_argument("--output", default=None)
a = p.parse_args()

splits = _load_and_split_dataset(a.data, 0.1, 0.1, a.split_seed)
base = VideoTactilePoseDataset(split=a.split, _preloaded=splits[a.split], hf_dataset_path=a.data,
    sequence_length=a.sequence_length, include_tactile=False, include_visual=False, include_pose=True)
ds = PoseTransitionDataset(base, a.horizon_k, shuffle_tactile=False, causal=False)
pose = ds._pose
vt = ds.valid_t_per_window
pt = pose[:, :vt].reshape(-1, 21, 3)
pf = pose[:, a.horizon_k:].reshape(-1, 21, 3)
good = ~hand_frame_degenerate(pt)
pt, pf = pt[good], pf[good]
print(f"n={pt.shape[0]}")

basis = hand_frame_basis(pt)
centered = pt - pt[:, WRIST_INDEX:WRIST_INDEX + 1, :]
local = to_hand_frame(centered, basis)          # (N,21,3) static pose in palm axes
variants = all_target_variants(pt, pf)
resid = variants["rigid_removed_handframe"]      # (N,21,3) non-rigid motion in palm axes

report = {}

# --- 1. GEOMETRY: where do fingertips sit relative to their MCPs?
tip_minus_mcp = local[:, TIPS] - local[:, MCPS]   # (N,5,3)
axis_share = tip_minus_mcp.pow(2).mean(dim=(0, 1))
axis_share = (axis_share / axis_share.sum()).tolist()
mean_flex = tip_minus_mcp[:, :, 1].mean().item()
report["geometry"] = {
    "energy_share_long_flex_normal": axis_share,
    "mean_flex_component_of_tip_minus_mcp": mean_flex,
}
print(f"\n1. GEOMETRY  (tip - mcp) in palm axes")
print(f"   energy share long/flex/normal: {axis_share[0]:.3f} / {axis_share[1]:.3f} / {axis_share[2]:.3f}")
print(f"   mean flex component: {mean_flex:+.4f}  "
      f"({'fingers sit on +flex side' if mean_flex > 0 else 'fingers sit on -flex side'})")

# --- 2. MOTION: does hand closing move fingers along `flex`?
grip = (pt[:, TIPS] - pt[:, WRIST_INDEX:WRIST_INDEX + 1]).norm(dim=-1).mean(dim=1)
grip_future = (pf[:, TIPS] - pf[:, WRIST_INDEX:WRIST_INDEX + 1]).norm(dim=-1).mean(dim=1)
closing = (grip_future - grip)                     # negative = hand closing
tip_resid = resid[:, TIPS].mean(dim=1)             # (N,3) mean fingertip residual, palm axes
corrs = []
for i, name in enumerate(("long", "flex", "normal")):
    c = np.corrcoef(closing.numpy(), tip_resid[:, i].numpy())[0, 1]
    corrs.append(float(c))
    print(f"   corr(closing, {name:<6}) = {c:+.3f}")
report["motion_corr_long_flex_normal"] = corrs
print(f"\n2. MOTION  correlation of grip open/close with residual per axis")
best = int(np.argmax(np.abs(corrs)))
print(f"   strongest: {('long','flex','normal')[best]} (|r|={abs(corrs[best]):.3f})")

# --- 3. SPREAD: finger abduction should avoid `flex`
spread = (pt[:, 8] - pt[:, 20]).norm(dim=-1)       # index tip to pinky tip
spread_future = (pf[:, 8] - pf[:, 20]).norm(dim=-1)
fanning = (spread_future - spread)
spread_corrs = []
for i, name in enumerate(("long", "flex", "normal")):
    c = np.corrcoef(fanning.numpy(), (resid[:, 8, i] - resid[:, 20, i]).numpy())[0, 1]
    spread_corrs.append(float(c))
report["spread_corr_long_flex_normal"] = spread_corrs
print(f"\n3. SPREAD  correlation of finger fanning with index-minus-pinky residual")
for name, c in zip(("long", "flex", "normal"), spread_corrs):
    print(f"   corr(fanning, {name:<6}) = {c:+.3f}")

verdict = (abs(corrs[1]) > 0.3) and (axis_share[1] > 0.15)
report["labels_behave_anatomically"] = bool(verdict)
print(f"\nVERDICT: flex axis behaves like a flexion axis: {verdict}")
print("  (requires |corr(closing, flex)| > 0.3 and flex carrying >15% of tip-MCP geometry)")

if a.output:
    json.dump(report, open(a.output, "w"), indent=2)
    print(f"wrote {a.output}")
