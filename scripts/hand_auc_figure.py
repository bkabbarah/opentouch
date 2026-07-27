"""Hand diagram coloured by per-joint AUC.

Takes export_per_joint.py's output and produces a figure-ready hand: an SVG
with the 21 keypoints laid out anatomically and shaded by whatever per-joint
quantity you pick, plus a flat CSV if you would rather draw it yourself.

TWO CHOICES THE FIGURE FORCES, both exposed rather than silently made:

  --metric decides WHAT is coloured.
    auc_tactile          how well touch alone decodes that joint. Answers
                         "where does touch read the hand?"
    marginal_vs_pose     how much touch adds over a pose encoder with matched
                         temporal access. Answers "where does touch tell you
                         something kinematics do not?" -- usually the more
                         interesting claim, and the one the paper argues.
    marginal_vs_raw_pose the same against full raw kinematics, if the export
                         was run with --include-raw-pose. Strongest version.

  --axis decides WHICH component. 'mean' averages the three palm axes.
    Naming a single axis (radial / spread / curl) is sharper but is a choice
    the reader cannot see from the figure, so state it in the caption.

THE WRIST IS NOT COLOURED. Articulation is defined wrist-relative, so the
wrist row of the target is identically zero and has no direction to decode.
It is drawn as an open circle, never shaded, because giving it a colour would
imply a measurement that does not exist.

Joint positions come from the dataset's own mean hand pose projected onto the
palm plane, so the diagram is the anatomy actually measured rather than a
hand-drawn schematic.
"""

import argparse
import csv
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from opentouch.articulation_frames import hand_frame_basis, hand_frame_degenerate
from opentouch.pose_regression import COORD_DIM, NUM_KEYPOINTS, WRIST_INDEX
from opentouch_train.data import VideoTactilePoseDataset, _load_and_split_dataset
from tactile_direction_probe import JOINT_LABELS

FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

# Perceptually ordered, colour-blind safe (viridis endpoints), light to dark.
COLOURS = ["#fde725", "#7ad151", "#22a884", "#2a788e", "#414487", "#440154"]


def canonical_layout(data_path, split_seed, sequence_length):
    """(21,2) mean hand in palm coordinates: x along the palm, y across it.

    Averaging in the PALM frame rather than the world frame is what makes this
    meaningful -- world-frame averaging over hands at arbitrary orientations
    collapses toward the centroid and produces a blob, not a hand.
    """
    splits = _load_and_split_dataset(data_path, 0.1, 0.1, split_seed)
    dataset = VideoTactilePoseDataset(
        split="val", _preloaded=splits["val"], hf_dataset_path=data_path,
        sequence_length=sequence_length, include_tactile=False,
        include_visual=False, include_pose=True,
    )
    poses = torch.stack([dataset[i]["hand_landmarks"].squeeze(1)[0] for i in range(len(dataset))])
    poses = poses[~hand_frame_degenerate(poses)]
    centered = poses - poses[:, WRIST_INDEX : WRIST_INDEX + 1, :]
    local = torch.einsum("bij,bkj->bki", hand_frame_basis(poses), centered)
    mean_local = local.mean(dim=0).numpy()
    # radial across the page, spread up it; drop the palm normal.
    return np.stack([mean_local[:, 0], mean_local[:, 1]], axis=1)


def per_joint_values(rows, metric, axis):
    values = {}
    for label in JOINT_LABELS:
        if label == JOINT_LABELS[WRIST_INDEX]:
            continue
        matching = [r for r in rows if r["joint"] == label and (axis == "mean" or r["axis"] == axis)]
        matching = [r for r in matching if metric in r]
        if matching:
            values[label] = float(np.mean([r[metric] for r in matching]))
    return values


def to_svg(layout, values, metric, axis, width=560, height=680, margin=70):
    finite = [v for v in values.values() if np.isfinite(v)]
    low, high = min(finite), max(finite)
    span = (high - low) or 1.0

    xs, ys = layout[:, 0], layout[:, 1]
    # Palm long axis runs up the page, so map radial -> screen y (inverted).
    def project(index):
        x = margin + (ys[index] - ys.min()) / ((ys.max() - ys.min()) or 1) * (width - 2 * margin)
        y = height - margin - (xs[index] - xs.min()) / ((xs.max() - xs.min()) or 1) * (height - 2 * margin)
        return x, y

    def shade(value):
        position = (value - low) / span
        return COLOURS[min(int(position * len(COLOURS)), len(COLOURS) - 1)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    for joints in FINGERS.values():
        chain = [WRIST_INDEX] + joints
        points = " ".join("%.1f,%.1f" % project(j) for j in chain)
        parts.append(f'<polyline points="{points}" fill="none" stroke="#c8c8c8" stroke-width="6" '
                     'stroke-linecap="round"/>')

    for index, label in enumerate(JOINT_LABELS):
        x, y = project(index)
        if index == WRIST_INDEX or label not in values:
            # Never shaded: articulation is wrist-relative, so the wrist has no
            # direction to decode and a colour would imply a measurement.
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="white" '
                         'stroke="#999" stroke-width="2" stroke-dasharray="3,3"/>')
            continue
        value = values[label]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" fill="{shade(value)}" '
                     'stroke="#333" stroke-width="1.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{y + 30:.1f}" font-size="10" fill="#444" '
                     f'text-anchor="middle">{value:.3f}</text>')

    parts.append(f'<text x="{width/2:.0f}" y="30" font-size="15" text-anchor="middle" '
                 f'fill="#222">{metric} ({axis} axis)</text>')
    bar_x, bar_y, bar_w = margin, height - 34, width - 2 * margin
    for i, colour in enumerate(COLOURS):
        parts.append(f'<rect x="{bar_x + i * bar_w / len(COLOURS):.1f}" y="{bar_y}" '
                     f'width="{bar_w / len(COLOURS):.1f}" height="14" fill="{colour}"/>')
    parts.append(f'<text x="{bar_x}" y="{bar_y - 5}" font-size="11" fill="#444">{low:.3f}</text>')
    parts.append(f'<text x="{bar_x + bar_w:.0f}" y="{bar_y - 5}" font-size="11" fill="#444" '
                 f'text-anchor="end">{high:.3f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-joint-json", required=True, help="export_per_joint.py output")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=36)
    parser.add_argument("--metric", default="marginal_vs_pose")
    parser.add_argument("--axis", default="mean", choices=["mean", "radial", "spread", "curl"])
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    with open(args.per_joint_json) as handle:
        report = json.load(handle)
    rows = report["per_joint"]
    available = sorted({k for r in rows for k in r if k.startswith(("auc_", "marginal_"))})
    if args.metric not in available:
        raise SystemExit("--metric %r not in this export. Available: %s" % (args.metric, available))

    values = per_joint_values(rows, args.metric, args.axis)
    if not values:
        raise SystemExit("no joints matched --axis %r" % args.axis)

    layout = canonical_layout(args.data, args.split_seed, args.sequence_length)
    with open(args.output_prefix + ".svg", "w") as handle:
        handle.write(to_svg(layout, values, args.metric, args.axis))
    with open(args.output_prefix + ".csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["joint", "keypoint_index", "value", "layout_x", "layout_y"])
        for index, label in enumerate(JOINT_LABELS):
            if label in values:
                writer.writerow([label, index, "%.6f" % values[label],
                                 "%.6f" % layout[index, 0], "%.6f" % layout[index, 1]])

    ordered = sorted(values.items(), key=lambda kv: -kv[1])
    print("metric=%s axis=%s   range %.4f to %.4f" % (
        args.metric, args.axis, min(values.values()), max(values.values())))
    print("highest:", ", ".join("%s %.4f" % (k, v) for k, v in ordered[:5]))
    print("lowest: ", ", ".join("%s %.4f" % (k, v) for k, v in ordered[-5:]))
    print("wrote %s.svg and %s.csv" % (args.output_prefix, args.output_prefix))


if __name__ == "__main__":
    main()
