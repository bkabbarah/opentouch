"""
Read results_log.csv (append-only, long format) and produce a wide
comparison table: one row per retrieval direction, one column per run.

Uses only the MOST RECENT entry for each (run_label, direction) pair,
so if you've logged the same run_label multiple times (e.g. after a
rerun), only the latest numbers show up here -- but nothing is deleted
from results_log.csv itself, the full history stays intact there.

Usage:
    python pivot_results.py
    python pivot_results.py --runs avgpool_single_p2t gru_single_p2t   # subset of columns
    python pivot_results.py --csv-out comparison_table.csv             # also write CSV
"""

import csv
import argparse
from collections import defaultdict

LOG_PATH = "/home/bashark/scratch/bashar/results_log.csv"

DIRECTION_ORDER = [
    "visual_to_tactile_mAP", "tactile_to_visual_mAP",
    "pose_to_tactile_mAP", "tactile_to_pose_mAP",
    "visual_to_pose_mAP", "pose_to_visual_mAP",
    "visual_pose_to_tactile_mAP", "tactile_to_visual_pose_mAP",
    "tactile_pose_to_visual_mAP", "visual_to_tactile_pose_mAP",
    "visual_tactile_to_pose_mAP", "pose_to_visual_tactile_mAP",
]
DIRECTION_LABELS = {
    "visual_to_tactile_mAP": "V -> T",
    "tactile_to_visual_mAP": "T -> V",
    "pose_to_tactile_mAP": "P -> T",
    "tactile_to_pose_mAP": "T -> P",
    "visual_to_pose_mAP": "V -> P",
    "pose_to_visual_mAP": "P -> V",
    "visual_pose_to_tactile_mAP": "VP -> T",
    "tactile_to_visual_pose_mAP": "T -> VP",
    "tactile_pose_to_visual_mAP": "TP -> V",
    "visual_to_tactile_pose_mAP": "V -> TP",
    "visual_tactile_to_pose_mAP": "VT -> P",
    "pose_to_visual_tactile_mAP": "P -> VT",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None,
                         help="Subset of run_labels to include as columns, in order. Default: all, in first-seen order.")
    parser.add_argument("--csv-out", default=None, help="Optional path to also write the table as CSV")
    args = parser.parse_args()

    # latest[(run_label, direction)] = (logged_at, mAP_pct)
    latest = {}
    run_order = []
    with open(LOG_PATH) as f:
        for row in csv.DictReader(f):
            key = (row["run_label"], row["direction"])
            if key not in latest or row["logged_at"] > latest[key][0]:
                latest[key] = (row["logged_at"], row["mAP_pct"])
            if row["run_label"] not in run_order:
                run_order.append(row["run_label"])

    runs = args.runs if args.runs else run_order

    # print table
    col_width = 12
    header = f"{'direction':18s}" + "".join(f"{r[:col_width]:>{col_width+2}s}" for r in runs)
    print(header)
    print("-" * len(header))
    table_rows = []
    for direction in DIRECTION_ORDER:
        label = DIRECTION_LABELS[direction]
        line = f"{label:18s}"
        row_vals = [label]
        for run in runs:
            val = latest.get((run, direction), (None, ""))[1]
            line += f"{val:>{col_width+2}s}"
            row_vals.append(val)
        print(line)
        table_rows.append(row_vals)

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["direction"] + runs)
            writer.writerows(table_rows)
        print(f"\nWrote {args.csv_out}")


if __name__ == "__main__":
    main()
