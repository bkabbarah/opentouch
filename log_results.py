"""
Append OpenTouch eval results to a permanent, append-only log with full
provenance (git commit, checkpoint path, epoch, task type, timestamp).

This NEVER overwrites past entries. Every time you eval a checkpoint and
have a results_*.json, run this script and it adds one row per retrieval
direction to results_log.csv, tagged with everything needed to know
exactly what produced those numbers later.

Usage (run from the repo dir containing the results json, e.g.
~/scratch/bashar/opentouch-gru or ~/scratch/bashar/opentouch):

    python log_results.py <results_json_path> <run_label> [--notes "free text"]

Example:
    python log_results.py results_multitask_gru.json gru_joint --notes "GRU pose encoder, linear fusion, all 6 tasks jointly"

Each call appends new rows. results_log.csv grows over time and is the
single source of truth. Nothing is ever silently overwritten.
"""

import json
import csv
import os
import sys
import subprocess
import argparse
from datetime import datetime

LOG_PATH = os.path.expanduser("~/scratch/bashar/results_log.csv")

DIRECTIONS = [
    "visual_to_tactile_mAP", "tactile_to_visual_mAP",
    "pose_to_tactile_mAP", "tactile_to_pose_mAP",
    "visual_to_pose_mAP", "pose_to_visual_mAP",
    "visual_pose_to_tactile_mAP", "tactile_to_visual_pose_mAP",
    "tactile_pose_to_visual_mAP", "visual_to_tactile_pose_mAP",
    "visual_tactile_to_pose_mAP", "pose_to_visual_tactile_mAP",
]


def get_git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        ) != 0
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def get_repo_path():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return os.getcwd()


def find_checkpoint_meta(results_json_path):
    """Try to recover checkpoint path / epoch / task from the eval script's
    own stdout convention -- the json itself doesn't store this, so we
    fall back to asking the user via CLI args if not inferable."""
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", help="Path to results_*.json from eval.py")
    parser.add_argument("run_label", help="Short identifier, e.g. gru_joint, gru_single_p2t")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path used (optional but recommended)")
    parser.add_argument("--epoch", default="", help="Epoch number (optional)")
    parser.add_argument("--task-type", default="", help="Task type trained on, e.g. p2t, all")
    parser.add_argument("--notes", default="", help="Free text description of what this run is")
    args = parser.parse_args()

    if not os.path.exists(args.results_json):
        print(f"ERROR: {args.results_json} not found")
        sys.exit(1)

    with open(args.results_json) as f:
        data = json.load(f)

    commit = get_git_commit()
    repo = get_repo_path()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow([
                "logged_at", "run_label", "direction", "mAP_pct",
                "git_commit", "repo_path", "checkpoint", "epoch",
                "task_type", "results_json_path", "notes",
            ])
        rows_written = 0
        for key in DIRECTIONS:
            if key in data:
                writer.writerow([
                    timestamp, args.run_label, key, round(data[key] * 100, 2),
                    commit, repo, args.checkpoint, args.epoch,
                    args.task_type, os.path.abspath(args.results_json), args.notes,
                ])
                rows_written += 1

    print(f"Appended {rows_written} rows to {LOG_PATH}")
    print(f"  run_label   : {args.run_label}")
    print(f"  git_commit  : {commit}")
    print(f"  repo        : {repo}")
    if args.checkpoint:
        print(f"  checkpoint  : {args.checkpoint}")
    print(f"This entry is permanent -- rerunning this script with the same")
    print(f"run_label will ADD a new timestamped entry, not replace this one.")


if __name__ == "__main__":
    main()
