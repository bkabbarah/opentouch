"""Summarize the participant-disjoint forecasting sweep (HANDOFF open #1).

Reads each sf_* run's out.log, takes the FINAL epoch's rigid-target moving
fingertip MSE, and builds the arm comparison with seed spread.

Two guards, because both failure modes are silent:

  1. The motion threshold is auto-computed from the train split. All arms at a
     given horizon must land on the same value or the "moving subset" differs
     between them and the MSEs are not comparable.
  2. Every run must record split_group_by=scene in params.txt. A run that
     silently fell back to a clip split would look like a valid
     unseen-participant result while being nothing of the kind.

Usage:
    python scripts/summarize_scene_forecast.py --logs logs --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
from collections import defaultdict

ARMS = ("frz", "frzshuf", "pose")
ARM_LABEL = {
    "frz": "frozen tactile + pose",
    "frzshuf": "frozen shuffled + pose",
    "pose": "pose-only",
}

_RIGID = re.compile(
    r"\[rigid\][^\n]*moving mse_fingertips: ([0-9.]+)\s+copy_baseline: ([0-9.]+)"
)
_THRESH = re.compile(r"threshold=([0-9.]+)")
_PARAM = re.compile(r"^\s*([a-z_]+):\s*(.*?)\s*$", re.MULTILINE)


def read_run(run_dir):
    log_path = os.path.join(run_dir, "out.log")
    par_path = os.path.join(run_dir, "params.txt")
    if not os.path.exists(log_path):
        return None
    text = open(log_path, errors="replace").read()
    rigid = _RIGID.findall(text)
    if not rigid:
        return None
    thresh = _THRESH.findall(text)
    params = dict(_PARAM.findall(open(par_path, errors="replace").read())) if os.path.exists(par_path) else {}
    return {
        "mse": float(rigid[-1][0]),
        "copy": float(rigid[-1][1]),
        "threshold": float(thresh[-1]) if thresh else None,
        "split_group_by": params.get("split_group_by"),
        "tactile_init_checkpoint": params.get("tactile_init_checkpoint"),
        "trainable": params.get("trainable_parameters"),
        "epochs": params.get("epochs"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="results_scene_forecast.json")
    ap.add_argument("--prefix", default="sf_")
    args = ap.parse_args()

    runs = {}
    for name in sorted(os.listdir(args.logs)):
        m = re.match(rf"{re.escape(args.prefix)}([a-z]+)_k(\d+)_s(\d+)$", name)
        if not m:
            continue
        info = read_run(os.path.join(args.logs, name))
        if info is None:
            print(f"  (incomplete, skipped) {name}")
            continue
        info.update(arm=m.group(1), k=int(m.group(2)), seed=int(m.group(3)), name=name)
        runs[name] = info

    if not runs:
        print("No completed sf_* runs found.")
        return

    problems = []
    for name, r in runs.items():
        if r["split_group_by"] != "scene":
            problems.append(f"{name}: split_group_by={r['split_group_by']!r}, expected 'scene'")
        if r["arm"] in ("frz", "frzshuf") and not (r["tactile_init_checkpoint"] or "").startswith("logs/p2t_scene_gru"):
            problems.append(f"{name}: encoder is {r['tactile_init_checkpoint']!r}, expected p2t_scene_gru")

    by_k_thresh = defaultdict(set)
    for r in runs.values():
        if r["threshold"] is not None:
            by_k_thresh[r["k"]].add(round(r["threshold"], 12))
    for k, vals in sorted(by_k_thresh.items()):
        if len(vals) > 1:
            problems.append(f"k={k}: arms disagree on motion threshold {sorted(vals)} -- MSEs not comparable")

    grouped = defaultdict(list)
    for r in runs.values():
        grouped[(r["arm"], r["k"])].append(r["mse"])

    summary = {}
    for (arm, k), vals in grouped.items():
        summary[f"{arm}_k{k}"] = {
            "arm": arm, "horizon_k": k, "n_seeds": len(vals),
            "mean": st.mean(vals),
            "std": st.stdev(vals) if len(vals) > 1 else 0.0,
            "values": sorted(vals),
        }

    print("\n=== Participant-disjoint forecasting: moving fingertip MSE (rigid target) ===")
    print("Lower is better. Encoder and split are both scene-disjoint.\n")
    ks = sorted({r["k"] for r in runs.values()})
    header = "| condition | " + " | ".join(f"k={k}" for k in ks) + " |"
    print(header)
    print("|---" * (len(ks) + 1) + "|")
    copy_by_k = {}
    for r in runs.values():
        copy_by_k[r["k"]] = r["copy"]
    for arm in ARMS:
        cells = []
        for k in ks:
            s = summary.get(f"{arm}_k{k}")
            cells.append(f"{s['mean']:.6f} (n={s['n_seeds']}, sd={s['std']:.6f})" if s else "--")
        print(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    print("| copy-zero | " + " | ".join(f"{copy_by_k.get(k, float('nan')):.6f}" for k in ks) + " |")

    print("\n=== Comparisons (negative = touch helps) ===")
    comparisons = {}
    for k in ks:
        frz = summary.get(f"frz_k{k}")
        for other in ("pose", "frzshuf"):
            o = summary.get(f"{other}_k{k}")
            if not (frz and o):
                continue
            pct = (frz["mean"] - o["mean"]) / o["mean"] * 100.0
            key = f"frz_vs_{other}_k{k}"
            comparisons[key] = pct
            label = "pose-only" if other == "pose" else "shuffled (capacity-matched)"
            print(f"  k={k:<3} frozen tactile vs {label:<28} {pct:+.1f}%")
        if frz and k in copy_by_k:
            pct = (frz["mean"] - copy_by_k[k]) / copy_by_k[k] * 100.0
            comparisons[f"frz_vs_copy_k{k}"] = pct
            print(f"  k={k:<3} frozen tactile vs {'copy-zero':<28} {pct:+.1f}%")
        fs, po = summary.get(f"frzshuf_k{k}"), summary.get(f"pose_k{k}")
        if fs and po:
            pct = (fs["mean"] - po["mean"]) / po["mean"] * 100.0
            comparisons[f"frzshuf_vs_pose_k{k}"] = pct
            print(f"  k={k:<3} frozen shuffled vs {'pose-only (branch cost)':<27} {pct:+.1f}%")

    print()
    if problems:
        print("!!! INTEGRITY PROBLEMS -- do not quote these numbers:")
        for p in problems:
            print("   ", p)
    else:
        print("Integrity checks passed: all runs scene-split, encoder from "
              "p2t_scene_gru, motion threshold consistent within each horizon.")

    payload = {
        "experiment": "participant-disjoint frozen-encoder forecasting (HANDOFF open #1)",
        "split_group_by": "scene",
        "encoder": "logs/p2t_scene_gru/checkpoints/epoch_300.pt",
        "n_runs": len(runs),
        "summary": summary,
        "comparisons_pct": comparisons,
        "copy_baseline_by_k": copy_by_k,
        "motion_threshold_by_k": {str(k): sorted(v) for k, v in by_k_thresh.items()},
        "integrity_problems": problems,
        "runs": runs,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
