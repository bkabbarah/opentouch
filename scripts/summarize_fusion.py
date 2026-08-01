"""Summarize the fusion sweep: does FiLM beat the scalar gate?

Reads fu_{fusion}_{arm}_k8_s{seed} final metrics and reports the
frz-vs-frzshuf contrast within each fusion, plus each arm against the shared
pose-only baseline (sf_pose_k8_s*, which needs no fusion variant).

The within-fusion contrast is the capacity-matched one. Across fusions the
trainable counts differ (66,045 gate vs 74,430 film), so a film-vs-gate
comparison of raw MSE is informative but not capacity-controlled -- stated in
the output rather than left for the reader to assume.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
from collections import defaultdict


def final_mse(run_dir):
    p = os.path.join(run_dir, "out.log")
    if not os.path.exists(p):
        return None
    t = open(p, errors="replace").read()
    if "Eval Epoch: 300" not in t:
        return None
    seg = t.split("Eval Epoch: 300")[-1]
    m = re.search(r"\[rigid\][^\n]*moving mse_fingertips: ([0-9.]+)\s+copy_baseline: ([0-9.]+)", seg)
    if not m:
        return None
    gate = re.findall(r"residual-fusion gate: (-?[0-9.]+)", t)
    return float(m.group(1)), float(m.group(2)), (float(gate[-1]) if gate else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="results_fusion_sweep.json")
    args = ap.parse_args()

    runs = defaultdict(list)
    copy_zero = None
    for d in sorted(os.listdir(args.logs)):
        g = re.match(r"fu_([a-z]+)_([a-z]+)_k(\d+)_s(\d)$", d)
        if not g:
            continue
        r = final_mse(os.path.join(args.logs, d))
        if r is None:
            print("  (incomplete) %s" % d); continue
        runs[(g.group(1), g.group(2))].append(r[0])
        copy_zero = r[1]

    pose = []
    for d in sorted(os.listdir(args.logs)):
        if re.match(r"sf_pose_k8_s\d$", d):
            r = final_mse(os.path.join(args.logs, d))
            if r:
                pose.append(r[0])
    pose_mean = st.mean(pose) if pose else None

    print("\n=== FUSION SWEEP (k=8, scene-disjoint, frozen scene-trained encoder) ===")
    print("%-8s %-9s %-4s %12s %10s" % ("fusion", "arm", "n", "mean MSE", "sd"))
    means = {}
    for key in sorted(runs):
        v = runs[key]
        means[key] = st.mean(v)
        print("%-8s %-9s %-4d %12.6f %10.6f"
              % (key[0], key[1], len(v), st.mean(v), st.stdev(v) if len(v) > 1 else 0.0))
    if pose_mean:
        print("%-8s %-9s %-4d %12.6f" % ("--", "pose-only", len(pose), pose_mean))
    if copy_zero:
        print("%-8s %-9s %-4s %12.6f" % ("--", "copy-zero", "-", copy_zero))

    print("\n=== CONTRASTS (negative = touch helps) ===")
    payload = {"copy_zero_mse": copy_zero, "pose_only_mse": pose_mean, "means": {}, "contrasts": {}}
    for k, v in means.items():
        payload["means"]["%s_%s" % k] = v

    for fusion in sorted({k[0] for k in means}):
        f, s = means.get((fusion, "frz")), means.get((fusion, "frzshuf"))
        if f and s:
            pct = (f - s) / s * 100
            payload["contrasts"]["%s_frz_vs_frzshuf" % fusion] = pct
            print("  %-6s frz vs frzshuf (capacity-matched)   %+7.2f%%" % (fusion, pct))
        if f and pose_mean:
            pct = (f - pose_mean) / pose_mean * 100
            payload["contrasts"]["%s_frz_vs_pose" % fusion] = pct
            print("  %-6s frz vs pose-only                    %+7.2f%%" % (fusion, pct))
        if s and pose_mean:
            pct = (s - pose_mean) / pose_mean * 100
            payload["contrasts"]["%s_frzshuf_vs_pose" % fusion] = pct
            print("  %-6s frzshuf vs pose-only (branch cost)   %+7.2f%%" % (fusion, pct))

    gf, gg = means.get(("film", "frz")), means.get(("gate", "frz"))
    if gf and gg:
        pct = (gf - gg) / gg * 100
        payload["contrasts"]["film_vs_gate_frz"] = pct
        print("\n  film vs gate, real-touch arm          %+7.2f%%" % pct)
        print("  NOTE: not capacity-matched (film 74,430 vs gate 66,045 trainable).")
        print("  The within-fusion frz-vs-frzshuf contrasts above are the controlled ones.")

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
