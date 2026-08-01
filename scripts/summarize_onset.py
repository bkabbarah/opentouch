"""Summarize the motion-onset sweep.

AUC over the currently-still subset, with the base rate reported so the number
can be read against what a constant predictor achieves (0.5, regardless of
imbalance). The headline is frz vs frzshuf -- same architecture, same
trainable parameter count, only the tactile pairing scrambled -- and both
against pose-only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
from collections import defaultdict

_PAT = re.compile(
    r"target_mode=motion_onset\s+still n=(\d+)\s+base_rate=([0-9.]+)\s+AUC=([0-9.]+)"
)


def final_metrics(run_dir):
    p = os.path.join(run_dir, "out.log")
    if not os.path.exists(p):
        return None
    t = open(p, errors="replace").read()
    if "Eval Epoch: 300" not in t:
        return None
    hits = _PAT.findall(t)
    if not hits:
        return None
    n, base, auc = hits[-1]
    return dict(n_still=int(n), base_rate=float(base), auc=float(auc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="results_onset_sweep.json")
    args = ap.parse_args()

    groups = defaultdict(list)
    for d in sorted(os.listdir(args.logs)):
        m = re.match(r"on_([a-z]+)_k(\d+)_s(\d)$", d)
        if not m:
            continue
        r = final_metrics(os.path.join(args.logs, d))
        if r is None:
            print("  (incomplete) %s" % d); continue
        groups[m.group(1)].append(r)

    if not groups:
        print("No completed on_* runs found."); return

    print("\n=== MOTION-ONSET SWEEP (k=8, scene-disjoint, frozen scene encoder) ===")
    print("Task: hand is currently STILL -- will it move over the next k frames?")
    print("Chance is 0.5 regardless of the base rate.\n")
    print("%-10s %-3s %10s %10s %10s" % ("arm", "n", "AUC", "sd", "base rate"))

    means = {}
    n_still = base_rate = None
    for arm in sorted(groups):
        v = groups[arm]
        aucs = [x["auc"] for x in v]
        means[arm] = st.mean(aucs)
        n_still = v[0]["n_still"]
        base_rate = v[0]["base_rate"]
        print("%-10s %-3d %10.4f %10.4f %10.4f"
              % (arm, len(v), st.mean(aucs),
                 st.stdev(aucs) if len(aucs) > 1 else 0.0, base_rate))

    print("\n  still subset: n=%s, base rate %.4f" % (n_still, base_rate))

    print("\n=== CONTRASTS (positive = touch helps) ===")
    payload = {"n_still": n_still, "base_rate": base_rate, "auc_means": means, "contrasts": {}}
    f, s, p = means.get("frz"), means.get("frzshuf"), means.get("pose")
    if f and s:
        payload["contrasts"]["frz_minus_frzshuf"] = f - s
        print("  frz - frzshuf (capacity-matched)  %+.4f AUC" % (f - s))
    if f and p:
        payload["contrasts"]["frz_minus_pose"] = f - p
        print("  frz - pose-only                   %+.4f AUC" % (f - p))
    if s and p:
        payload["contrasts"]["frzshuf_minus_pose"] = s - p
        print("  frzshuf - pose-only (branch cost) %+.4f AUC" % (s - p))

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
