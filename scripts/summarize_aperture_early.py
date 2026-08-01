"""Locate the grip-aperture peak, and report the arm contrast AT that peak.

The 300-epoch sweep may be reporting a number well past the model's best. This
reads the dense early-validation runs, averages over seeds at each epoch, and
prints the trajectory plus the best epoch per arm.

The contrast is reported at the epoch that is best for POSE-ONLY, not for the
touch arm. Picking each arm's own best epoch would let the touch arm choose a
more favourable stopping point than its baseline -- a subtle way to manufacture
a margin. Using a single shared epoch, selected on the arm that has no tactile
input, cannot flatter touch.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics as st
from collections import defaultdict

_PAT = re.compile(
    r"Eval Epoch: (\d+)\s+target_mode=grip_aperture[^\n]*?"
    r"R2_vs_zero: ([-0-9.]+)\s+AUC_sign: ([0-9.]+)"
)


def trajectory(run_dir):
    p = os.path.join(run_dir, "out.log")
    if not os.path.exists(p):
        return {}
    text = open(p, errors="replace").read()
    return {int(e): (float(r2), float(auc)) for e, r2, auc in _PAT.findall(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    args = ap.parse_args()

    per_arm = defaultdict(lambda: defaultdict(list))
    for d in sorted(os.listdir(args.logs)):
        m = re.match(r"ape_([a-z]+)_k\d+_s(\d)$", d)
        if not m:
            continue
        for epoch, (r2, auc) in trajectory(os.path.join(args.logs, d)).items():
            per_arm[m.group(1)][epoch].append((r2, auc))

    if not per_arm:
        print("No ape_* runs found."); return

    arms = sorted(per_arm)
    epochs = sorted(set().union(*[set(per_arm[a]) for a in arms]))

    print("\n=== GRIP-APERTURE, DENSE EARLY VALIDATION (k=8, scene-disjoint) ===")
    print("R2 vs predicting no change / AUC on the sign, averaged over seeds.\n")
    header = "epoch " + "".join("%22s" % a for a in arms)
    print(header)
    for e in epochs:
        row = "%5d " % e
        for a in arms:
            vals = per_arm[a].get(e)
            row += "%22s" % ("%.4f / %.4f" % (st.mean(v[0] for v in vals),
                                              st.mean(v[1] for v in vals)) if vals else "--")
        print(row)

    print("\n=== BEST EPOCH PER ARM (by R2) ===")
    best = {}
    for a in arms:
        cand = [(st.mean(v[0] for v in per_arm[a][e]), e) for e in per_arm[a]]
        r2, e = max(cand)
        auc = st.mean(v[1] for v in per_arm[a][e])
        best[a] = (e, r2, auc)
        print("  %-9s epoch %3d   R2=%.4f   AUC=%.4f" % (a, e, r2, auc))

    if "pose" in best:
        sel = best["pose"][0]
        print("\n=== CONTRAST AT EPOCH %d (pose-only's best, chosen to not flatter touch) ===" % sel)
        ref = per_arm["pose"].get(sel)
        for a in arms:
            vals = per_arm[a].get(sel)
            if not vals or not ref:
                continue
            r2 = st.mean(v[0] for v in vals)
            auc = st.mean(v[1] for v in vals)
            line = "  %-9s R2=%+.4f  AUC=%.4f" % (a, r2, auc)
            if a != "pose":
                line += "   (vs pose-only: R2 %+.4f, AUC %+.4f)" % (
                    r2 - st.mean(v[0] for v in ref), auc - st.mean(v[1] for v in ref))
            print(line)

    print("\nFor reference, the 300-epoch sweep ended at:")
    print("  frz R2=0.155 AUC=0.645   pose R2=0.116 AUC=0.589   frzshuf R2=-0.012 AUC=0.545")


if __name__ == "__main__":
    main()
