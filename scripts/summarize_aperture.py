"""Summarize the grip-aperture sweep.

Three numbers per condition, because MSE alone is misleading on this project's
targets: R^2 against predicting no change says how much of the signal is
actually reachable, and AUC on the sign says how often the DIRECTION of the
grip change is right -- which is the part a policy can act on.

The headline comparison is frz vs frzshuf within a fusion (capacity-matched),
and both against pose-only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
from collections import defaultdict

_PAT = re.compile(
    r"target_mode=grip_aperture\s+moving mse: ([0-9.eE+-]+)\s+"
    r"copy_baseline: ([0-9.eE+-]+)\s+R2_vs_zero: ([0-9.eE+-]+)\s+AUC_sign: ([0-9.eE+-]+)"
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
    mse, copy, r2, auc = hits[-1]
    return dict(mse=float(mse), copy=float(copy), r2=float(r2), auc=float(auc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="results_aperture_sweep.json")
    args = ap.parse_args()

    groups = defaultdict(list)
    copy_zero = None
    for d in sorted(os.listdir(args.logs)):
        m = re.match(r"ap_pose_k(\d+)_s(\d)$", d)
        f = re.match(r"ap_([a-z]+)_([a-z]+)_k(\d+)_s(\d)$", d)
        r = None
        if m:
            r = final_metrics(os.path.join(args.logs, d))
            key = ("--", "pose")
        elif f:
            r = final_metrics(os.path.join(args.logs, d))
            key = (f.group(1), f.group(2))
        else:
            continue
        if r is None:
            print("  (incomplete) %s" % d); continue
        groups[key].append(r)
        copy_zero = r["copy"]

    print("\n=== GRIP-APERTURE SWEEP (k=8, scene-disjoint, frozen scene encoder) ===")
    print("Target: change in mean fingertip-to-wrist distance. Rotation invariant.")
    print("\n%-8s %-9s %-3s %12s %10s %10s" % ("fusion", "arm", "n", "MSE", "R2_vs_0", "AUC_sign"))
    means = {}
    for key in sorted(groups):
        v = groups[key]
        means[key] = {k: st.mean([x[k] for x in v]) for k in ("mse", "r2", "auc")}
        sd_auc = st.stdev([x["auc"] for x in v]) if len(v) > 1 else 0.0
        print("%-8s %-9s %-3d %12.8f %10.4f %10.4f (sd %.4f)"
              % (key[0], key[1], len(v), means[key]["mse"], means[key]["r2"],
                 means[key]["auc"], sd_auc))
    if copy_zero:
        print("%-8s %-9s %-3s %12.8f %10.4f" % ("--", "copy-zero", "-", copy_zero, 0.0))

    pose = means.get(("--", "pose"))
    print("\n=== CONTRASTS ===")
    payload = {"copy_zero_mse": copy_zero,
               "means": {"%s_%s" % k: v for k, v in means.items()},
               "contrasts": {}}
    for fusion in sorted({k[0] for k in means if k[0] != "--"}):
        f, s = means.get((fusion, "frz")), means.get((fusion, "frzshuf"))
        if f and s:
            d_mse = (f["mse"] - s["mse"]) / s["mse"] * 100
            payload["contrasts"]["%s_frz_vs_frzshuf_mse_pct" % fusion] = d_mse
            payload["contrasts"]["%s_frz_vs_frzshuf_auc_delta" % fusion] = f["auc"] - s["auc"]
            print("  %-5s frz vs frzshuf (capacity-matched): MSE %+6.2f%%   AUC %+.4f"
                  % (fusion, d_mse, f["auc"] - s["auc"]))
        if f and pose:
            d_mse = (f["mse"] - pose["mse"]) / pose["mse"] * 100
            payload["contrasts"]["%s_frz_vs_pose_mse_pct" % fusion] = d_mse
            payload["contrasts"]["%s_frz_vs_pose_auc_delta" % fusion] = f["auc"] - pose["auc"]
            print("  %-5s frz vs pose-only:                  MSE %+6.2f%%   AUC %+.4f"
                  % (fusion, d_mse, f["auc"] - pose["auc"]))

    if pose:
        print("\n  pose-only reaches R2=%.4f, AUC=%.4f on this target." % (pose["r2"], pose["auc"]))
        print("  For scale: the 63-d delta target's best linear model reached R2=0.19,")
        print("  and pose-only there beat copy-zero by only 8-9%.")

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
