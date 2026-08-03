"""Collect the random-encoder forecasting control into one JSON.

The metric is the one HANDOFF 2.19d reports: `moving mse_fingertips` from the
`[rigid]` block of the final eval, i.e. MSE on the corrected target restricted
to samples where the hand is actually moving. Verified against the existing
k=8 run -- logs/rnd_frz_k8 gives 0.000261, which is 2.19d's published figure
for the random frozen encoder.

Lower is better. The two contrasts that matter:

  frz vs frzshuf   capacity-matched: same parameter count, tactile input
                   deranged. Isolates "is the tactile signal being used".
  frz vs pose-only does the tactile branch beat not having one at all.

2.19d's claim is that RANDOM beats PRETRAINED on both. This script exists to
test whether that survives at k=2 and k=4, or whether -- like the probe
version in 2.27 -- it was specific to k=8.
"""

from __future__ import annotations

import argparse
import json
import os
import re

ARMS = ["frz", "frzshuf", "rndfrz", "rndfrzshuf", "pose"]

# The training logs print this metric with 6 decimal places, so each value
# carries +/- 5e-7 of rounding. At k=2 the MSEs are ~4e-5, i.e. TWO significant
# figures, and a 1e-6 gap between two arms is not a finding.
LOG_PRECISION = 5e-7
_RIGID = re.compile(r"\[rigid\].*?moving mse_fingertips:\s*([0-9.eE+-]+)"
                    r"\s+copy_baseline:\s*([0-9.eE+-]+)")


def final_metrics(run):
    """Last [rigid] moving-fingertip MSE and its copy baseline, or None."""
    path = os.path.join("logs", run, "out.log")
    if not os.path.exists(path):
        return None
    mse = base = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _RIGID.search(line)
            if m:
                mse, base = float(m.group(1)), float(m.group(2))
    if mse is None:
        return None
    return {"run": run, "moving_mse_fingertips": mse, "copy_baseline": base}


def pct(a, b):
    """Percent change of a relative to b; negative means a is better."""
    return None if (a is None or b is None or b == 0) else 100.0 * (a - b) / b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--out", default="results_randenc_horizons.json")
    args = ap.parse_args()

    report = {"metric": "moving mse_fingertips on the rigid (corrected) target",
              "protocol": "scene-disjoint, split_seed 42, 300 epochs, seed 1",
              "horizons": {}}

    for k in args.horizons:
        got = {a: final_metrics("rh%d_%s" % (k, a)) for a in ARMS}
        missing = [a for a, v in got.items() if v is None]
        entry = {"runs": got, "missing": missing}
        if not missing:
            frz = got["frz"]["moving_mse_fingertips"]
            rnd = got["rndfrz"]["moving_mse_fingertips"]
            shuf = got["frzshuf"]["moving_mse_fingertips"]
            rshuf = got["rndfrzshuf"]["moving_mse_fingertips"]
            pose = got["pose"]["moving_mse_fingertips"]
            cap_pre, cap_rnd = pct(frz, shuf), pct(rnd, rshuf)
            pose_pre, pose_rnd = pct(frz, pose), pct(rnd, pose)
            entry["contrasts"] = {
                "pretrained_vs_its_shuffled_pct": cap_pre,
                "random_vs_its_shuffled_pct": cap_rnd,
                "pretrained_vs_pose_pct": pose_pre,
                "random_vs_pose_pct": pose_rnd,
                # 2.22's rule: never read the capacity-matched gap without the
                # branch cost. A shuffled control that is itself worse than
                # pose-only inflates its own arm's capacity-matched gap.
                "pretrained_branch_cost_pct": pct(shuf, pose),
                "random_branch_cost_pct": pct(rshuf, pose),
                "capacity_matched_favours": "pretrained" if cap_pre < cap_rnd else "random",
                "vs_pose_favours": "pretrained" if pose_pre < pose_rnd else "random",
                "contrasts_agree": (cap_pre < cap_rnd) == (pose_pre < pose_rnd),
                # The logs carry 6 decimals. At k=2 that is two significant
                # figures, so a 1e-6 difference on a 4e-5 value is not a
                # finding. Report resolvability instead of asserting a winner.
                "resolvable_at_logged_precision": abs(rnd - frz) > 2 * LOG_PRECISION,
                "random_minus_pretrained_mse": rnd - frz,
            }
        report["horizons"][str(k)] = entry

    print("\n=== RANDOM vs PRETRAINED FROZEN ENCODER, FORECASTING ===")
    print("metric: %s (lower is better)\n" % report["metric"])
    print("%-5s %12s %12s %12s %12s %12s"
          % ("k", "frz", "frzshuf", "rndfrz", "rndfrzshuf", "pose"))
    for k, e in report["horizons"].items():
        if e["missing"]:
            print("%-5s MISSING: %s" % (k, ", ".join(e["missing"])))
            continue
        print("%-5s %12.6f %12.6f %12.6f %12.6f %12.6f"
              % (k, *[e["runs"][a]["moving_mse_fingertips"] for a in ARMS]))

    print("\n=== THE TWO CONTRASTS, WHICH DO NOT AGREE ===")
    print("%-5s %28s %28s %8s"
          % ("k", "capacity-matched (own shuf)", "vs pose-only", "agree?"))
    for k, e in report["horizons"].items():
        if e["missing"]:
            continue
        c = e["contrasts"]
        print("%-5s  pre %+7.2f%% rnd %+7.2f%% -> %-9s  pre %+7.2f%% rnd %+7.2f%% -> %-9s %s"
              % (k, c["pretrained_vs_its_shuffled_pct"], c["random_vs_its_shuffled_pct"],
                 c["capacity_matched_favours"], c["pretrained_vs_pose_pct"],
                 c["random_vs_pose_pct"], c["vs_pose_favours"],
                 "yes" if c["contrasts_agree"] else "NO"))

    print("\n=== WHY: the two shuffled controls are not equivalent ===")
    print("%-5s %26s %26s" % ("k", "pretrained shuf vs pose", "random shuf vs pose"))
    for k, e in report["horizons"].items():
        if e["missing"]:
            continue
        c = e["contrasts"]
        print("%-5s %25.2f%% %25.2f%%"
              % (k, c["pretrained_branch_cost_pct"], c["random_branch_cost_pct"]))
    print("A shuffled control that is itself worse than pose-only inflates its")
    print("own arm's capacity-matched gap (HANDOFF 2.22, the FiLM trap).")

    print("\n=== resolvable at the logged precision (6 dp)? ===")
    for k, e in report["horizons"].items():
        if e["missing"]:
            continue
        c = e["contrasts"]
        print("  k=%-3s %s" % (k, "yes" if c["resolvable_at_logged_precision"]
                               else "NO -- difference is inside rounding"))

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
