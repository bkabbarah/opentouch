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
            entry["contrasts"] = {
                "pretrained_vs_its_shuffled_pct": pct(frz, got["frzshuf"]["moving_mse_fingertips"]),
                "random_vs_its_shuffled_pct": pct(rnd, got["rndfrzshuf"]["moving_mse_fingertips"]),
                "pretrained_vs_pose_pct": pct(frz, got["pose"]["moving_mse_fingertips"]),
                "random_vs_pose_pct": pct(rnd, got["pose"]["moving_mse_fingertips"]),
                "random_minus_pretrained_mse": rnd - frz,
                "random_beats_pretrained": rnd < frz,
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

    print("\n%-5s %22s %22s %18s" % ("k", "pretrained vs shuf", "random vs shuf", "random beats pre?"))
    for k, e in report["horizons"].items():
        if e["missing"]:
            continue
        c = e["contrasts"]
        print("%-5s %21.2f%% %21.2f%% %18s"
              % (k, c["pretrained_vs_its_shuffled_pct"], c["random_vs_its_shuffled_pct"],
                 "YES" if c["random_beats_pretrained"] else "no"))

    print("\nCompare against k=8 (HANDOFF 2.19d): random 0.000261 beat "
          "pretrained 0.000270.\nIf 'random beats pre?' is 'no' at k=2 and k=4, "
          "2.19d was horizon-specific\nand contribution 2 must be reworded.")

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
