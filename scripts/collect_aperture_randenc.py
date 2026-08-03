"""Does the LEARNED tactile representation beat a random one on grip aperture?

Pairs the random-encoder arms (results_aperture_randenc_k*_ss*.json, produced
by scripts/aperture_randenc.sh) against the pretrained arms already committed
in results/, horizon by horizon and partition by partition.

WHY THIS COMPARISON AND NOT THE DELTA-MSE ONE. See the header of
scripts/aperture_randenc.sh: HANDOFF 2.22 argues MSE on the corrected delta is
a low-power instrument, and 2.31 shows the k=8 delta result did not replicate
at k=2. Grip aperture is scored by AUC and R^2 on a scalar with a real sign,
and it is where the tactile effect actually lives.

THE TWO CONTRASTS.

  touch - shuffled   capacity-matched: identical parameter count, tactile
                     input deranged. Asks "is the tactile signal used at all".
                     Computed separately for the pretrained and the random
                     encoder, because each needs its OWN shuffled control --
                     reading one arm's gap against the other's control is the
                     trap HANDOFF 2.22 records for FiLM.

  pretrained - random
                     the question. Positive means the learned representation
                     buys something a fixed random projection does not.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st

HORIZONS = [2, 4, 8, 16]
PARTITIONS = [42, 1, 2, 3]

# Where the PRETRAINED arms live. k=8 is split across two files for historical
# reasons: partition 42 came from the early-stopping study, 1/2/3 from the
# split-seed study.
PRETRAINED = {
    (2, ss): "results_aperture_k2_ss%d.json" % ss for ss in PARTITIONS
}
PRETRAINED.update({(4, ss): "results_aperture_k4_ss%d.json" % ss for ss in PARTITIONS})
PRETRAINED.update({(16, ss): "results_aperture_k16_ss%d.json" % ss for ss in PARTITIONS})
PRETRAINED[(8, 42)] = "results_aperture_earlystop.json"
for ss in (1, 2, 3):
    PRETRAINED[(8, ss)] = "results_aperture_splitseed_ss%d.json" % ss


def load(path):
    for cand in (path, os.path.join("results", path)):
        if os.path.exists(cand):
            with open(cand) as fh:
                return json.load(fh)
    return None


def arm_means(payload, arm):
    if not payload:
        return None
    m = payload.get("means", {}).get(arm)
    return m if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_aperture_randenc_summary.json")
    args = ap.parse_args()

    rows, report = [], {"cells": {}, "by_horizon": {}}
    for k in HORIZONS:
        for ss in PARTITIONS:
            pre = load(PRETRAINED[(k, ss)])
            rnd = load("results_aperture_randenc_k%d_ss%d.json" % (k, ss))
            cell = {
                "pretrained_frz": arm_means(pre, "frz"),
                "pretrained_frzshuf": arm_means(pre, "frzshuf"),
                "random_frz": arm_means(rnd, "rndfrz"),
                "random_frzshuf": arm_means(rnd, "rndfrzshuf"),
            }
            if not all(cell.values()):
                cell["missing"] = [n for n, v in cell.items() if not v]
                report["cells"]["k%d_ss%d" % (k, ss)] = cell
                continue
            for metric in ("test_auc", "test_r2"):
                cell["pre_capacity_%s" % metric] = (
                    cell["pretrained_frz"][metric] - cell["pretrained_frzshuf"][metric])
                cell["rnd_capacity_%s" % metric] = (
                    cell["random_frz"][metric] - cell["random_frzshuf"][metric])
                cell["pre_minus_rnd_%s" % metric] = (
                    cell["pretrained_frz"][metric] - cell["random_frz"][metric])
            report["cells"]["k%d_ss%d" % (k, ss)] = cell
            rows.append((k, ss, cell))

    print("\n=== GRIP APERTURE: LEARNED vs RANDOM TACTILE ENCODER ===")
    print("epoch chosen on val by R2, scored on TEST. AUC never used to select.\n")
    print("%-4s %-5s %10s %10s %14s %14s %14s"
          % ("k", "part", "pre AUC", "rnd AUC", "pre cap-match", "rnd cap-match", "pre - rnd"))
    for k, ss, c in rows:
        print("%-4d %-5d %10.4f %10.4f %+14.4f %+14.4f %+14.4f"
              % (k, ss, c["pretrained_frz"]["test_auc"], c["random_frz"]["test_auc"],
                 c["pre_capacity_test_auc"], c["rnd_capacity_test_auc"],
                 c["pre_minus_rnd_test_auc"]))

    print("\n=== MEAN OVER PARTITIONS, PER HORIZON ===")
    print("%-4s %16s %16s %20s %10s"
          % ("k", "pre cap-match", "rnd cap-match", "pretrained - random", "positive"))
    for k in HORIZONS:
        cells = [c for kk, _, c in rows if kk == k]
        if not cells:
            continue
        pre = [c["pre_capacity_test_auc"] for c in cells]
        rnd = [c["rnd_capacity_test_auc"] for c in cells]
        diff = [c["pre_minus_rnd_test_auc"] for c in cells]
        report["by_horizon"][str(k)] = {
            "n_partitions": len(cells),
            "pre_capacity_auc": st.mean(pre),
            "rnd_capacity_auc": st.mean(rnd),
            "pre_minus_rnd_auc": st.mean(diff),
            "pre_minus_rnd_range": [min(diff), max(diff)],
            "n_positive": sum(d > 0 for d in diff),
        }
        print("%-4d %+16.4f %+16.4f %+20.4f %6d/%d"
              % (k, st.mean(pre), st.mean(rnd), st.mean(diff),
                 sum(d > 0 for d in diff), len(diff)))

    print("\nReading: 'pretrained - random' > 0 means the LEARNED representation")
    print("buys something a fixed random projection does not. If that shrinks")
    print("with horizon, it matches HANDOFF 2.27 (probe) and 2.31 (delta MSE),")
    print("and contribution 2 is horizon-specific rather than general.")

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
