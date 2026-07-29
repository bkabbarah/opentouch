"""Summarize the encoder-quality ladder: does forecasting gain track tactile
representation quality, with the participant partition held fixed?

Pairs each rung's frz/frzshuf runs, reads the encoder's retrieval mAP out of
the retrieval log at that epoch, and reports gain against mAP. Also checks the
prediction the ladder exists to test: the three redrawn partitions' encoders
(mAP 18.74 / 20.78 / 24.29) fall inside the ladder's range, so if quality is
what drives the gain, the ladder should reproduce their forecasting values.
"""

from __future__ import annotations

import argparse
import json
import os
import re

# Observed in the split-seed study: (encoder val t2p mAP, frz-vs-frzshuf % at k=8)
SPLIT_SEED_OBSERVED = [
    ("seed 1", 24.29, -5.77),
    ("seed 2", 20.78, -10.03),
    ("seed 3", 18.74, +5.02),
    ("seed 42", 28.24, -14.96),
]


def final_mse(run_dir):
    p = os.path.join(run_dir, "out.log")
    if not os.path.exists(p):
        return None
    t = open(p, errors="replace").read()
    if "Eval Epoch: 300" not in t:
        return None
    seg = t.split("Eval Epoch: 300")[-1]
    m = re.search(r"\[rigid\][^\n]*moving mse_fingertips: ([0-9.]+)\s+copy_baseline: ([0-9.]+)", seg)
    return (float(m.group(1)), float(m.group(2))) if m else None


def encoder_map(logs, epoch):
    """t2p mAP recorded for p2t_scene_gru at this epoch."""
    p = os.path.join(logs, "p2t_scene_gru", "out.log")
    t = open(p, errors="replace").read()
    eps = re.findall(r"Eval Epoch: (\d+)", t)
    maps = re.findall(r"tactile->pose[^\n]*mAP: ([0-9.]+)", t)
    for e, m in zip(eps, maps):
        if int(e) == int(epoch):
            return float(m) * 100
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="results_encoder_ladder.json")
    args = ap.parse_args()

    rungs = {}
    for d in sorted(os.listdir(args.logs)):
        g = re.match(r"el(\d+)_([a-z]+)_k(\d+)$", d)
        if not g:
            continue
        r = final_mse(os.path.join(args.logs, d))
        if r is None:
            print("  (incomplete) %s" % d); continue
        rungs.setdefault(int(g.group(1)), {})[g.group(2)] = r[0]
        rungs[int(g.group(1))]["copy"] = r[1]

    rows = []
    for ep in sorted(rungs):
        r = rungs[ep]
        if "frz" not in r or "frzshuf" not in r:
            print("  (unpaired rung, skipped) epoch %s" % ep); continue
        gain = (r["frz"] - r["frzshuf"]) / r["frzshuf"] * 100
        rows.append({
            "encoder_epoch": ep,
            "encoder_map": encoder_map(args.logs, ep),
            "frz_mse": r["frz"], "frzshuf_mse": r["frzshuf"],
            "copy_zero_mse": r["copy"],
            "frz_vs_frzshuf_pct": gain,
        })

    POSE_ONLY = 0.000290  # sf_pose_k8_s{1,2,3}, same partition and horizon
    for r in rows:
        r["frz_vs_pose_pct"] = (r["frz_mse"] - POSE_ONLY) / POSE_ONLY * 100

    print("\n=== ENCODER-QUALITY LADDER (partition fixed, split_seed 42, k=8) ===")
    print("%-8s %-10s %11s %11s %14s %13s" %
          ("enc_ep", "enc_mAP", "frz_mse", "frzshuf_mse", "frz-v-frzshuf", "frz-v-pose"))
    for r in rows:
        print("%-8d %-10s %11.6f %11.6f %+13.2f%% %+12.2f%%" %
              (r["encoder_epoch"],
               "%.2f" % r["encoder_map"] if r["encoder_map"] else "?",
               r["frz_mse"], r["frzshuf_mse"],
               r["frz_vs_frzshuf_pct"], r["frz_vs_pose_pct"]))

    corr = None
    if len(rows) >= 3:
        xs = [r["encoder_map"] for r in rows if r["encoder_map"]]
        ys = [r["frz_vs_frzshuf_pct"] for r in rows if r["encoder_map"]]
        if len(xs) >= 3:
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
            corr = num / den if den else None
            print("\nPearson r(encoder mAP, frz-vs-frzshuf gain) = %.3f  (n=%d)" % (corr, n))
            print("Negative r means: better encoder -> more negative gain -> touch helps MORE.")

    print("\n=== THE PREDICTION THIS LADDER TESTS ===")
    print("Split-seed study observed, at encoder qualities inside the ladder's range:")
    for name, mp, obs in sorted(SPLIT_SEED_OBSERVED, key=lambda x: x[1]):
        near = min(rows, key=lambda r: abs((r["encoder_map"] or 0) - mp)) if rows else None
        pred = near["frz_vs_frzshuf_pct"] if near else float("nan")
        print("  %-8s encoder mAP %5.2f  observed %+7.2f%%   ladder at mAP %5.2f predicts %+7.2f%%"
              % (name, mp, obs, near["encoder_map"] if near else float("nan"), pred))
    print("\nIf observed and predicted broadly agree, representation quality explains the")
    print("split-seed instability. If they do not, the participant draw does.")

    with open(args.out, "w") as fh:
        json.dump({"partition": "split_seed 42, scene-disjoint", "horizon_k": 8,
                   "pose_only_mse": POSE_ONLY, "rungs": rows,
                   "pearson_r_map_vs_gain": corr,
                   "split_seed_observed": SPLIT_SEED_OBSERVED}, fh, indent=2)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
