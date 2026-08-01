"""Pick each aperture run's stopping epoch on VAL, then score it on TEST.

Why the two splits. The touch arm peaks around epoch 4 and decays; pose-only
climbs for far longer. Any single shared epoch mis-states one of them. But
choosing each arm's best VALIDATION epoch and then reporting VALIDATION
metrics is selection on the eval set, which inflates the result exactly the
way quoting the highest-scoring seed does.

Selecting on val and reporting on test separates the two roles. The test split
has been used by almost nothing in this project, so the number it produces is
genuinely out of sample.

The selection criterion is R^2 on val, applied identically to every arm. AUC is
reported but never used to select, so the two numbers stay independent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
from collections import defaultdict

_VAL = re.compile(
    r"Eval Epoch: (\d+)\s+target_mode=grip_aperture[^\n]*?"
    r"R2_vs_zero: ([-0-9.]+)\s+AUC_sign: ([0-9.]+)"
)


def val_trajectory(run_dir):
    p = os.path.join(run_dir, "out.log")
    if not os.path.exists(p):
        return {}
    return {int(e): (float(r2), float(auc))
            for e, r2, auc in _VAL.findall(open(p, errors="replace").read())}


def eval_on_test(checkpoint, data):
    """Run the standalone evaluator on the test split and pull the metrics."""
    cmd = [sys.executable, "-m", "opentouch_train.regression_eval",
           "--checkpoint", checkpoint, "--data", data, "--split", "test",
           "--batch-size", "256", "--workers", "4"]
    out = subprocess.run(cmd, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": "src"})
    text = out.stdout + out.stderr
    m = re.search(r"R2_vs_zero: ([-0-9.]+)\s+AUC_sign: ([0-9.]+)", text)
    if not m:
        return None, text[-1500:]
    return (float(m.group(1)), float(m.group(2))), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--data", required=True)
    ap.add_argument("--prefix", default="aes_")
    ap.add_argument("--out", default="results_aperture_earlystop.json")
    args = ap.parse_args()

    per_arm = defaultdict(list)
    for d in sorted(os.listdir(args.logs)):
        m = re.match(re.escape(args.prefix) + r"([a-z]+)_k\d+_s(\d)$", d)
        if not m:
            continue
        traj = val_trajectory(os.path.join(args.logs, d))
        if not traj:
            print("  (no val trajectory) %s" % d); continue
        best_epoch = max(traj, key=lambda e: traj[e][0])
        ck = os.path.join(args.logs, d, "checkpoints", "epoch_%d.pt" % best_epoch)
        if not os.path.exists(ck):
            print("  (missing checkpoint %s) %s" % (best_epoch, d)); continue
        test, err = eval_on_test(ck, args.data)
        if test is None:
            print("  (test eval failed) %s\n%s" % (d, err)); continue
        per_arm[m.group(1)].append({
            "run": d, "best_val_epoch": best_epoch,
            "val_r2": traj[best_epoch][0], "val_auc": traj[best_epoch][1],
            "test_r2": test[0], "test_auc": test[1],
        })
        print("  %-22s best-val epoch %2d  val R2=%.4f  ->  TEST R2=%.4f AUC=%.4f"
              % (d, best_epoch, traj[best_epoch][0], test[0], test[1]))

    if not per_arm:
        print("Nothing to summarise."); return

    print("\n=== GRIP APERTURE: epoch selected on VAL, scored on TEST ===")
    print("%-10s %-3s %14s %10s %10s" % ("arm", "n", "best epoch", "TEST R2", "TEST AUC"))
    means = {}
    for arm in sorted(per_arm):
        v = per_arm[arm]
        means[arm] = {
            "test_r2": st.mean(x["test_r2"] for x in v),
            "test_auc": st.mean(x["test_auc"] for x in v),
            "epochs": [x["best_val_epoch"] for x in v],
        }
        print("%-10s %-3d %14s %10.4f %10.4f"
              % (arm, len(v), str(means[arm]["epochs"]),
                 means[arm]["test_r2"], means[arm]["test_auc"]))

    print("\n=== CONTRASTS ON TEST (out of sample) ===")
    payload = {"per_run": {a: v for a, v in per_arm.items()}, "means": means, "contrasts": {}}
    f, s, p = means.get("frz"), means.get("frzshuf"), means.get("pose")
    if f and p:
        payload["contrasts"]["frz_minus_pose_r2"] = f["test_r2"] - p["test_r2"]
        payload["contrasts"]["frz_minus_pose_auc"] = f["test_auc"] - p["test_auc"]
        print("  frz - pose-only                   R2 %+.4f   AUC %+.4f"
              % (f["test_r2"] - p["test_r2"], f["test_auc"] - p["test_auc"]))
    if f and s:
        payload["contrasts"]["frz_minus_frzshuf_r2"] = f["test_r2"] - s["test_r2"]
        payload["contrasts"]["frz_minus_frzshuf_auc"] = f["test_auc"] - s["test_auc"]
        print("  frz - frzshuf (capacity-matched)  R2 %+.4f   AUC %+.4f"
              % (f["test_r2"] - s["test_r2"], f["test_auc"] - s["test_auc"]))
    if s and p:
        print("  frzshuf - pose-only (branch cost)  R2 %+.4f   AUC %+.4f"
              % (s["test_r2"] - p["test_r2"], s["test_auc"] - p["test_auc"]))

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nWrote %s" % args.out)
    print("\nNOTE: epochs were chosen on val by R2 and never on AUC, so the AUC")
    print("figures above are independent of the selection.")


if __name__ == "__main__":
    main()
