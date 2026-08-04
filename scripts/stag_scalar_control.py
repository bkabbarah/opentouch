"""POSITIVE CONTROL for the scalar tactile ablation, on STAG.

WHY THIS EXISTS. §2.34 found that destroying all spatial structure in the
tactile map costs only ~12% of the benefit on grip aperture. The obvious
objection is that the ablation is weak -- that broadcasting the spatial mean
does not actually remove usable information, so the null result says nothing.

STAG (Sundaram et al., Nature 2019, "Learning the signatures of the human
grasp using a scalable tactile glove") is the right control. Its headline
claim is that the SPATIAL pattern of a tactile glove identifies which object
is being held. Same 16x16 grid, same reduction, a task where spatial structure
is known to be essential.

If scalar-only collapses toward chance here while preserving ~88% on aperture,
the ablation demonstrably destroys real tactile information and the aperture
null is a fact about the task, not about the ablation.

    python scripts/stag_scalar_control.py --stag-dir /scratch/bashar/stag \
        --out results_stag_scalar_control.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def reduce_scalar(x):
    """Identical reduction to PoseTransitionRegressor._reduce_tactile:
    spatial mean broadcast back. Per-frame total preserved exactly."""
    return x.mean(dim=(-2, -1), keepdim=True).expand_as(x).contiguous()


class SmallCNN(nn.Module):
    """Deliberately the same shape of encoder as the tactile path elsewhere:
    three conv blocks with pooling, then a linear classifier."""

    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 2 * 2, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_stag(stag_dir):
    import scipy.io as sio
    press = np.load("%s/pressure_16x16.npy" % stag_dir)
    meta = sio.loadmat("%s/metadata.mat" % stag_dir)
    obj = meta["objectId"].reshape(-1).astype(np.int64)
    split = meta["splitId"].reshape(-1).astype(np.int64)
    valid = meta["hasValidLabel"].reshape(-1).astype(bool)
    balanced = meta["isBalanced"].reshape(-1).astype(bool)
    if len(press) != len(obj):
        raise SystemExit("pressure (%d) and metadata (%d) are not aligned"
                         % (len(press), len(obj)))
    keep = valid & balanced
    return press[keep], obj[keep], split[keep], meta


def run(x_tr, y_tr, x_te, y_te, n_classes, reduce_mode, epochs, device, seed):
    torch.manual_seed(seed)
    model = SmallCNN(n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(x_tr, y_tr), batch_size=256, shuffle=True, drop_last=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            if reduce_mode == "scalar":
                xb = reduce_scalar(xb)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()

    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for i in range(0, len(x_te), 512):
            xb = x_te[i:i + 512].to(device)
            if reduce_mode == "scalar":
                xb = reduce_scalar(xb)
            pred = model(xb).argmax(dim=-1).cpu()
            correct += int((pred == y_te[i:i + 512]).sum()); total += len(pred)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stag-dir", default="/scratch/bashar/stag")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    press, obj, split, meta = load_stag(args.stag_dir)
    classes = sorted(set(obj.tolist()))
    remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[v] for v in obj], dtype=np.int64)
    n_classes = len(classes)

    vals = sorted(set(split.tolist()))
    if len(vals) < 2:
        raise SystemExit("expected at least two splitId values, got %s" % vals)
    tr_mask, te_mask = split == vals[0], split == vals[-1]

    x = torch.from_numpy(press).float().unsqueeze(1)
    y_t = torch.from_numpy(y)
    x_tr, y_tr = x[tr_mask], y_t[tr_mask]
    x_te, y_te = x[te_mask], y_t[te_mask]

    # STANDARDISE, and this is load-bearing. The raw STAG values sit in
    # [0.115, 0.354] with std 0.008, so an unnormalised CNN sees essentially
    # constant input and never leaves the majority-class solution -- the first
    # version of this control scored 5.9% against a 3.7% baseline with two of
    # three seeds not learning at all, which is a broken control, not a
    # finding. Statistics come from TRAIN only.
    #
    # Applied BEFORE the scalar reduction, and it is an affine per-tensor
    # transform, so it cannot change what the reduction does: the spatial mean
    # of a standardised frame is the standardised spatial mean.
    mu, sd = x_tr.mean(), x_tr.std()
    x_tr = (x_tr - mu) / sd
    x_te = (x_te - mu) / sd
    print("standardised with train mean %.4f std %.5f" % (float(mu), float(sd)))

    chance = float(np.bincount(y[te_mask]).max() / te_mask.sum())
    print("\n=== STAG: POSITIVE CONTROL FOR THE SCALAR ABLATION ===")
    print("frames %d (valid+balanced)   classes %d   train %d  test %d"
          % (len(x), n_classes, len(x_tr), len(x_te)))
    print("majority-class baseline on test: %.1f%%   uniform chance: %.1f%%\n"
          % (100 * chance, 100.0 / n_classes))

    results = {}
    for mode in ("none", "scalar"):
        accs = [run(x_tr, y_tr, x_te, y_te, n_classes, mode, args.epochs,
                    torch.device(args.device), s) for s in args.seeds]
        results[mode] = {"accs": accs, "mean": float(np.mean(accs)),
                         "sd": float(np.std(accs))}
        print("%-10s test accuracy: %.1f%%  (seeds: %s)"
              % (mode, 100 * np.mean(accs), ", ".join("%.1f" % (100 * a) for a in accs)))

    full, scal = results["none"]["mean"], results["scalar"]["mean"]
    retained = (scal - chance) / (full - chance) if full > chance else float("nan")
    print("\nabove-baseline accuracy retained by scalar-only: %.0f%%" % (100 * retained))
    print("\nCompare with HANDOFF 2.34: on grip aperture the same reduction retained")
    print("~88%% of the benefit. If it retains far less here, the ablation genuinely")
    print("destroys spatial tactile information, and the aperture null is a fact")
    print("about the task rather than a weak ablation.")

    payload = {"n_classes": n_classes, "n_train": int(len(x_tr)), "n_test": int(len(x_te)),
               "majority_baseline": chance, "results": results,
               "above_baseline_retained_by_scalar": retained}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
