"""Clip-clustered CIs for the forecasting arms, and the gate-zero diagnostic.

Two questions the sweep could not answer, both needing per-sample errors rather
than the aggregate MSE the training loop prints.

(1) CONFIDENCE INTERVALS. Every forecasting number in HANDOFF rests on a
    3-seed standard deviation. The direction probes get clip-clustered
    bootstrap CIs; the forecasting headline gets nothing. Same reasoning as
    HANDOFF 2.3 applies here -- eval windows come from a few hundred clips and
    neighbouring windows share 19 of 20 frames, so the cluster is the clip.

    Because all arms are evaluated on the SAME windows, the bootstrap is
    PAIRED: resample clips once per draw and recompute every arm on that same
    resample, so the difference between arms is not swamped by which clips got
    drawn. Seeds within an arm are averaged per sample first, so the interval
    describes the condition, not one training run.

(2) THE GATE-ZERO DIAGNOSTIC. The model computes
        output = pose_head(pose) + gate * tactile_head(tactile)
    with `gate` a learnable scalar initialised to zero, so gate=0 reproduces
    the pose-only head exactly (tests/test_pose_regression.py::
    test_residual_gate_zero_matches_pose_only_head).

    The shuffled arm should therefore be able to learn gate~0 and match
    pose-only. It does not -- it lands 9-14% WORSE, with gates around -0.03.
    That matters for interpretation: if the shuffled arm's deficit is the
    optimiser failing to switch off a useless branch, then part of the
    reported "touch content" is really "real touch gives the gate something
    useful to do". Forcing gate=0 at inference separates the two:

      frzshuf @ gate=0 ~= pose-only   -> the deficit is gate misuse, and the
                                        frz-vs-frzshuf gap is partly that
      frzshuf @ gate=0 still worse    -> the pose head itself was damaged by
                                        co-training, a real capacity cost

Usage:
    python scripts/forecast_ci.py --logs logs --data <dataset> \
        --runs "sf_{arm}_k{k}_s{seed}" --horizons 8 16 --seeds 1 2 3 \
        --arms frz frzshuf pose --n-boot 1000 --out results_forecast_ci.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from opentouch.pose_regression import FINGERTIP_COLUMNS, PoseTransitionRegressor
from opentouch.regression_metrics import fingertip_displacement
from opentouch_train.data import VideoTactilePoseDataset
from opentouch_train.regression_data import PoseTransitionDataset, regression_collate_fn
from opentouch_train.regression_eval import _read_checkpoint_meta

# Imported, not reimplemented. _extract_batch is what the training loop itself
# uses to build the rigid target (Kabsch rotation removal in float32, then the
# palm-frame rotation), and fingertip_displacement is what defines the moving
# subset. Recomputing either by hand here would risk a CI that describes a
# slightly different quantity than the MSEs it is meant to bracket.
from opentouch_train.regression_train import _extract_batch

log = logging.getLogger("forecast_ci")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def per_sample_errors(ckpt_path, data, split, device, batch_size, workers,
                      force_gate_zero=False):
    """Per-sample fingertip squared error on the rigid target, plus the clip
    index of each sample and its copy-zero baseline error.

    Returns (err, clip_ids, copy_err, moving_mask, gate_value).
    """
    meta = _read_checkpoint_meta(ckpt_path)
    pose_only = meta["pose_only"]

    model = PoseTransitionRegressor(
        use_tactile=not pose_only,
        tactile_emb_dim=meta["tactile_emb_dim"],
        hidden_dim=meta["hidden_dim"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    if next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)

    gate_value = None
    if model.gate is not None:
        gate_value = float(model.gate.detach().cpu().item())
        if force_gate_zero:
            model.gate.data.zero_()
    model.eval()

    base = VideoTactilePoseDataset(
        hf_dataset_path=data,
        split=split,
        sequence_length=meta["sequence_length"],
        val_ratio=0.1, test_ratio=0.1,
        random_seed=meta["split_seed"],
        split_group_by=meta["split_group_by"],
        include_tactile=not pose_only, include_visual=False, include_pose=True,
    )
    ds = PoseTransitionDataset(
        base, meta["horizon_k"],
        shuffle_tactile=meta["shuffle_tactile"], shuffle_seed=meta["split_seed"],
        causal=meta["causal"], causal_window=meta["causal_window"],
        min_history=meta["min_history"],
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    pin_memory=True, drop_last=False, collate_fn=regression_collate_fn)

    tip = list(FINGERTIP_COLUMNS)
    errs, copies, disps, clips = [], [], [], []
    with torch.inference_mode():
        for batch in dl:
            pose_t, _, articulation_delta, rigid_delta, tac = _extract_batch(
                batch, not pose_only, device, None)
            pred = model(pose_t, tac).float()
            tgt = rigid_delta.float()
            # Matches regression_metrics.per_joint_squared_error: squared L2
            # per joint (summed over coords), then averaged over the five
            # fingertips. Averaging over coords instead would rescale by 3 --
            # harmless for ratios, wrong for anything absolute.
            errs.append((pred[:, tip] - tgt[:, tip]).pow(2).sum(-1).mean(-1).cpu())
            copies.append(tgt[:, tip].pow(2).sum(-1).mean(-1).cpu())
            # The moving mask is defined on ARTICULATION displacement, not the
            # rigid target (regression_train.py:304) -- median L2 over tips.
            disps.append(fingertip_displacement(articulation_delta.float()).cpu())
            clips.extend(f"{s}||{c}" for s, c in zip(batch["scene"], batch["clip_id"]))

    err = torch.cat(errs).numpy()
    copy_err = torch.cat(copies).numpy()
    disp = torch.cat(disps).numpy()

    # Cluster on (scene, clip_id) straight off the batch, so the ids cannot
    # drift out of step with the errors regardless of how the dataset indexes.
    uniq = {c: i for i, c in enumerate(dict.fromkeys(clips))}
    clip_ids = np.asarray([uniq[c] for c in clips])

    return err, clip_ids, copy_err, disp, gate_value


def clustered_paired_boot(arm_errs, clip_ids, n_boot, rng):
    """Paired clip-clustered bootstrap. arm_errs: {name: per-sample error}.
    Returns {name: mean} and the draw matrix for pairwise comparisons."""
    by_clip = {}
    for pos, c in enumerate(clip_ids):
        by_clip.setdefault(int(c), []).append(pos)
    clips = [np.asarray(v) for v in by_clip.values()]
    n_clips = len(clips)

    names = list(arm_errs)
    draws = {n: np.zeros(n_boot) for n in names}
    for b in range(n_boot):
        chosen = rng.integers(0, n_clips, size=n_clips)
        idx = np.concatenate([clips[c] for c in chosen])
        for n in names:
            draws[n][b] = arm_errs[n][idx].mean()
    return draws, n_clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--prefix", default="sf")
    ap.add_argument("--arms", nargs="+", default=["frz", "frzshuf", "pose"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gate-zero", action="store_true",
                    help="Also evaluate every tactile arm with gate forced to 0.")
    ap.add_argument("--out", default="results_forecast_ci.json")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    device = torch.device(args.device)
    payload = {"split": args.split, "n_boot": args.n_boot, "horizons": {}}

    for k in args.horizons:
        log.info(f"===== horizon k={k}")
        arm_mean_err, gates = {}, {}
        clip_ids_ref = copy_ref = disp_ref = None
        gate0_mean_err = {}

        for arm in args.arms:
            per_seed, per_seed_g0 = [], []
            for s in args.seeds:
                name = f"{args.prefix}_{arm}_k{k}_s{s}"
                ck = os.path.join(args.logs, name, "checkpoints", "epoch_300.pt")
                if not os.path.exists(ck):
                    log.warning(f"  missing {ck}, skipping")
                    continue
                err, cids, copy_err, disp, gate = per_sample_errors(
                    ck, args.data, args.split, device, args.batch_size, args.workers)
                per_seed.append(err)
                gates.setdefault(arm, []).append(gate)
                if clip_ids_ref is None:
                    clip_ids_ref, copy_ref, disp_ref = cids, copy_err, disp
                elif len(cids) != len(clip_ids_ref):
                    raise SystemExit(
                        f"{name}: {len(cids)} samples but reference has "
                        f"{len(clip_ids_ref)} -- arms are not on the same eval set, "
                        "the paired bootstrap would be meaningless")
                log.info(f"  {name}: mean={err.mean():.8f} gate={gate}")

                if args.gate_zero and gate is not None:
                    e0, _, _, _, _ = per_sample_errors(
                        ck, args.data, args.split, device, args.batch_size,
                        args.workers, force_gate_zero=True)
                    per_seed_g0.append(e0)
                    log.info(f"    gate=0 -> mean={e0.mean():.8f}")

            if per_seed:
                arm_mean_err[arm] = np.mean(per_seed, axis=0)
            if per_seed_g0:
                gate0_mean_err[arm + "@gate0"] = np.mean(per_seed_g0, axis=0)

        if not arm_mean_err:
            continue

        # Restrict to the moving subset the headline numbers use: the training
        # loop's threshold is on median fingertip displacement, which for the
        # rigid target is sqrt of the copy-zero error.
        meta_any = _read_checkpoint_meta(os.path.join(
            args.logs, f"{args.prefix}_{args.arms[0]}_k{k}_s{args.seeds[0]}",
            "checkpoints", "epoch_300.pt"))
        thr = float(meta_any["motion_threshold"])
        moving = disp_ref >= thr
        log.info(f"  moving subset: {moving.sum()}/{len(moving)} (threshold={thr})")

        allarms = {**arm_mean_err, **gate0_mean_err}
        sub = {n: e[moving] for n, e in allarms.items()}
        draws, n_clips = clustered_paired_boot(sub, clip_ids_ref[moving], args.n_boot, rng)

        entry = {
            "n_samples_moving": int(moving.sum()),
            "n_clusters": int(n_clips),
            "motion_threshold": thr,
            "copy_zero_mse": float(copy_ref[moving].mean()),
            "gates": gates,
            "arms": {n: {"mse": float(sub[n].mean()),
                         "boot_mean": float(draws[n].mean()),
                         "ci_low": float(np.percentile(draws[n], 2.5)),
                         "ci_high": float(np.percentile(draws[n], 97.5))}
                     for n in sub},
            "comparisons": {},
        }

        def compare(a, b):
            if a not in draws or b not in draws:
                return
            rel = (draws[a] - draws[b]) / draws[b] * 100.0
            entry["comparisons"][f"{a}_vs_{b}"] = {
                "relative_pct": float((sub[a].mean() - sub[b].mean()) / sub[b].mean() * 100),
                "ci_low_pct": float(np.percentile(rel, 2.5)),
                "ci_high_pct": float(np.percentile(rel, 97.5)),
                "excludes_zero": bool(np.percentile(rel, 2.5) * np.percentile(rel, 97.5) > 0),
            }

        compare("frz", "frzshuf")
        compare("frz", "pose")
        compare("frzshuf", "pose")
        if args.gate_zero:
            compare("frzshuf@gate0", "pose")
            compare("frz@gate0", "pose")
            compare("frz", "frz@gate0")
        payload["horizons"][str(k)] = entry

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 78)
    for k, e in payload["horizons"].items():
        print(f"\nk={k}  n={e['n_samples_moving']} windows from {e['n_clusters']} clips")
        for n, v in e["arms"].items():
            print(f"  {n:16s} mse={v['mse']:.6f}  95% CI [{v['ci_low']:.6f}, {v['ci_high']:.6f}]")
        print("  --- paired comparisons (negative = first is better) ---")
        for n, v in e["comparisons"].items():
            flag = "EXCLUDES 0" if v["excludes_zero"] else "includes 0"
            print(f"  {n:26s} {v['relative_pct']:+7.2f}%  "
                  f"[{v['ci_low_pct']:+7.2f}, {v['ci_high_pct']:+7.2f}]  {flag}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
