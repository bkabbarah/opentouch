"""
Bootstrap confidence intervals for retrieval mAP.

Runs the model once to extract embeddings, then resamples the test set
with replacement N times to estimate mean mAP and 95% CI without retraining.

Usage:
    python bootstrap_eval.py \
        --checkpoint <path> \
        --data <path_to_dataset> \
        --n-bootstrap 1000 \
        --output bootstrap_results.json

Focused tasks (Kai's direction -- tactile and pose only):
    python bootstrap_eval.py \
        --checkpoint logs/2026_06_22-21_01_54-.../checkpoints/epoch_latest.pt \
        --data ../opentouch/preprocessed_data/train_dataset \
        --n-bootstrap 1000 \
        --output bootstrap_p2t.json
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


POSE_INVOLVED_DIRECTIONS = {
    "pose_to_tactile", "tactile_to_pose",
    "visual_pose_to_tactile", "tactile_to_visual_pose",
    "visual_tactile_to_pose", "pose_to_visual_tactile",
    "tactile_pose_to_visual", "visual_to_tactile_pose",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--precision", default="amp")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-directions", action="store_true",
                   help="Bootstrap all directions, not just pose-involved ones.")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--split-group-by", type=str, default=None, choices=["clip", "scene"],
                   help="Split geometry. Defaults to whatever the checkpoint recorded, "
                        "so a scene-disjoint checkpoint is bootstrapped over its own "
                        "scene-disjoint gallery. Overriding this evaluates the model "
                        "against a split it was not trained for.")
    p.add_argument("--model", type=str, default=None,
                   help="Override the model config name recorded in the checkpoint. Needed "
                        "for the avg-pool retrieval arm: those checkpoints predate the "
                        "-AvgPool config and record the GRU name, so loading them without "
                        "this fails on a pose.gru/projection-width mismatch (see HANDOFF "
                        "2.15).")
    p.add_argument("--cluster", type=str, default="clip", choices=["clip", "window"],
                   help="Bootstrap resampling unit. 'clip' (default) resamples whole "
                        "clips, which is required because sliding windows within a clip "
                        "overlap by 19 of 20 frames. 'window' reproduces the naive "
                        "per-window bootstrap for comparison only -- it is too tight.")
    return p.parse_args()


def bootstrap_map(query_emb: torch.Tensor, target_emb: torch.Tensor,
                  n: int, rng: np.random.Generator,
                  clip_ids: np.ndarray | None = None) -> np.ndarray:
    """Bootstrap mAP. Returns array of shape (n,), one mAP per draw.

    Two things here are deliberate and both matter for the reported interval.

    RESAMPLING UNIT. With `clip_ids`, whole clips are resampled with
    replacement and every window of a chosen clip comes along. Windows are
    20-frame sliding windows, so neighbours within a clip overlap by 19 frames
    and are nowhere near independent -- the same argument that forced
    clip-clustered CIs on the direction probe (HANDOFF 2.3, where a per-window
    bootstrap came out ~6x too tight). Passing clip_ids=None reproduces the
    per-window bootstrap and is kept only so the two can be compared; it
    understates the interval and should not be quoted.

    FIXED GALLERY. Only the QUERIES are resampled; the gallery stays the full
    split. mAP here is mean(1/rank) against the entire eval split as gallery
    (metrics.py:75), so it is a function of gallery SIZE -- the same weights
    score 14.42 against 1572 candidates and 16.76 against 1399 (HANDOFF 2.1).
    Resampling the gallery too would blend "how much does mAP vary across
    samples of people" with "how much does mAP vary with gallery size", and
    the second is an artifact of the metric, not a property of the model.
    Resampling the gallery with replacement also duplicates entries, and
    duplicates tie with the correct target under `sim >= correct_sims`, which
    inflates ranks. A fixed gallery avoids both.
    """
    query_emb = F.normalize(query_emb, dim=1)
    target_emb = F.normalize(target_emb, dim=1)
    num_samples = len(query_emb)
    maps = np.zeros(n)

    clip_windows: list[np.ndarray] = []
    if clip_ids is not None:
        by_clip: dict[int, list[int]] = {}
        for pos, cid in enumerate(clip_ids):
            by_clip.setdefault(int(cid), []).append(pos)
        clip_windows = [np.asarray(v) for v in by_clip.values()]
    n_clips = len(clip_windows)

    sim_full = query_emb @ target_emb.t()          # (N_query, N_gallery), gallery fixed
    diag = sim_full.diag()

    for i in range(n):
        if clip_ids is not None:
            chosen = rng.integers(0, n_clips, size=n_clips)
            idx = np.concatenate([clip_windows[c] for c in chosen])
        else:
            idx = rng.integers(0, num_samples, size=num_samples)
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        sim = sim_full[idx_t]                      # queries resampled, gallery intact
        correct_sims = diag[idx_t].unsqueeze(1)
        ranks = (sim >= correct_sims).sum(dim=1).float()
        maps[i] = (1.0 / ranks).mean().item()
    return maps


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    # import here so pip install -e . errors surface clearly
    from opentouch import create_model, get_input_dtype
    from opentouch_train.data import (
        VideoTactilePoseDataset, collate_fn, parse_task,
        _determine_modality_flags, MODALITY_TO_FEATURE_KEY,
    )
    from opentouch_train.precision import get_autocast
    from opentouch_train.eval import _read_checkpoint_meta

    meta = _read_checkpoint_meta(args.checkpoint)
    task_type = meta.get("task_type")
    model_name = args.model or meta.get("model")
    if args.model and meta.get("model") and args.model != meta["model"]:
        log.warning("--model=%s OVERRIDES the checkpoint's recorded %s",
                    args.model, meta["model"])
    log.info(f"Checkpoint task: {task_type}  model: {model_name}  epoch: {meta.get('epoch')}")

    from opentouch_train.train import ALL_TASKS
    if task_type == "all":
        task_list = ALL_TASKS
        all_mods = ["visual", "tactile", "pose"]
        modality_flags = {"include_visual": True, "include_tactile": True, "include_pose": True}
    else:
        query_mods, target_mods = parse_task(task_type)
        task_list = [(query_mods, target_mods)]
        all_mods = list(set(query_mods) | set(target_mods))
        modality_flags = _determine_modality_flags(task_type)

    model = create_model(
        model_name,
        pretrained=args.checkpoint,
        precision=args.precision,
        device=device,
        enabled_modalities=all_mods,
    )
    model.eval()

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    # Match eval.py: the split geometry must be the one the model was trained
    # against, or the gallery is a different set of clips and the mAP is not
    # the number being quoted. An older checkpoint predating the field falls
    # back to the historical 'clip' behaviour.
    split_group_by = args.split_group_by or meta.get("split_group_by") or "clip"
    if args.split_group_by and meta.get("split_group_by") and args.split_group_by != meta["split_group_by"]:
        log.warning(
            "--split-group-by=%s OVERRIDES the checkpoint's recorded %s -- the gallery "
            "will not be the split this model was trained against.",
            args.split_group_by, meta["split_group_by"],
        )
    # No retrieval checkpoint in this project actually records split_group_by --
    # the field postdates all of them. Falling back to 'clip' silently is how a
    # scene-trained model gets scored on a gallery full of its own training
    # participants: p2t_scene_gru read 72.09 mAP that way instead of its true
    # 28.31. The fallback is still the historical default, but it is never
    # allowed to be quiet again.
    if not args.split_group_by and not meta.get("split_group_by"):
        log.warning(
            "This checkpoint does not record split_group_by, so 'clip' is being ASSUMED. "
            "If it was trained scene-disjoint, pass --split-group-by scene explicitly -- "
            "otherwise its gallery contains participants it trained on and the mAP is "
            "inflated, not merely approximate."
        )
        split_group_by_source = "assumed (checkpoint records none)"
    else:
        split_group_by_source = "cli" if args.split_group_by else "checkpoint"
    log.info(f"Bootstrapping with split_group_by={split_group_by} "
             f"({split_group_by_source}), cluster unit={args.cluster}")

    dataset = VideoTactilePoseDataset(
        hf_dataset_path=args.data,
        split=args.split,
        sequence_length=20,
        image_size=(224, 224),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        split_group_by=split_group_by,
        **modality_flags,
    )

    # One cluster id per window, in dataloader order (shuffle=False below), so
    # positions line up with the extracted embeddings.
    clip_ids = np.asarray([clip_idx for clip_idx, _ in dataset.windows])
    n_clips = len(set(clip_ids.tolist()))
    log.info(f"{len(dataset)} windows from {n_clips} clips "
             f"({len(dataset) / max(n_clips, 1):.1f} windows/clip)")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    log.info(f"Test samples: {len(dataset)}")

    # extract embeddings once
    all_features = {mod: [] for mod in all_mods}
    with torch.inference_mode():
        for batch in dataloader:
            batch_tensors = {}
            from opentouch_train.data import MODALITY_TO_BATCH_KEY
            for mod in all_mods:
                key = MODALITY_TO_BATCH_KEY[mod]
                if key in batch:
                    t = batch[key]
                    if input_dtype is not None:
                        t = t.to(device=device, dtype=input_dtype, non_blocking=True)
                    else:
                        t = t.to(device=device, non_blocking=True)
                    batch_tensors[key] = t
            with autocast():
                model_out = model(**batch_tensors)
            for mod in all_mods:
                feat_key = MODALITY_TO_FEATURE_KEY[mod]
                if feat_key in model_out:
                    all_features[mod].append(model_out[feat_key].cpu())

    for mod in all_mods:
        if all_features[mod]:
            all_features[mod] = torch.cat(all_features[mod])

    log.info(f"Embeddings extracted. Running {args.n_bootstrap} bootstrap iterations per direction.")

    results = {}
    for query_mods, target_mods in task_list:
        query_label = "+".join(query_mods)
        target_label = "+".join(target_mods)
        fwd_key = f"{'_'.join(query_mods)}_to_{'_'.join(target_mods)}"
        rev_key = f"{'_'.join(target_mods)}_to_{'_'.join(query_mods)}"

        if not args.all_directions:
            if fwd_key not in POSE_INVOLVED_DIRECTIONS and rev_key not in POSE_INVOLVED_DIRECTIONS:
                log.info(f"Skipping {fwd_key} (not pose-involved)")
                continue

        if len(query_mods) == 1:
            query_features = all_features[query_mods[0]]
        else:
            encoded = {mod: all_features[mod].to(device) for mod in query_mods}
            with torch.no_grad():
                query_features = model.fuse_encoded_features(encoded, target_mods[0]).detach().cpu()

        target_features = all_features[target_mods[0]].clone()

        for direction, q_emb, t_emb in [
            (fwd_key, query_features, target_features),
            (rev_key, target_features, query_features),
        ]:
            if not args.all_directions and direction not in POSE_INVOLVED_DIRECTIONS:
                continue

            log.info(f"Bootstrapping {direction}...")
            maps = bootstrap_map(
                q_emb, t_emb, args.n_bootstrap, rng,
                clip_ids=clip_ids if args.cluster == "clip" else None,
            )
            mean = float(np.mean(maps))
            std = float(np.std(maps))
            ci_lo = float(np.percentile(maps, 2.5))
            ci_hi = float(np.percentile(maps, 97.5))
            results[direction] = {
                "mean_mAP": round(mean * 100, 3),
                "std_mAP": round(std * 100, 3),
                "ci_95_lo": round(ci_lo * 100, 3),
                "ci_95_hi": round(ci_hi * 100, 3),
                "n_bootstrap": args.n_bootstrap,
                "n_test_samples": len(query_features),
                "cluster_unit": args.cluster,
                "n_clusters": n_clips,
                "split": args.split,
                "split_group_by": split_group_by,
                "split_group_by_source": split_group_by_source,
                "gallery": "fixed (queries resampled only)",
            }
            log.info(
                f"  {direction}: mean={mean*100:.2f}  std={std*100:.2f}  "
                f"95% CI=[{ci_lo*100:.2f}, {ci_hi*100:.2f}]"
            )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
