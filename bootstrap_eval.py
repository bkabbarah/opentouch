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
    return p.parse_args()


def bootstrap_map(query_emb: torch.Tensor, target_emb: torch.Tensor,
                  n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Bootstrap mAP by resampling (query_i, target_i) pairs with replacement.
    Returns array of shape (n,) containing mAP for each bootstrap sample.
    """
    query_emb = F.normalize(query_emb, dim=1)
    target_emb = F.normalize(target_emb, dim=1)
    num_samples = len(query_emb)
    maps = np.zeros(n)
    for i in range(n):
        idx = rng.integers(0, num_samples, size=num_samples)
        q = query_emb[idx]
        t = target_emb[idx]
        sim = q @ t.t()
        correct_sims = sim.diag().unsqueeze(1)
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
    model_name = meta.get("model")
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

    dataset = VideoTactilePoseDataset(
        hf_dataset_path=args.data,
        split=args.split,
        sequence_length=20,
        image_size=(224, 224),
        **modality_flags,
    )
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
            maps = bootstrap_map(q_emb, t_emb, args.n_bootstrap, rng)
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
