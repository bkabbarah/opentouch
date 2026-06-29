"""Standalone evaluation for retrieval checkpoints.

Usage::
    python -m opentouch_train.eval \
        --checkpoint logs/multi_gpu_v2t/checkpoints/epoch_30.pt \
        --data preprocessed_data/train_dataset
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from opentouch import create_model, compute_retrieval_metrics, get_input_dtype
from opentouch_train.data import (
    VideoTactilePoseDataset, collate_fn, parse_task, _determine_modality_flags,
)
from opentouch_train.precision import get_autocast


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _read_params_file(params_file: Path) -> dict:
    params = {}
    if not params_file.exists():
        return params
    for line in params_file.read_text().splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            params[key.strip()] = value.strip()
    return params



def _read_checkpoint_meta(path):
    """Read metadata from checkpoint, falling back to params.txt in the log dir."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = {
        "task_type": ckpt.get("task_type"),
        "model": ckpt.get("model"),
        "epoch": ckpt.get("epoch"),
    }

    if meta["task_type"] is None or meta["model"] is None:
        params_file = Path(path).resolve().parent.parent / "params.txt"
        params = _read_params_file(params_file)
        if meta["task_type"] is None:
            meta["task_type"] = params.get("task_type")
        if meta["model"] is None:
            meta["model"] = params.get("model")

    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a retrieval checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    p.add_argument("--data", required=True, help="Path to preprocessed HF dataset.")
    p.add_argument("--model", default=None, help="Model config name (auto-detected from checkpoint).")
    p.add_argument("--task-type", default=None, help="Retrieval task (auto-detected from checkpoint).")
    p.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--precision", default="amp", choices=["fp32", "amp", "bf16"])
    p.add_argument("--sequence-length", type=int, default=20)
    p.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio (must match training).")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio (must match training).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split (must match training).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None, help="Optional path to save metrics JSON.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    meta = _read_checkpoint_meta(args.checkpoint)
    if args.task_type is None:
        args.task_type = meta.get("task_type")
        if args.task_type is None:
            raise ValueError(
                "Checkpoint does not contain task_type. "
                "Specify --task-type explicitly (v2t, p2t, v2p, vp2t, …)."
            )
    if args.model is None:
        args.model = meta.get("model") or "OpenTouch-DINOv3-B16-Retrieval"

    log.info(f"Model: {args.model}  Task: {args.task_type}  Epoch: {meta.get('epoch', '?')}")

    device = torch.device(args.device)
    from opentouch_train.train import ALL_TASKS
    if args.task_type == "all":
        task_list = ALL_TASKS
        modality_flags = {"include_visual": True, "include_tactile": True, "include_pose": True}
        all_mods = ["visual", "tactile", "pose"]
    else:
        query_mods, target_mods = parse_task(args.task_type)
        task_list = [(query_mods, target_mods)]
        modality_flags = _determine_modality_flags(args.task_type)
        all_mods = list(set(query_mods) | set(target_mods))

    model = create_model(args.model, pretrained=args.checkpoint, precision=args.precision, device=device)
    model.eval()

    dataset = VideoTactilePoseDataset(
        hf_dataset_path=args.data,
        split=args.split,
        sequence_length=args.sequence_length,
        image_size=(224, 224),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        **modality_flags,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn, persistent_workers=args.workers > 0,
    )
    log.info(f"Split: {args.split}  samples: {len(dataset)}  batches: {len(dataloader)}")

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    from opentouch_train.data import MODALITY_TO_FEATURE_KEY
    all_features = {"visual": [], "tactile": [], "pose": []}
    all_metadata = []
    logit_scale_val = None

    from opentouch_train.train import _extract_batch_tensors

    with torch.inference_mode():
        for batch in dataloader:
            batch_tensors = _extract_batch_tensors(batch, all_mods, device, input_dtype)
            with autocast():
                model_out = model(**batch_tensors)
                if logit_scale_val is None:
                    logit_scale_val = model_out["logit_scale"].mean().cpu()
                for mod in all_mods:
                    feat_key = MODALITY_TO_FEATURE_KEY[mod]
                    if feat_key in model_out:
                        all_features[mod].append(model_out[feat_key].cpu())
                if "scene" in batch:
                    all_metadata.extend(zip(batch["scene"], batch["clip_id"]))

    for mod in all_mods:
        if all_features[mod]:
            all_features[mod] = torch.cat(all_features[mod])

    metrics = {"split": args.split}
    print(f"\n{'='*60}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split      : {args.split}")
    print(f"{'='*60}")

    for query_mods, target_mods in task_list:
        query_label = "+".join(query_mods)
        target_label = "+".join(target_mods)
        query_key = query_label.replace("+", "_")
        target_key = target_label.replace("+", "_")
        fwd = f"{query_key}_to_{target_key}"
        rev = f"{target_key}_to_{query_key}"

        if len(query_mods) == 1:
            query_features = all_features[query_mods[0]]
        else:
            encoded = {mod: all_features[mod].to(device) for mod in query_mods}
            with torch.no_grad():
                query_features = model.fuse_encoded_features(encoded, target_mods[0]).detach().cpu()

        target_features = all_features[target_mods[0]].clone()

        retrieval_metrics = compute_retrieval_metrics(
            query_features, target_features, top_k=[1, 5, 10],
            query_label=query_label, target_label=target_label,
        )
        for direction, values in retrieval_metrics.items():
            for name, value in values.items():
                metrics[f"{direction}_{name}"] = value

        print(f"\n  {query_label} -> {target_label}")
        print(f"    R@1  : {metrics.get(f'{fwd}_recall@1', 0):.4f}")
        print(f"    R@5  : {metrics.get(f'{fwd}_recall@5', 0):.4f}")
        print(f"    R@10 : {metrics.get(f'{fwd}_recall@10', 0):.4f}")
        print(f"    mAP  : {metrics.get(f'{fwd}_mAP', 0):.4f}")
        print(f"  {target_label} -> {query_label}")
        print(f"    R@1  : {metrics.get(f'{rev}_recall@1', 0):.4f}")
        print(f"    R@5  : {metrics.get(f'{rev}_recall@5', 0):.4f}")
        print(f"    R@10 : {metrics.get(f'{rev}_recall@10', 0):.4f}")
        print(f"    mAP  : {metrics.get(f'{rev}_mAP', 0):.4f}")

    print(f"\n{'='*60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Saved metrics to {args.output}")

    return metrics


if __name__ == "__main__":
    main()
