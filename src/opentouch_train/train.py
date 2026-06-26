"""Training loop for OpenTouch cross-modal retrieval."""

import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from opentouch import get_input_dtype
from opentouch.constants import LOGIT_SCALE_MIN, LOGIT_SCALE_MAX
from opentouch.metrics import compute_retrieval_metrics
from opentouch_train.data import (
    parse_task,
    MODALITY_TO_BATCH_KEY,
    MODALITY_TO_FEATURE_KEY,
)
from opentouch_train.distributed import is_master
from opentouch_train.precision import get_autocast


ALL_TASKS = [
    (["visual"], ["tactile"]),
    (["pose"], ["tactile"]),
    (["visual"], ["pose"]),
    (["visual", "pose"], ["tactile"]),
    (["tactile", "pose"], ["visual"]),
    (["visual", "tactile"], ["pose"]),
]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def _extract_batch_tensors(batch, modality_names, device, input_dtype):
    """Move modality tensors from batch dict to device."""
    tensors = {}
    kwargs = dict(device=device, non_blocking=True)
    if input_dtype is not None:
        kwargs["dtype"] = input_dtype
    for mod in modality_names:
        batch_key = MODALITY_TO_BATCH_KEY[mod]
        if batch_key in batch:
            tensors[batch_key] = batch[batch_key].to(**kwargs)
    return tensors


def _get_query_target_features(model_out, query_mods, target_mods, model):
    """Extract query/target features from model output, fusing if multi-modal query."""
    if len(target_mods) == 1:
        target_features = model_out[MODALITY_TO_FEATURE_KEY[target_mods[0]]]
    else:
        raise ValueError(f"Multiple target modalities not supported: {target_mods}")

    if len(query_mods) == 1:
        query_features = model_out[MODALITY_TO_FEATURE_KEY[query_mods[0]]]
    else:
        raw_model = unwrap_model(model)
        encoded = {}
        for mod in query_mods:
            encoded[mod] = model_out[MODALITY_TO_FEATURE_KEY[mod]]
        query_features = raw_model.fuse_encoded_features(encoded, target_mods[0])

    return query_features, target_features


def _compute_multitask_loss(model_out, task_list, loss, logit_scale, model, device):
    """Compute summed InfoNCE loss over all task pairs."""
    total_loss = torch.tensor(0.0, device=device)
    losses_dict = {}
    for query_mods, target_mods in task_list:
        query_features, target_features = _get_query_target_features(
            model_out, query_mods, target_mods, model,
        )
        task_losses = loss(
            query_features,
            target_features,
            logit_scale,
            logit_bias=model_out.get("logit_bias"),
            output_dict=True,
        )
        task_key = f"{'_'.join(query_mods)}_to_{'_'.join(target_mods)}"
        for k, v in task_losses.items():
            losses_dict[f"{task_key}/{k}"] = v
        total_loss = total_loss + sum(task_losses.values())
    losses_dict["loss"] = total_loss
    return total_loss, losses_dict


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    if args.task_type == "all":
        task_list = ALL_TASKS
        all_mods = ["visual", "tactile", "pose"]
    else:
        query_mods, target_mods = parse_task(args.task_type)
        task_list = [(query_mods, target_mods)]
        all_mods = list(set(query_mods) | set(target_mods))

    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_batches, accum_features = [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    pbar = tqdm(
        enumerate(dataloader),
        total=num_batches_per_epoch * args.accum_freq,
        desc=f"Epoch {epoch}",
        disable=not is_master(args),
    )
    for i, batch in pbar:
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler and scheduler is not None:
            scheduler(step)

        batch_tensors = _extract_batch_tensors(batch, all_mods, device, input_dtype)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(**batch_tensors)
                logit_scale = model_out["logit_scale"]
                total_loss, losses_dict = _compute_multitask_loss(
                    model_out, task_list, loss, logit_scale, model, device,
                )

            backward(total_loss, scaler)
        else:
            with torch.no_grad():
                with autocast():
                    model_out = model(**batch_tensors)
                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)
                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]
                accum_batches.append(batch_tensors)

            if ((i + 1) % args.accum_freq) > 0:
                continue

            optimizer.zero_grad()
            for j in range(args.accum_freq):
                batch_j = accum_batches[j]
                with autocast():
                    model_out = model(**batch_j)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(
                            accumulated[:j] + [model_out[key]] + accumulated[j + 1:]
                        )

                    merged_out = {**inputs, **inputs_no_accum}
                    total_loss, losses_dict = _compute_multitask_loss(
                        merged_out, task_list, loss, logit_scale, model, device,
                    )
                    del inputs, inputs_no_accum

                backward(total_loss, scaler)

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0,
                )
            optimizer.step()

        if args.accum_freq > 1:
            accum_batches, accum_features = [], {}

        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX)

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == num_batches_per_epoch
        ):
            batch_size = args.batch_size
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            for key, val in losses_dict.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item() if hasattr(val, 'item') else val, batch_size)

            logit_scale_scalar = logit_scale.item()
            samples_per_second = (
                args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            )

            pbar.set_postfix(
                loss=f"{losses_m['loss'].avg:.4f}" if 'loss' in losses_m else "N/A",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                sps=f"{samples_per_second:.0f}",
            )

            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": (
                    args.accum_freq * args.batch_size / batch_time_m.val
                ),
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_data.update({name: val.val for name, val in losses_m.items()})
            log_data = {"train/" + name: val for name, val in log_data.items()}

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step
                wandb.log(log_data, step=step)

            batch_time_m.reset()
            data_time_m.reset()


def evaluate(model, data, epoch, args, tb_writer=None):
    """Evaluate on validation data across all task pairs."""
    metrics = {}
    if not is_master(args):
        return metrics

    device = torch.device(args.device)
    model.eval()

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if args.task_type == "all":
        task_list = ALL_TASKS
        all_mods = ["visual", "tactile", "pose"]
    else:
        query_mods, target_mods = parse_task(args.task_type)
        task_list = [(query_mods, target_mods)]
        all_mods = list(set(query_mods) | set(target_mods))

    if "val" not in data:
        return metrics
    if not (args.val_frequency and (
        (epoch % args.val_frequency) == 0 or epoch == args.epochs
    )):
        return metrics

    eval_model = unwrap_model(model) if args.distributed else model
    dataloader = data['val'].dataloader

    # Encode all val samples once
    all_features = {mod: [] for mod in ["visual", "tactile", "pose"]}
    logit_scale_val = None

    with torch.inference_mode():
        for batch in dataloader:
            batch_tensors = _extract_batch_tensors(batch, all_mods, device, input_dtype)
            with autocast():
                model_out = eval_model(**batch_tensors)
                if logit_scale_val is None:
                    logit_scale_val = model_out["logit_scale"].mean().cpu()
                for mod in all_mods:
                    feat_key = MODALITY_TO_FEATURE_KEY[mod]
                    if feat_key in model_out:
                        all_features[mod].append(model_out[feat_key].cpu())

    for mod in all_mods:
        if all_features[mod]:
            all_features[mod] = torch.cat(all_features[mod])

    # Compute val loss and retrieval metrics for each task
    total_val_loss = 0.0
    num_tasks = 0

    for query_mods, target_mods in task_list:
        query_label = "+".join(query_mods)
        target_label = "+".join(target_mods)

        if len(query_mods) == 1:
            query_features = all_features[query_mods[0]]
        else:
            # fuse encoded features for multi-modal query
            encoded = {mod: all_features[mod] for mod in query_mods}
            query_features = eval_model.fuse_encoded_features(encoded, target_mods[0])

        target_features = all_features[target_mods[0]]
        num_samples = len(query_features)

        logits_q2t = logit_scale_val * query_features @ target_features.t()
        labels = torch.arange(num_samples).long()
        task_val_loss = (
            F.cross_entropy(logits_q2t, labels)
            + F.cross_entropy(logits_q2t.t(), labels)
        ) / 2
        total_val_loss += task_val_loss.item()
        num_tasks += 1

        retrieval_metrics = compute_retrieval_metrics(
            query_features,
            target_features,
            top_k=[1, 5, 10],
            query_label=query_label,
            target_label=target_label,
        )
        for direction, values in retrieval_metrics.items():
            for name, value in values.items():
                metrics[f"{direction}_{name}"] = value

    metrics["val_loss"] = total_val_loss / max(num_tasks, 1)
    metrics["epoch"] = epoch
    metrics["num_samples"] = num_samples

    # Log summary
    logging.info(f"Eval Epoch: {epoch}  avg_val_loss: {metrics['val_loss']:.4f}")
    for query_mods, target_mods in task_list:
        query_label = "+".join(query_mods)
        target_label = "+".join(target_mods)
        query_key = query_label.replace("+", "_")
        target_key = target_label.replace("+", "_")
        fwd = f"{query_key}_to_{target_key}"
        rev = f"{target_key}_to_{query_key}"
        logging.info(
            f"  {query_label}->{target_label}  "
            f"R@1/5/10: {metrics.get(f'{fwd}_recall@1', 0):.4f}/"
            f"{metrics.get(f'{fwd}_recall@5', 0):.4f}/"
            f"{metrics.get(f'{fwd}_recall@10', 0):.4f}  "
            f"mAP: {metrics.get(f'{fwd}_mAP', 0):.4f}\n"
            f"  {target_label}->{query_label}  "
            f"R@1/5/10: {metrics.get(f'{rev}_recall@1', 0):.4f}/"
            f"{metrics.get(f'{rev}_recall@5', 0):.4f}/"
            f"{metrics.get(f'{rev}_recall@10', 0):.4f}  "
            f"mAP: {metrics.get(f'{rev}_mAP', 0):.4f}"
        )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics