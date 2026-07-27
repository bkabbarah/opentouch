"""Training loop for the pose-transition regression task."""

from __future__ import annotations

import json
import logging
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from opentouch import get_input_dtype
from opentouch.articulation_frames import (
    decompose_rigid,
    hand_frame_basis,
    to_hand_frame,
)
from opentouch.pose_regression import decompose_world_delta
from opentouch.regression_metrics import (
    _metrics_given_mask,
    compute_dual_target_metrics,
    fingertip_displacement,
)
from opentouch_train.distributed import is_master
from opentouch_train.precision import get_autocast
from opentouch_train.train import AverageMeter, unwrap_model, backward


def _extract_batch(batch, use_tactile, device, input_dtype):
    """Move pose_t/world_delta/tactile_pressure to device and decompose the
    world-space delta into (wrist_delta, articulation_delta).

    In pose-only runs (use_tactile=False), the dataset was built with
    include_tactile=False, so "tactile_pressure" is simply absent from the
    batch dict -- there is no tensor here to zero out or ignore. The asserts
    below catch a config mismatch (e.g. --pose-only flag not matched by the
    dataset build) loudly rather than silently training/evaluating the
    wrong condition. The non-zero check on tactile_pressure additionally
    guards the --shuffle-tactile path: shuffled tactile is still REAL
    tactile (from a different window), never a placeholder zero tensor.
    """
    kwargs = dict(device=device, non_blocking=True)
    if input_dtype is not None:
        kwargs["dtype"] = input_dtype
    pose_t = batch["pose_t"].to(**kwargs)
    world_delta = batch["world_delta"].to(**kwargs)
    _, articulation_delta = decompose_world_delta(world_delta)

    # Third target space: whole-hand ROTATION removed as well as translation,
    # expressed in palm axes. articulation_delta keeps rigid rotation, which is
    # ~95% of its energy at the median sample, so a model trained against it is
    # mostly learning to extrapolate wrist re-orientation. pose_future is
    # recoverable from what the dataset already emits, so no dataset change is
    # needed. Done in float32 regardless of autocast: the Kabsch SVD is
    # ill-conditioned on near-planar hands in half precision.
    with torch.autocast(device_type=pose_t.device.type, enabled=False):
        pose_t32 = pose_t.float()
        pose_future32 = pose_t32 + world_delta.float()
        _, rigid_residual = decompose_rigid(pose_t32, pose_future32)
        rigid_delta = to_hand_frame(rigid_residual, hand_frame_basis(pose_t32))
    rigid_delta = rigid_delta.to(world_delta.dtype)

    if use_tactile:
        assert "tactile_pressure" in batch, (
            "use_tactile=True but the batch has no tactile_pressure key -- the "
            "dataset must be built with include_tactile=True (i.e. NOT --pose-only)"
        )
        tactile_pressure = batch["tactile_pressure"].to(**kwargs)
        assert tactile_pressure.abs().sum().item() > 0, (
            "tactile_pressure batch is all-zero -- use_tactile=True (plain or "
            "--shuffle-tactile) must always see real, non-placeholder tactile "
            "data; this is not the pose-only path"
        )
    else:
        assert "tactile_pressure" not in batch, (
            "pose-only run (use_tactile=False) but the batch contains "
            "tactile_pressure -- the dataset must be built with "
            "include_tactile=False for --pose-only runs"
        )
        tactile_pressure = None
    return pose_t, world_delta, articulation_delta, rigid_delta, tactile_pressure


def _select_target(
    world_delta: torch.Tensor,
    articulation_delta: torch.Tensor,
    rigid_delta: torch.Tensor,
    target_mode: str,
) -> torch.Tensor:
    """Which delta space the loss is computed in.

    'rigid_articulation' is the corrected target: whole-hand rotation removed
    and expressed in palm axes. The frozen-feature probes show tactile's
    contribution is roughly ten times larger against this space than against
    'articulation_delta', which retains the rotation.
    """
    if target_mode == "articulation_delta":
        return articulation_delta
    if target_mode == "world_delta":
        return world_delta
    if target_mode == "rigid_articulation":
        return rigid_delta
    raise ValueError(
        f"Unknown target_mode {target_mode!r}, expected 'world_delta', "
        "'articulation_delta' or 'rigid_articulation'"
    )


def _clip_gradients(model, max_norm: float, scope: str) -> None:
    """Gradient clipping, with the scope made explicit because it silently
    couples the two conditions being compared.

    'global' (the historical default) clips the L2 norm over EVERY parameter
    at once. The tactile model carries ~540k more parameters than the
    pose-only model, so its total norm is larger, so clipping fires more often
    and rescales the pose head's updates too. The pose head then sees a
    different effective learning rate in the two arms, and part of any
    measured difference is that, not tactile.

    'per_branch' clips the pose path and the tactile path separately, each
    against the same max_norm. The pose head's clipping is then identical
    whether or not a tactile branch exists, so the comparison isolates
    tactile rather than an optimisation side effect.
    """
    if scope == "global":
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2.0)
        return
    if scope != "per_branch":
        raise ValueError(f"grad_clip_scope must be 'global' or 'per_branch', got {scope!r}")

    pose_params, tactile_params = [], []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        (tactile_params if name.startswith(("tactile_encoder", "tactile_head", "gate"))
         else pose_params).append(param)
    if pose_params:
        torch.nn.utils.clip_grad_norm_(pose_params, max_norm, norm_type=2.0)
    if tactile_params:
        torch.nn.utils.clip_grad_norm_(tactile_params, max_norm, norm_type=2.0)


def _is_val_epoch(epoch: int, val_frequency: int, total_epochs: int) -> bool:
    if not val_frequency:
        return True
    return (epoch % val_frequency) == 0 or epoch == total_epochs


def train_one_epoch_regression(model, data, epoch, optimizer, scaler, scheduler, args):
    """Run one training epoch. Loss is plain MSE on the 21 3D keypoint deltas
    in whichever space --target-mode selects (default articulation_delta)."""
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    use_tactile = not args.pose_only

    data["train"].set_epoch(epoch)
    dataloader = data["train"].dataloader
    num_batches = dataloader.num_batches

    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    pbar = tqdm(
        enumerate(dataloader), total=num_batches,
        desc=f"Epoch {epoch}", disable=not is_master(args),
    )
    for i, batch in pbar:
        step = num_batches * epoch + i
        if not args.skip_scheduler and scheduler is not None:
            scheduler(step)

        pose_t, world_delta, articulation_delta, rigid_delta, tactile_pressure = _extract_batch(
            batch, use_tactile, device, input_dtype,
        )
        target = _select_target(world_delta, articulation_delta, rigid_delta, args.target_mode)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            pred_delta = model(pose_t, tactile_pressure)
            loss = F.mse_loss(pred_delta, target)

        backward(loss, scaler)

        clip_scope = getattr(args, "grad_clip_scope", "global")
        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                _clip_gradients(model, args.grad_clip_norm, clip_scope)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                _clip_gradients(model, args.grad_clip_norm, clip_scope)
            optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()

        batch_size = pose_t.shape[0]
        losses_m.update(loss.item(), batch_size)

        if is_master(args) and ((i % args.log_every_n_steps == 0) or (i + 1 == num_batches)):
            samples_per_second = args.batch_size * args.world_size / batch_time_m.val
            pbar.set_postfix(
                loss=f"{losses_m.avg:.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                sps=f"{samples_per_second:.0f}",
            )
            log_data = {
                "train/loss": losses_m.val,
                "train/loss_avg": losses_m.avg,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/data_time": data_time_m.val,
                "train/batch_time": batch_time_m.val,
                "train/samples_per_second": samples_per_second,
            }
            if args.wandb:
                assert wandb is not None, "Please install wandb."
                log_data["step"] = step
                wandb.log(log_data, step=step)
            batch_time_m.reset()
            data_time_m.reset()


def _flatten_dual_metrics(dual_metrics: dict) -> dict:
    """{"world": {...}, "articulation": {...}, "wrist_translation_mse": x}
    -> flat {"world_*": ..., "articulation_*": ..., "wrist_translation_mse": x}
    for logging/results.jsonl (wandb.log and json.dumps both want flat dicts)."""
    flat = {"wrist_translation_mse": dual_metrics["wrist_translation_mse"]}
    for space in ("world", "articulation"):
        for key, val in dual_metrics[space].items():
            flat[f"{space}_{key}"] = val
    return flat


def evaluate_regression(model, data, epoch, args):
    """Evaluate on validation data. ALWAYS reports metrics for BOTH
    world_delta and articulation_delta target spaces (regardless of
    --target-mode) plus the wrist-translation error on its own -- see
    opentouch.regression_metrics.compute_dual_target_metrics. The
    moving-subset breakdown (within each space) uses args.motion_threshold,
    which by this point is always a resolved float (either user-supplied or
    computed once from the train split by regression_main.py before the
    epoch loop -- never recomputed here).
    """
    metrics = {}
    if not is_master(args):
        return metrics
    if "val" not in data:
        return metrics
    if not _is_val_epoch(epoch, args.val_frequency, args.epochs):
        return metrics

    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    use_tactile = not args.pose_only

    eval_model = unwrap_model(model) if args.distributed else model
    eval_model.eval()

    dataloader = data["val"].dataloader
    all_pred, all_world, all_articulation, all_rigid = [], [], [], []
    with torch.inference_mode():
        for batch in dataloader:
            pose_t, world_delta, articulation_delta, rigid_delta, tactile_pressure = _extract_batch(
                batch, use_tactile, device, input_dtype,
            )
            with autocast():
                pred_delta = eval_model(pose_t, tactile_pressure)
            all_pred.append(pred_delta.float().cpu())
            all_world.append(world_delta.float().cpu())
            all_articulation.append(articulation_delta.float().cpu())
            all_rigid.append(rigid_delta.float().cpu())

    all_pred_t = torch.cat(all_pred)
    all_world_t = torch.cat(all_world)
    all_articulation_t = torch.cat(all_articulation)
    all_rigid_t = torch.cat(all_rigid)

    dual_metrics = compute_dual_target_metrics(
        all_pred_t, all_world_t, all_articulation_t, motion_threshold=args.motion_threshold,
    )
    # The rigid space is scored on the SAME moving subset as the other two --
    # the mask comes from articulation displacement in every case -- so the
    # eval population is identical to every previously reported run and the
    # numbers stay comparable.
    moving_mask = None
    if args.motion_threshold is not None:
        moving_mask = fingertip_displacement(all_articulation_t) >= args.motion_threshold
    dual_metrics["rigid_articulation"] = _metrics_given_mask(
        all_pred_t, all_rigid_t, moving_mask, args.motion_threshold,
    )
    metrics = _flatten_dual_metrics(dual_metrics)
    for key, value in dual_metrics["rigid_articulation"].items():
        metrics["rigid_articulation_%s" % key] = value
    trained_space = dual_metrics.get(
        args.target_mode, dual_metrics.get(args.target_mode.replace("_delta", ""))
    )
    metrics["val_loss"] = trained_space["all_mse_all_joints"]
    metrics["epoch"] = epoch

    # The residual-fusion gate (PoseTransitionRegressor.gate) is None in
    # pose-only mode and a learnable scalar, zero-initialized, in
    # tactile+pose mode. Logged every eval unconditionally -- a gate that
    # stays ~0 is itself the finding (tactile carries no transition signal
    # beyond what pose already implies), not a failure to be hidden.
    if use_tactile:
        gate_value = eval_model.gate.detach().float().mean().item()
        metrics["gate_value"] = gate_value

    logging.info(
        f"Eval Epoch: {epoch}  target_mode={args.target_mode}  val_loss({args.target_mode}): "
        f"{metrics['val_loss']:.6f}  wrist_translation_mse: {dual_metrics['wrist_translation_mse']:.6f}\n"
        f"  [rigid]        all mse_fingertips: {dual_metrics['rigid_articulation']['all_mse_fingertips']:.6f}  "
        f"copy_baseline: {dual_metrics['rigid_articulation']['all_copy_baseline_mse_fingertips']:.6f}  |  "
        f"moving mse_fingertips: {dual_metrics['rigid_articulation'].get('moving_mse_fingertips', float('nan')):.6f}  "
        f"copy_baseline: {dual_metrics['rigid_articulation'].get('moving_copy_baseline_mse_fingertips', float('nan')):.6f}\n"
        f"  [world]        all mse_fingertips: {dual_metrics['world']['all_mse_fingertips']:.6f}  "
        f"copy_baseline: {dual_metrics['world']['all_copy_baseline_mse_fingertips']:.6f}  |  "
        f"moving mse_fingertips: {dual_metrics['world'].get('moving_mse_fingertips', float('nan')):.6f}  "
        f"copy_baseline: {dual_metrics['world'].get('moving_copy_baseline_mse_fingertips', float('nan')):.6f}\n"
        f"  [articulation] all mse_fingertips: {dual_metrics['articulation']['all_mse_fingertips']:.6f}  "
        f"copy_baseline: {dual_metrics['articulation']['all_copy_baseline_mse_fingertips']:.6f}  |  "
        f"moving mse_fingertips: {dual_metrics['articulation'].get('moving_mse_fingertips', float('nan')):.6f}  "
        f"copy_baseline: {dual_metrics['articulation'].get('moving_copy_baseline_mse_fingertips', float('nan')):.6f}  "
        f"(moving/{int(dual_metrics['articulation']['num_samples'])}="
        f"{int(dual_metrics['articulation'].get('num_moving_samples', 0))}, "
        f"threshold={args.motion_threshold})"
    )
    if use_tactile:
        logging.info(f"  residual-fusion gate: {metrics['gate_value']:.6f}")

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, "Please install wandb."
        log_data = {"val/" + name: val for name, val in metrics.items()}
        if "train" in data:
            wandb_step = data["train"].dataloader.num_batches * epoch
        else:
            wandb_step = None
        log_data["epoch"] = epoch
        wandb.log(log_data, step=wandb_step)

    return metrics
