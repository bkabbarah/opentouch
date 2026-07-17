"""Main training entry point for the pose-transition regression task.

BASELINE-FIRST WORKFLOW: run this with --pose-only first and record its
val metrics before ever training a tactile+pose run. That pose-only number
is the bar -- if tactile+pose does not beat it, tactile does not predict
pose transitions beyond what pose alone implies, and that is the finding.
Do not tune the tactile+pose run's hyperparameters against the pose-only
baseline after the fact; both must use the same head, training config, and
--split-seed for the comparison to mean anything.
"""

from __future__ import annotations

import glob
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from torch import optim

try:
    import wandb
except ImportError:
    wandb = None

from opentouch.factory import natural_key
from opentouch.pose_regression import PoseTransitionRegressor
from opentouch_train.distributed import is_master, init_distributed_device, broadcast_object
from opentouch_train.logger import setup_logging
from opentouch_train.regression_data import compute_motion_threshold, get_regression_data
from opentouch_train.regression_params import parse_regression_args
from opentouch_train.regression_train import train_one_epoch_regression, evaluate_regression
from opentouch_train.scheduler import cosine_lr, const_lr


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_latest_checkpoint(path: str):
    checkpoints = glob.glob(os.path.join(path, "**", "*.pt"), recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def _adapt_state_dict_keys(state_dict, distributed):
    has_module_prefix = next(iter(state_dict)).startswith("module.")
    if distributed and not has_module_prefix:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif not distributed and has_module_prefix:
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _build_experiment_name(args) -> str:
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if args.distributed:
        date_str = broadcast_object(args, date_str)
    if args.pose_only:
        mode = "pose_only"
    elif args.shuffle_tactile:
        mode = "shuffle_tactile"
    else:
        mode = "tactile_pose"
    return "-".join([
        date_str, "pose_regression", mode, f"k{args.horizon_k}",
        f"lr_{args.lr}", f"b_{args.batch_size}", f"p_{args.precision}",
    ])


def _build_optimizer(model, args):
    def is_excluded(name, param):
        return param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if is_excluded(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if not is_excluded(n, p) and p.requires_grad]
    return optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
    )


def main(args):
    args = parse_regression_args(args)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)

    if args.name is None:
        args.name = _build_experiment_name(args)

    resume_latest = args.resume == "latest"
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print("Error. Experiment already exists. Use --name to specify a new experiment.")
            return -1

    args.log_level = logging.INFO
    setup_logging(args.log_path, args.log_level)

    if args.debug:
        args.report_to = ""
        args.save_most_recent = False
        args.save_frequency = 0
        logging.getLogger("torch").setLevel(logging.WARNING)

    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        os.makedirs(args.checkpoint_path, exist_ok=True)

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        if is_master(args):
            if args.save_most_recent:
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    resume_from = None
            else:
                resume_from = get_latest_checkpoint(checkpoint_path)
            if resume_from:
                logging.info(f"Found latest resume checkpoint at {resume_from}.")
            else:
                logging.info(f"No latest resume checkpoint found in {checkpoint_path}.")
        if args.distributed:
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.distributed:
        logging.info(
            f"Running in distributed mode. Device: {args.device}. "
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    random_seed(args.seed, 0)

    logging.info(
        f"Pose-transition regression: horizon_k={args.horizon_k}  target_mode={args.target_mode}  "
        f"pose_only={args.pose_only}  shuffle_tactile={args.shuffle_tactile}"
    )
    if args.pose_only:
        logging.info(
            "POSE-ONLY BASELINE run. This must be run and its val metrics recorded "
            "BEFORE any tactile+pose run's numbers are trusted -- see module docstring."
        )
    if args.shuffle_tactile:
        logging.info(
            "SHUFFLED-TACTILE CONTROL run: tactile is real but paired with a "
            "different window's pose (fixed derangement, deterministic given "
            "--split-seed). Same architecture and parameter count as the real "
            "tactile+pose model -- this isolates tactile CONTENT from raw capacity."
        )

    data = get_regression_data(args)
    assert "train" in data, "Training data is required."

    if args.motion_threshold is None:
        # Computed ONCE, here, before the epoch loop -- never recomputed per
        # epoch. Always from ARTICULATION displacement (see
        # opentouch_train.regression_data.compute_motion_threshold), never
        # raw world-space displacement.
        computed_threshold = compute_motion_threshold(data["train"].dataloader.dataset, percentile=25.0)
        args.motion_threshold = computed_threshold
        logging.info(
            f"--motion-threshold not given: computed 25th percentile of median-fingertip "
            f"ARTICULATION displacement over the train split at k={args.horizon_k}: "
            f"{computed_threshold:.6f}"
        )
    else:
        logging.info(f"--motion-threshold explicitly set, overriding auto-computation: {args.motion_threshold}")

    model = PoseTransitionRegressor(
        use_tactile=not args.pose_only,
        tactile_emb_dim=args.tactile_emb_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    random_seed(args.seed, args.rank)

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Trainable parameters: {num_params:,}")
        if args.shuffle_tactile:
            # Hard equality, not a tolerance -- shuffle_tactile never changes
            # the model constructor args (only which window's tactile the
            # DATASET pairs with which window's pose), so this model is
            # architecturally IDENTICAL to a plain tactile+pose model built
            # with the same tactile_emb_dim/hidden_dim.
            reference = PoseTransitionRegressor(
                use_tactile=True, tactile_emb_dim=args.tactile_emb_dim, hidden_dim=args.hidden_dim,
            )
            reference_params = sum(p.numel() for p in reference.parameters() if p.requires_grad)
            assert num_params == reference_params, (
                f"shuffle_tactile model has {num_params:,} params but a plain tactile+pose "
                f"model with the same tactile_emb_dim/hidden_dim has {reference_params:,} -- "
                "these must be EXACTLY equal for the shuffled-tactile control to isolate "
                "content from capacity"
            )
            logging.info(f"Parameter parity with tactile+pose confirmed: {num_params:,} == {reference_params:,}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    optimizer = _build_optimizer(model, args)

    scaler = None
    if args.precision == "amp":
        try:
            scaler = torch.amp.GradScaler(device=device)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
            state_dict = _adapt_state_dict_keys(checkpoint["state_dict"], args.distributed)
            model.load_state_dict(state_dict, strict=True)
            if optimizer is not None and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            state_dict = _adapt_state_dict_keys(checkpoint, args.distributed)
            model.load_state_dict(state_dict, strict=True)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    scheduler = None
    total_steps = data["train"].dataloader.num_batches * args.epochs
    warmup_steps = int(args.warmup * total_steps) if args.warmup < 1 else int(args.warmup)
    logging.info(f"Warmup steps: {warmup_steps} / {total_steps} total steps")
    if args.lr_scheduler == "cosine":
        scheduler = cosine_lr(optimizer, args.lr, warmup_steps, total_steps)
    elif args.lr_scheduler == "const":
        scheduler = const_lr(optimizer, args.lr, warmup_steps, total_steps)

    args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
    if args.wandb and is_master(args):
        assert wandb is not None, "Please install wandb."
        args.train_sz = data["train"].dataloader.num_samples
        if "val" in data:
            args.val_sz = data["val"].dataloader.num_samples
        wandb.init(
            project=args.wandb_project_name, name=args.name, id=args.name,
            notes=args.wandb_notes, resume="auto" if args.resume == "latest" else None,
            config=vars(args),
        )
        if is_master(args):
            params_file = os.path.join(args.logs, args.name, "params.txt")
            if os.path.exists(params_file):
                wandb.save(params_file)

    original_model = model

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch_regression(model, data, epoch, optimizer, scaler, scheduler, args)
        completed_epoch = epoch + 1

        if "val" in data:
            evaluate_regression(model, data, completed_epoch, args)
            if args.distributed:
                torch.distributed.barrier()

        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                # Metadata a standalone eval run needs to reconstruct the
                # EXACT model/data config -- mirrors what main.py records
                # for tactile_encoder_type/tactile_b_matrices_path.
                "task_type": args.task_type,
                "horizon_k": args.horizon_k,
                "target_mode": args.target_mode,
                "pose_only": args.pose_only,
                "shuffle_tactile": args.shuffle_tactile,
                # Always the RESOLVED value (user-supplied or computed once
                # above) -- eval must reuse this exact number, not recompute it.
                "motion_threshold": args.motion_threshold,
                "tactile_emb_dim": args.tactile_emb_dim,
                "hidden_dim": args.hidden_dim,
                "sequence_length": args.sequence_length,
                "split_seed": args.split_seed,
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"))
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)
            if args.save_most_recent:
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

        if args.distributed:
            torch.distributed.barrier()

    if args.wandb and is_master(args):
        wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
