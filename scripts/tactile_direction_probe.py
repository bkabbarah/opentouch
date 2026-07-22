"""Frozen-feature linear probe: does the tactile embedding carry directional
articulation information, before committing to a full direction-
classification task?

MOTIVATION. The k=16 pose-transition regression null (see
src/opentouch/pose_regression.py, opentouch_train/regression_main.py) has a
confound: its tactile encoder is randomly initialized and trained
end-to-end on a scale-dominated MSE regression loss, so it may simply have
failed to learn useful features rather than tactile lacking signal. This
script instead takes the tactile encoder from a RETRIEVAL checkpoint (CNN+
biGRU, tactile_encoder.CNNetEmbedding, T->P mAP 45.46 -- see experiments.md
"Multi-seed validation"), FREEZES it, and asks a much weaker question: can a
plain sklearn logistic regression, given only the frozen 64-dim embedding,
predict the SIGN of each joint's wrist-relative articulation motion at
k=16? This sidesteps both confounds (encoder quality, MSE scale-domination)
at the cost of only asking about direction, not magnitude.

TEMPORAL LEAKAGE (fixed): this script originally computed ONE embedding per
T=20-frame window and broadcast it to every valid t in that window. At
k=16, valid t is 0..3, so t+k (the target frame) lands on frames 16..19 --
frames the broadcast embedding had ALREADY encoded. That measures
contemporaneous tactile-pose correspondence (already established: retrieval
T->P mAP 45.46), not prediction. Evidence it was leakage: tactile AUC was
nearly flat across horizons (~0.53 at k=2 through ~0.54 at k=16), where
genuine forward prediction must decay with horizon; at k=2 one fixed
embedding was serving 18 different t values with 18 different targets.
--causal (default True) fixes this: the embedding for sample (window, t) is
now computed from a FIXED-LENGTH causal window of frames [t-W+1, t] only
(edge-padded at the start by repeating frame 0), using the same
_causal_frame_indices arithmetic as opentouch_train.regression_data's fix
to the same leak in PoseTransitionDataset.__getitem__. --noncausal
reproduces the pre-fix numbers so the leakage magnitude stays measurable;
the report always shows both side by side.

Five conditions, each producing one probe per (joint, axis):
  - tactile_causal / tactile_noncausal:
        embedding from the SAME window as the pose target, causal or the
        pre-fix full-window broadcast, respectively.
  - shuffled_tactile_causal / shuffled_tactile_noncausal:
        embedding from a DIFFERENT window (fixed derangement,
        opentouch_train.regression_data._make_derangement -- the exact same
        control used by the regression task), i.e. real tactile carrying no
        information about *this* sample's motion. The capacity-matched
        null, causal or noncausal to match the tactile_* condition it pairs
        with.
  - pose:  the raw pose_t vector (63-dim) as a reference ceiling --
        information we already know is partially predictive, so probe
        scores here bound what a probe of this size/kind can achieve at
        all. Unaffected by causal/noncausal (no tactile involved).

MIN-HISTORY FILTERING (fixes a second problem the causal fix exposed): at
k=16 with T=20, valid t is 0..3, so causal samples have at most 1-4 REAL
(non-padded) frames -- t=0's causal window is a single static pressure map
repeated causal_window times, fed to a biGRU trained on 20-frame dynamics.
A near-chance AUC there is uninterpretable: "no tactile signal" and "no
tactile history was provided" look identical. It also confounds the k
sweep, since real history is capped at min(t+1, causal_window) and
valid_t=T-k, so max available history SHRINKS as k GROWS -- any AUC decay
across k is partly a history artifact, not (only) evidence against
long-horizon prediction. --min-history H (default 10) excludes every
(window, t) sample with fewer than H real frames -- applied ONCE, globally,
before any of the five conditions are built, so tactile_causal,
tactile_noncausal, shuffled_tactile_*, and pose are all scored on the
IDENTICAL retained sample set (same targets, same population): history is a
controlled variable for the whole comparison, not an implicit one that
differs per condition. At horizon_k=16, sequence_length=20, EVERY t has
real history <= 4 < 10, so min_history=10 leaves zero samples -- this
raises immediately with a message naming the fix (sequence_length >=
horizon_k + min_history), not a silently empty report.

Direction target is SIGN PER COORDINATE AXIS (3 binary targets per joint),
not a flexion-angle projection. A flexion-axis projection (motion along
each joint's actual rotation axis, from the MANO kinematic chain) would be
more physically meaningful -- articulation happens along a 1-DOF hinge, not
uniformly across x/y/z -- and is a natural follow-up if this per-axis probe
finds signal.

Does NOT train anything deep, and does NOT build the classification task --
this is a frozen-feature go/no-go check only.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from opentouch.pose_regression import (
    COORD_DIM,
    NUM_KEYPOINTS,
    WRIST_INDEX,
    decompose_world_delta,
)
from opentouch.regression_metrics import fingertip_displacement
from opentouch.tactile_encoder import CNNetEmbedding
from opentouch_train.data import VideoTactilePoseDataset, _load_and_split_dataset
from opentouch_train.regression_data import (
    PoseTransitionDataset,
    _causal_frame_indices,
    _make_derangement,
    compute_motion_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

AUC_CLEAR_THRESHOLD = 0.55


def _joint_labels() -> list[str]:
    """21-keypoint layout is wrist-first + 5 four-joint finger blocks
    (MCP/PIP/DIP/TIP) -- see pose_regression.py's WRIST_INDEX/FINGERTIP_COLUMNS
    derivation, the single source of truth for this layout."""
    labels = ["wrist"]
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        for joint in ("mcp", "pip", "dip", "tip"):
            labels.append(f"{finger}_{joint}")
    return labels


JOINT_LABELS = _joint_labels()
AXIS_LABELS = ("x", "y", "z")
assert len(JOINT_LABELS) == NUM_KEYPOINTS


def _read_params_file(params_file: Path) -> dict:
    params = {}
    if not params_file.exists():
        return params
    for line in params_file.read_text().splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            params[key.strip()] = value.strip()
    return params


def load_tactile_encoder(checkpoint_path: str, emb_dim: int, device: torch.device) -> CNNetEmbedding:
    """Load ONLY the 'tactile.*' submodule of a retrieval checkpoint's
    state_dict into a standalone, frozen CNNetEmbedding. strict=True: an
    architecture mismatch (e.g. a contact-encoder checkpoint, whose
    'tactile.*' keys don't match CNNetEmbedding at all) must raise, not
    silently leave this probe running on random weights.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        full_state = ckpt["state_dict"]
    elif "model" in ckpt:
        full_state = ckpt["model"]
    else:
        full_state = ckpt
    if next(iter(full_state)).startswith("module."):
        full_state = {k[len("module."):]: v for k, v in full_state.items()}

    tactile_encoder_type = ckpt.get("tactile_encoder_type")
    if tactile_encoder_type is None:
        params = _read_params_file(Path(checkpoint_path).resolve().parent.parent / "params.txt")
        tactile_encoder_type = params.get("tactile_encoder_type")
    if tactile_encoder_type not in (None, "cnn_gru"):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' was trained with "
            f"--tactile-encoder-type={tactile_encoder_type!r}, not 'cnn_gru'. This probe "
            "targets the plain CNN+biGRU tactile encoder specifically (the "
            "tactile_contact_encoder direction is dead -- see pose_regression.py's module "
            "docstring); pass a cnn_gru retrieval checkpoint instead."
        )

    tactile_state = {
        k[len("tactile."):]: v for k, v in full_state.items() if k.startswith("tactile.")
    }
    if not tactile_state:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' has no 'tactile.*' weights in its state_dict "
            "-- was this trained with tactile enabled?"
        )

    encoder = CNNetEmbedding(emb_dim=emb_dim)
    encoder.load_state_dict(tactile_state, strict=True)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    log.info(
        f"Loaded frozen tactile encoder: strict=True, {len(tactile_state)} tensors, from "
        f"checkpoint '{checkpoint_path}' (epoch={ckpt.get('epoch', '?')}, "
        f"tactile_encoder_type={tactile_encoder_type or 'cnn_gru (assumed, predates field)'})."
    )
    return encoder


@torch.no_grad()
def encode_windows(
    encoder: CNNetEmbedding, tactile: torch.Tensor, batch_size: int, device: torch.device,
) -> torch.Tensor:
    """(N, T, 1, 16, 16) -> (N, emb_dim): one embedding per whole T=20-frame
    window, unconditional on t. NONCAUSAL / PRE-FIX path: broadcasting this
    single embedding to every t in the window is exactly the leak (see
    module docstring) -- kept only so --noncausal can reproduce the
    original numbers for comparison."""
    embeds = []
    for start in range(0, tactile.shape[0], batch_size):
        chunk = tactile[start:start + batch_size].to(device)
        embeds.append(encoder(chunk).cpu())
    return torch.cat(embeds, dim=0)


@torch.no_grad()
def encode_causal(
    encoder: CNNetEmbedding,
    tactile: torch.Tensor,          # (N, T, 1, 16, 16) -- one full window per row
    window_idx: np.ndarray,         # (M,) which window each sample's tactile is read from
    t_values: np.ndarray,           # (M,) which t within that window
    causal_frame_idx: np.ndarray,   # (valid_t, W) row t = frame indices for that t, all <= t
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """(M, emb_dim): one embedding per (window, t) SAMPLE -- CAUSAL / FIXED
    path. For each sample, gathers only its W causal frames (indices <= t,
    from causal_frame_idx -- opentouch_train.regression_data's
    _causal_frame_indices, the same arithmetic PoseTransitionDataset.
    __getitem__ uses) from `tactile` immediately before that sample's
    encoder pass. No frame with index > t is ever assembled into the
    encoder's input tensor. `window_idx` is the window each sample reads
    tactile FROM -- pass the real window for the tactile_causal condition,
    or a deranged window (same t) for the shuffled_tactile_causal control.

    Cost: M = sum over windows of valid_t_per_window encoder passes'-worth
    of gather+forward, vs. encode_windows' N (one per window) -- exactly
    valid_t_per_window x more work, done here in batches of `batch_size`
    samples so the (M, W, 1, 16, 16) gathered tensor is never materialized
    in full (see module report for concrete numbers).
    """
    M = len(window_idx)
    window_idx_t = torch.as_tensor(window_idx, dtype=torch.long)
    frame_idx_per_sample = causal_frame_idx[t_values]  # (M, W) numpy gather, one row per sample
    frame_idx_t = torch.as_tensor(frame_idx_per_sample, dtype=torch.long)

    embeds = []
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        w = window_idx_t[start:end]                  # (b,)
        f = frame_idx_t[start:end]                    # (b, W)
        chunk = tactile[w.unsqueeze(1), f]             # (b, W, 1, 16, 16) -- advanced-index gather
        chunk = chunk.to(device)
        embeds.append(encoder(chunk).cpu())
    return torch.cat(embeds, dim=0)


def extract_samples(dataset: PoseTransitionDataset):
    """Vectorized equivalent of iterating dataset[i] for every valid
    (window, t) -- avoids the ~1500x-per-sample cost of the real __getitem__
    for what's needed here (pose_t, articulation_delta, window/t index).
    Ordering (window-major, t-minor) matches
    PoseTransitionDataset.__len__/__getitem__'s divmod(idx, valid_t_per_window).
    """
    pose = dataset._pose  # (N, T, 21, 3)
    n_windows, seq_len = pose.shape[0], pose.shape[1]
    k = dataset.horizon_k
    valid_t = dataset.valid_t_per_window
    assert valid_t == seq_len - k

    pose_t = pose[:, :valid_t]      # (N, valid_t, 21, 3)
    pose_future = pose[:, k:]       # (N, valid_t, 21, 3)
    world_delta = (pose_future - pose_t).reshape(-1, NUM_KEYPOINTS, COORD_DIM)
    _, articulation_delta = decompose_world_delta(world_delta)  # (N*valid_t, 21, 3)

    window_idx = np.repeat(np.arange(n_windows), valid_t)  # (N*valid_t,) window-major
    t_values = np.tile(np.arange(valid_t), n_windows)       # (N*valid_t,) t-minor
    pose_t_flat = pose_t.reshape(-1, NUM_KEYPOINTS * COORD_DIM)  # (N*valid_t, 63)

    return window_idx, t_values, pose_t_flat, articulation_delta


def real_history_length(t_values: np.ndarray, causal_window: int) -> np.ndarray:
    """(M,) -> (M,): number of REAL (non-padded) causal frames for each
    sample's t -- min(t+1, causal_window), same formula as
    opentouch_train.regression_data's min-history filter."""
    return np.minimum(t_values + 1, causal_window)


def min_history_mask(t_values: np.ndarray, causal_window: int, min_history: int) -> np.ndarray:
    return real_history_length(t_values, causal_window) >= min_history


def raise_if_no_samples_survive_min_history(
    split_name: str, n_before: int, n_after: int,
    horizon_k: int, sequence_length: int, causal_window: int, min_history: int,
) -> None:
    if n_after > 0:
        return
    raise ValueError(
        f"--min-history={min_history} leaves ZERO valid '{split_name}' samples (had "
        f"{n_before} before filtering) for --horizon-k={horizon_k} "
        f"--sequence-length={sequence_length} --causal-window={causal_window}: the longest "
        f"real (non-padded) causal history any t can reach is "
        f"min(sequence_length-horizon_k, causal_window)="
        f"{min(sequence_length - horizon_k, causal_window)} < min_history={min_history}. "
        "With the default causal_window (== sequence_length), this requires "
        f"sequence_length >= horizon_k + min_history ({sequence_length} >= {horizon_k} + "
        f"{min_history} = {horizon_k + min_history}). Increase --sequence-length, reduce "
        "--horizon-k, or lower --min-history."
    )


def report_real_history_distribution(split_name: str, t_values: np.ndarray, causal_window: int) -> None:
    """FIX 3: history is a controlled, STATED variable -- print the exact
    (t -> real_history -> sample count) breakdown of whatever sample set
    survived --min-history filtering, per split."""
    real_history = real_history_length(t_values, causal_window)
    unique_t = np.unique(t_values)
    log.info(f"  [{split_name}] real-history distribution over {len(t_values)} retained samples:")
    for t in unique_t:
        t_mask = t_values == t
        n = int(t_mask.sum())
        h = int(real_history[t_mask][0])
        log.info(f"    t={t:<3d} real_history={h:<3d} n={n:<8d} ({n / len(t_values):.1%})")


def fit_and_eval_probe(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """One sklearn LogisticRegression per (condition, joint, axis). Returns
    None for a degenerate target (e.g. the wrist row of articulation_delta
    is IDENTICALLY zero by construction -- decompose_world_delta -- so
    y_train has a single class and there is no direction to probe)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    if len(np.unique(y_train)) < 2:
        return None

    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)

    if len(np.unique(y_val)) < 2:
        # Moving-subset val labels happen to be single-class for this
        # particular joint/axis -- AUC is undefined, but accuracy still is.
        preds = clf.predict(X_val)
        return {"auc": float("nan"), "acc": float(accuracy_score(y_val, preds)), "n": int(len(y_val))}

    probs = clf.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_val, probs)),
        "acc": float(accuracy_score(y_val, preds)),
        "n": int(len(y_val)),
    }


def standardize(X_train: np.ndarray, X_val: np.ndarray):
    """Fit-on-train StandardScaler -- embeddings/pose vectors are not
    unit-scaled, and sklearn's lbfgs solver converges much more reliably on
    standardized features."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_val)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Frozen-feature linear probe for tactile-carries-direction, at fixed k=16."
    )
    p.add_argument("--checkpoint", required=True, help="Retrieval checkpoint with a cnn_gru tactile encoder.")
    p.add_argument("--data", required=True, help="Path to preprocessed HF dataset directory.")
    p.add_argument("--horizon-k", type=int, default=16, help="Frames ahead for the articulation-delta target.")
    p.add_argument("--sequence-length", type=int, default=20)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42, help="Clip-level train/val split seed; also the derangement seed for the shuffled-tactile control.")
    p.add_argument("--tactile-emb-dim", type=int, default=64, help="Must match the checkpoint's embed_dim.")
    p.add_argument(
        "--motion-threshold", type=float, default=None,
        help="'Moving' cutoff on median-fingertip ARTICULATION displacement at k. Default: "
             "computed from the train split as the 25th percentile of that statistic "
             "(opentouch_train.regression_data.compute_motion_threshold), same as "
             "regression_main.py's default -- at k=16 this has previously computed to "
             "~0.023973 on this dataset. Pass explicitly to pin an exact value.",
    )
    p.add_argument(
        "--causal", dest="causal", action="store_true", default=True,
        help="Tactile embedding for sample (window, t) computed only from frames <= t "
             "(fixed-length causal window ending at t, edge-padded at the start -- see "
             "--causal-window). Default: True. Both causal and noncausal conditions are "
             "always computed and reported side by side regardless of this flag -- it "
             "only controls --causal-window's default resolution and is recorded in "
             "--output for provenance.",
    )
    p.add_argument(
        "--noncausal", dest="causal", action="store_false",
        help="See --causal. Reproduces the PRE-FIX numbers (full T-window fed to the "
             "encoder for every t) -- kept as the *_noncausal columns for measuring the "
             "leak's magnitude, not because this script stops computing the causal "
             "condition.",
    )
    p.add_argument(
        "--causal-window", type=int, default=None,
        help="Fixed length W of the causal tactile window ending at t. Default: "
             "--sequence-length, matching the length the tactile encoder was trained on.",
    )
    p.add_argument(
        "--min-history", type=int, default=10,
        help="Exclude sample (window, t) unless it has >= this many REAL (non-padded) "
             "causal frames, i.e. min(t+1, causal_window) >= --min-history. Applied ONCE, "
             "globally, before any of the five conditions are built, so every condition is "
             "scored on the identical retained sample set. At horizon_k=16, "
             "sequence_length=20 (the default), EVERY t has real history <= 4 < 10, so this "
             "raises immediately rather than silently reporting on zero samples -- see "
             "module docstring's 'MIN-HISTORY FILTERING' section. Pass 1 to effectively "
             "disable filtering.",
    )
    p.add_argument("--batch-size", type=int, default=256, help="Tactile-encoder inference batch size.")
    p.add_argument("--auc-threshold", type=float, default=AUC_CLEAR_THRESHOLD)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None, help="Optional path to save the full per-joint/axis/condition JSON report.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    causal_window = args.causal_window if args.causal_window is not None else args.sequence_length

    encoder = load_tactile_encoder(args.checkpoint, args.tactile_emb_dim, device)

    splits = _load_and_split_dataset(args.data, args.val_ratio, args.test_ratio, args.split_seed)
    assert "train" in splits, "Training split is required."
    assert "val" in splits, "Val split is required."

    common_kwargs = dict(
        hf_dataset_path=args.data, sequence_length=args.sequence_length,
        include_tactile=True, include_visual=False, include_pose=True,
    )
    train_base = VideoTactilePoseDataset(split="train", _preloaded=splits["train"], **common_kwargs)
    val_base = VideoTactilePoseDataset(split="val", _preloaded=splits["val"], **common_kwargs)

    # shuffle_tactile=False here -- PoseTransitionDataset just gives us the
    # materialized (pose, tactile) tensors + valid_t_per_window bookkeeping,
    # at causal=False (irrelevant to this script's own encoding path below,
    # which reads dataset._tactile directly and applies causal/noncausal
    # encoding itself rather than going through __getitem__). The
    # shuffled-tactile CONTROL is built by deranging WINDOW INDICES (not
    # embeddings -- causal embeddings are per-sample, not per-window; see
    # below), same _make_derangement as regression_data.py.
    train_ds = PoseTransitionDataset(train_base, args.horizon_k, shuffle_tactile=False, causal=False)
    val_ds = PoseTransitionDataset(val_base, args.horizon_k, shuffle_tactile=False, causal=False)
    assert train_ds._tactile is not None and val_ds._tactile is not None, (
        "include_tactile=True above should guarantee materialized tactile tensors"
    )
    log.info(f"train windows: {len(train_ds._pose)}  val windows: {len(val_ds._pose)}")

    motion_threshold = args.motion_threshold
    if motion_threshold is None:
        motion_threshold = compute_motion_threshold(train_ds, percentile=25.0)
        log.info(f"--motion-threshold not given: computed from train split at k={args.horizon_k}: {motion_threshold:.6f}")
    else:
        log.info(f"--motion-threshold explicitly set: {motion_threshold}")

    train_window_idx, train_t_values, train_pose_t, train_articulation_delta = extract_samples(train_ds)
    val_window_idx, val_t_values, val_pose_t, val_articulation_delta = extract_samples(val_ds)

    # --- FIX 1: min-history filtering, applied ONCE globally (before any of
    # the five conditions are built) so every condition is scored on the
    # IDENTICAL retained (window, t) sample set -- history is a controlled
    # variable for the whole comparison, not an implicit one that differs
    # per condition. See module docstring's "MIN-HISTORY FILTERING" section.
    n_train_before, n_val_before = len(train_window_idx), len(val_window_idx)
    train_keep = min_history_mask(train_t_values, causal_window, args.min_history)
    val_keep = min_history_mask(val_t_values, causal_window, args.min_history)
    raise_if_no_samples_survive_min_history(
        "train", n_train_before, int(train_keep.sum()),
        args.horizon_k, args.sequence_length, causal_window, args.min_history,
    )
    raise_if_no_samples_survive_min_history(
        "val", n_val_before, int(val_keep.sum()),
        args.horizon_k, args.sequence_length, causal_window, args.min_history,
    )
    log.info(
        f"--min-history={args.min_history}: train {n_train_before} -> {int(train_keep.sum())} "
        f"samples ({int(train_keep.sum()) / n_train_before:.1%} retained, "
        f"{n_train_before - int(train_keep.sum())} dropped); "
        f"val {n_val_before} -> {int(val_keep.sum())} samples "
        f"({int(val_keep.sum()) / n_val_before:.1%} retained, "
        f"{n_val_before - int(val_keep.sum())} dropped)"
    )
    train_window_idx, train_t_values = train_window_idx[train_keep], train_t_values[train_keep]
    train_pose_t, train_articulation_delta = train_pose_t[train_keep], train_articulation_delta[train_keep]
    val_window_idx, val_t_values = val_window_idx[val_keep], val_t_values[val_keep]
    val_pose_t, val_articulation_delta = val_pose_t[val_keep], val_articulation_delta[val_keep]

    # FIX 3: real-history distribution over the retained samples, reported
    # explicitly rather than left implicit.
    report_real_history_distribution("train", train_t_values, causal_window)
    report_real_history_distribution("val", val_t_values, causal_window)

    val_moving_mask = (fingertip_displacement(val_articulation_delta) >= motion_threshold).numpy()
    n_moving = int(val_moving_mask.sum())
    log.info(
        f"val samples: {len(val_moving_mask)}  moving (>= {motion_threshold:.6f}): "
        f"{n_moving} ({n_moving / len(val_moving_mask):.1%})"
    )
    if n_moving == 0:
        raise ValueError("No val samples clear --motion-threshold; cannot evaluate the moving subset.")

    train_perm = _make_derangement(len(train_ds._pose), seed=args.split_seed)
    val_perm = _make_derangement(len(val_ds._pose), seed=args.split_seed)

    # --- NONCAUSAL (pre-fix): one embedding per WINDOW, broadcast to every
    # t via window_idx -- reproduces the original (leaky) numbers.
    log.info("Encoding tactile: noncausal (one embedding per window, broadcast to every t)...")
    train_embed_noncausal = encode_windows(encoder, train_ds._tactile, args.batch_size, device)  # (N_train, D)
    val_embed_noncausal = encode_windows(encoder, val_ds._tactile, args.batch_size, device)      # (N_val, D)
    train_embed_noncausal_shuffled = train_embed_noncausal[train_perm]
    val_embed_noncausal_shuffled = val_embed_noncausal[val_perm]

    # --- CAUSAL (fixed): one embedding per (window, t) SAMPLE, frames <= t
    # only. shuffled_tactile_causal reads the SAME t from the DERANGED
    # window (train_perm[train_window_idx]), not a separately-deranged
    # embedding array (embeddings aren't per-window here, so there is
    # nothing to derange after the fact -- the derangement has to happen at
    # the window-index level, before encoding).
    log.info(f"Encoding tactile: causal (one embedding per (window,t) sample, window={causal_window})...")
    train_causal_frame_idx = _causal_frame_indices(train_ds.valid_t_per_window, causal_window)
    val_causal_frame_idx = _causal_frame_indices(val_ds.valid_t_per_window, causal_window)

    train_embed_causal = encode_causal(
        encoder, train_ds._tactile, train_window_idx, train_t_values,
        train_causal_frame_idx, args.batch_size, device,
    )  # (M_train, D)
    val_embed_causal = encode_causal(
        encoder, val_ds._tactile, val_window_idx, val_t_values,
        val_causal_frame_idx, args.batch_size, device,
    )  # (M_val, D)
    train_embed_causal_shuffled = encode_causal(
        encoder, train_ds._tactile, train_perm[train_window_idx], train_t_values,
        train_causal_frame_idx, args.batch_size, device,
    )
    val_embed_causal_shuffled = encode_causal(
        encoder, val_ds._tactile, val_perm[val_window_idx], val_t_values,
        val_causal_frame_idx, args.batch_size, device,
    )

    conditions_raw = {
        "tactile_causal": (train_embed_causal.numpy(), val_embed_causal.numpy()),
        "tactile_noncausal": (
            train_embed_noncausal[train_window_idx].numpy(), val_embed_noncausal[val_window_idx].numpy(),
        ),
        "shuffled_tactile_causal": (
            train_embed_causal_shuffled.numpy(), val_embed_causal_shuffled.numpy(),
        ),
        "shuffled_tactile_noncausal": (
            train_embed_noncausal_shuffled[train_window_idx].numpy(),
            val_embed_noncausal_shuffled[val_window_idx].numpy(),
        ),
        "pose": (train_pose_t.numpy(), val_pose_t.numpy()),
    }
    # Standardize once per condition (not per joint/axis) -- fit depends
    # only on X, and is reused across all 60 joint/axis probes.
    conditions = {
        name: standardize(X_train, X_val) for name, (X_train, X_val) in conditions_raw.items()
    }
    # Moving-subset val features precomputed ONCE per condition -- the mask
    # is the same for every joint/axis, so this avoids re-slicing X_val 180
    # times (21 joints x 3 axes x 3 conditions) inside the loop below.
    conditions_val_moving = {name: X_val[val_moving_mask] for name, (_, X_val) in conditions.items()}
    for name, (X_train, _) in conditions.items():
        log.info(f"condition '{name}': X_train shape {X_train.shape}")

    results: dict[str, dict] = {name: {} for name in conditions}
    for j in range(NUM_KEYPOINTS):
        if j == WRIST_INDEX:
            # articulation_delta's wrist row is IDENTICALLY zero by
            # construction (decompose_world_delta) -- no direction exists
            # to probe, so this joint is skipped rather than fit against a
            # single-class target.
            continue
        for axis in range(COORD_DIM):
            y_train = (train_articulation_delta[:, j, axis] > 0).numpy().astype(int)
            y_val_all = (val_articulation_delta[:, j, axis] > 0).numpy().astype(int)
            y_val_moving = y_val_all[val_moving_mask]

            for name, (X_train, _) in conditions.items():
                res = fit_and_eval_probe(X_train, y_train, conditions_val_moving[name], y_val_moving)
                results[name][(j, axis)] = res

    _report(results, args.auc_threshold)

    if args.output:
        serializable = {
            name: {f"{JOINT_LABELS[j]}_{AXIS_LABELS[a]}": r for (j, a), r in per_cond.items()}
            for name, per_cond in results.items()
        }
        with open(args.output, "w") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "horizon_k": args.horizon_k,
                    "causal_default": args.causal,
                    "causal_window": causal_window,
                    "min_history": args.min_history,
                    "n_train_before_min_history": n_train_before,
                    "n_train_after_min_history": int(train_keep.sum()),
                    "n_val_before_min_history": n_val_before,
                    "n_val_after_min_history": int(val_keep.sum()),
                    "motion_threshold": motion_threshold,
                    "n_val_moving": n_moving,
                    "auc_threshold": args.auc_threshold,
                    "results": serializable,
                },
                f, indent=2,
            )
        log.info(f"Saved full report to {args.output}")

    return results


def _report(results: dict, auc_threshold: float) -> None:
    conditions = list(results.keys())
    width = max(16, max(len(c) for c in conditions) + 2)
    total_width = 14 + width * len(conditions)

    print(f"\n{'='*total_width}")
    print("  Per-joint mean AUC (val, moving subset), averaged over x/y/z axes")
    print("  *_causal: frames <= t only (fixed).  *_noncausal: full window, pre-fix leak.")
    print(f"{'='*total_width}")
    header = f"  {'joint':<12}" + "".join(f"{c:>{width}}" for c in conditions)
    print(header)
    for j in range(NUM_KEYPOINTS):
        if j == WRIST_INDEX:
            continue
        row = f"  {JOINT_LABELS[j]:<12}"
        for name in conditions:
            aucs = [
                results[name][(j, a)]["auc"] for a in range(COORD_DIM)
                if results[name][(j, a)] is not None and not np.isnan(results[name][(j, a)]["auc"])
            ]
            cell = f"{np.mean(aucs):.4f}" if aucs else "n/a"
            row += f"{cell:>{width}}"
        print(row)

    print(f"\n{'='*total_width}")
    print(f"  Joint/axis combinations (of {(NUM_KEYPOINTS - 1) * COORD_DIM}) clearing AUC >= {auc_threshold}")
    print(f"{'='*total_width}")
    for name in conditions:
        cleared = sum(
            1 for r in results[name].values()
            if r is not None and not np.isnan(r["auc"]) and r["auc"] >= auc_threshold
        )
        total = len(results[name])
        print(f"  {name:<26}: {cleared} / {total}")
    print(f"{'='*total_width}\n")


if __name__ == "__main__":
    main()
