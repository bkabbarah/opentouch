"""Does tactile's predictive contribution hide in a SUBSET of the data?

MOTIVATION. scripts/tactile_direction_probe.py established, on the full
population, that tactile's marginal contribution over a matched-temporal pose
baseline is ~0.000-0.004 AUC -- redundancy, not unique signal. That is an
AVERAGE over every sample. It is entirely consistent with tactile mattering a
great deal in a small regime and not at all elsewhere; an average near zero
cannot distinguish "no signal anywhere" from "signal in 8% of samples".

This script tests the regime hypothesis directly, and is built so that a
positive result would survive review rather than being the first of many
slices that happened to look good.

WHAT MAKES THIS NOT A FISHING EXPEDITION. Four things, all enforced in code:

  1. PRE-REGISTERED GRID. The subsets are declared in SUBSET_FAMILIES below,
     before seeing any result, and every declared cell is reported whether it
     wins or loses -- `--subsets` can only narrow to a named family, never
     invent a new cut. The grid size is written into the output so the
     multiplicity correction cannot be quietly recomputed later on a smaller
     grid.
  2. CAUSAL SUBSET DEFINITIONS. Every subset variable is computed from frames
     <= t only (assert_causal_subset_definition enforces this by
     construction, the same poisoning argument the causal encoder uses). A
     subset defined using the future -- "samples where contact begins during
     [t, t+k]" -- would select on the answer and manufacture a positive.
  3. MATCHED CONTROL INSIDE EVERY SUBSET. The shuffled-tactile twin is
     refit within each subset. A subset with easier targets raises pose AND
     tactile AND shuffled together; only tactile-minus-shuffled is read as
     signal. Sample size is also matched by construction, since all three
     conditions share the subset's rows.
  4. SELECT ON VAL, CONFIRM ON TEST. --stage discover fits on train and
     scores on val. --stage confirm re-runs ONLY the cells named in a
     discover run's output, on the held-out test split, which
     tactile_direction_probe.py never touched (it asserts train/val only).
     A cell that survives confirm is a genuine out-of-sample finding; a cell
     that wins in discover and dies in confirm was multiplicity.

WHAT ELSE THIS FIXES. Two problems the audit of the original probe surfaced,
both of which affect the headline numbers regardless of the subset question:

  A. UNCERTAINTY. The original probe reports point AUCs with no interval, and
     its samples are far from independent -- adjacent t within a window share
     nearly all their causal frames, and windows within a clip are
     correlated, so the nominal n overstates information by roughly the
     window-to-sample ratio. Every number here carries a CLIP-CLUSTERED
     paired bootstrap CI (clip_clustered_bootstrap), resampling whole clips,
     which is the correct unit of independence.
  B. REFERENCE FRAME. The existing target ("articulation delta") removes only
     wrist TRANSLATION, so whole-hand ROTATION is still in it, expressed in
     arbitrary world axes -- see opentouch.articulation_frames. The original
     probe's entire signal sat on world y, which is what a global
     re-orientation effect looks like and is not what finger articulation
     should look like. --target-variant runs the same probe against the
     rigid-rotation-removed, palm-framed target. If the signal survives, it
     is articulation; if it vanishes, the "tactile predicts finger direction"
     claim was whole-hand motion and must be reworded.

Nothing here trains a network. Encoders are frozen and loaded from the same
retrieval checkpoint the original probe uses, via that script's own loaders,
so the two cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from opentouch.articulation_frames import (
    AXIS_NAMES_HAND,
    AXIS_NAMES_WORLD,
    all_target_variants,
    hand_frame_degenerate,
    rigid_fraction,
)
from opentouch.pose_regression import COORD_DIM, NUM_KEYPOINTS, WRIST_INDEX
from opentouch.regression_metrics import fingertip_displacement
from opentouch_train.data import VideoTactilePoseDataset, _load_and_split_dataset
from opentouch_train.regression_data import (
    PoseTransitionDataset,
    _causal_frame_indices,
    _make_derangement,
    compute_motion_threshold,
)

# Reuse the original probe's loaders/encoders verbatim -- same checkpoint
# parsing, same strict=True, same causal gather -- so this script cannot
# silently diverge from the result it is extending.
from tactile_direction_probe import (  # noqa: E402
    JOINT_LABELS,
    encode_causal,
    encode_pose_causal,
    load_pose_encoder,
    load_tactile_encoder,
    min_history_mask,
    raise_if_no_samples_survive_min_history,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ==========================================================================
# Subset definitions -- PRE-REGISTERED. Each is a function of causal signal
# only: (tactile_window, pose_window, t) -> scalar per sample, then binned by
# TRAIN-fitted thresholds.
# ==========================================================================


@dataclass(frozen=True)
class SubsetFamily:
    """One pre-registered way of cutting the data.

    `statistic` maps (tactile (M,W,1,16,16) causal slice, pose (M,W,21,3)
    causal slice) -> (M,) float. Both inputs are the CAUSAL slices already
    gathered for each sample, so a statistic physically cannot see the
    future: there is no frame > t in either argument. `bin_edges_percentiles`
    are fitted on the TRAIN split and applied unchanged to val/test.
    """

    name: str
    statistic: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    bin_edges_percentiles: tuple[float, ...]
    bin_labels: tuple[str, ...]
    hypothesis: str

    def __post_init__(self) -> None:
        if len(self.bin_labels) != len(self.bin_edges_percentiles) + 1:
            raise ValueError(
                f"family '{self.name}': {len(self.bin_edges_percentiles)} cut points need "
                f"{len(self.bin_edges_percentiles) + 1} labels, got {len(self.bin_labels)}"
            )


def _contact_level(tactile: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Mean pressure at the CURRENT frame (last row of the causal slice).

    Tests the most basic version of the hypothesis: a pressure sensor can
    only report contact forces, so if tactile informs motion anywhere it
    should be where the hand is actually loaded. If the marginal contribution
    is flat across no-contact / light / firm, tactile's silence is not
    explained by "most samples aren't touching anything".
    """
    return tactile[:, -1].flatten(1).mean(dim=1)


def _contact_change(tactile: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Signed change in mean pressure across the causal slice: (recent
    quarter mean) - (earlier quarter mean), both strictly <= t.

    This is the sharpest hypothesis in the grid. Pose kinematics predict
    future motion by continuation -- where the hand is heading, it keeps
    heading. That fails exactly when an external constraint changes: the
    moment contact is made the hand is about to be stopped or redirected, and
    the moment contact breaks it is about to move freely. Pose at t cannot
    know a constraint is about to bind; a pressure signal that is rising
    can. If tactile has any unique predictive content at all, the contact
    transitions are where mechanism says it must be.
    """
    quarter = max(1, tactile.shape[1] // 4)
    recent = tactile[:, -quarter:].flatten(1).mean(dim=1)
    earlier = tactile[:, :quarter].flatten(1).mean(dim=1)
    return recent - earlier


def _pose_speed(tactile: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Mean per-joint displacement between the last two causal frames.

    The complementary-information hypothesis: the matched pose baseline wins
    on the full population largely because velocity extrapolates well. Where
    the hand is nearly static, velocity carries almost nothing and pose's
    advantage should collapse -- if tactile is ever going to add value over
    pose, a low-velocity regime is its best chance.
    """
    if pose.shape[1] < 2:
        return torch.zeros(pose.shape[0], dtype=pose.dtype)
    step = pose[:, -1] - pose[:, -2]
    return step.norm(dim=-1).mean(dim=1)


def _grip_aperture(tactile: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Thumb-tip to index-tip distance at the current frame, wrist-normalized.

    A coarse, purely kinematic proxy for grasp type that needs no annotation
    file: small aperture is a pinch/precision configuration, large is an open
    or power configuration. Included because the PI's regime hypothesis is
    most often stated in terms of grip type, and this tests it on the data
    everyone already has -- `--metadata-csv` covers the annotated version.
    """
    current = pose[:, -1]
    thumb_tip, index_tip = current[:, 4], current[:, 8]
    scale = (current[:, 9] - current[:, WRIST_INDEX]).norm(dim=-1).clamp(min=1e-6)
    return (thumb_tip - index_tip).norm(dim=-1) / scale


SUBSET_FAMILIES: tuple[SubsetFamily, ...] = (
    SubsetFamily(
        name="contact_level",
        statistic=_contact_level,
        bin_edges_percentiles=(33.0, 67.0),
        bin_labels=("no_contact", "light_contact", "firm_contact"),
        hypothesis="tactile informs motion only where the hand is actually loaded",
    ),
    SubsetFamily(
        name="contact_change",
        statistic=_contact_change,
        bin_edges_percentiles=(20.0, 80.0),
        bin_labels=("releasing", "steady", "onsetting"),
        hypothesis="tactile informs motion at contact transitions, where kinematic continuation breaks",
    ),
    SubsetFamily(
        name="pose_speed",
        statistic=_pose_speed,
        bin_edges_percentiles=(33.0, 67.0),
        bin_labels=("slow", "medium", "fast"),
        hypothesis="tactile helps where pose velocity is uninformative",
    ),
    SubsetFamily(
        name="grip_aperture",
        statistic=_grip_aperture,
        bin_edges_percentiles=(33.0, 67.0),
        bin_labels=("pinch", "medium_aperture", "open"),
        hypothesis="tactile's value depends on grasp configuration",
    ),
)

FAMILIES_BY_NAME = {f.name: f for f in SUBSET_FAMILIES}


# ==========================================================================
# Causality guard for subset statistics
# ==========================================================================


def assert_causal_subset_definition(
    family: SubsetFamily, tactile_slice: torch.Tensor, pose_slice: torch.Tensor
) -> None:
    """A subset statistic must be a function of the causal slice ALONE.

    The slices handed to `statistic` are already gathered from frames <= t,
    so the future is structurally absent -- but a statistic could still be
    accidentally non-deterministic or stateful, which would break the
    train-fitted-threshold contract. This re-evaluates on a clone and on a
    row permutation and requires exact per-row agreement, which catches any
    dependence on batch composition or ordering.
    """
    baseline = family.statistic(tactile_slice, pose_slice)
    if baseline.shape != (tactile_slice.shape[0],):
        raise ValueError(
            f"family '{family.name}' must return one scalar per sample, got {tuple(baseline.shape)}"
        )
    repeat = family.statistic(tactile_slice.clone(), pose_slice.clone())
    if not torch.allclose(baseline, repeat, atol=0, rtol=0):
        raise ValueError(f"family '{family.name}' is not deterministic")

    order = torch.randperm(tactile_slice.shape[0])
    permuted = family.statistic(tactile_slice[order], pose_slice[order])
    if not torch.allclose(permuted, baseline[order], atol=1e-6):
        raise ValueError(
            f"family '{family.name}' depends on batch composition -- it must be computed "
            "per-sample, or train-fitted bin thresholds do not transfer"
        )


def gather_causal_slices(
    source: torch.Tensor,
    window_idx: np.ndarray,
    t_values: np.ndarray,
    causal_frame_idx: np.ndarray,
) -> torch.Tensor:
    """Same gather `encode_causal` performs, without the encoder -- so subset
    statistics see EXACTLY the frames the tactile encoder saw, no more."""
    frame_idx = torch.as_tensor(causal_frame_idx[t_values], dtype=torch.long)
    windows = torch.as_tensor(window_idx, dtype=torch.long).unsqueeze(1)
    return source[windows, frame_idx]


def fit_bin_edges(values: np.ndarray, percentiles: Sequence[float]) -> np.ndarray:
    """Bin edges from the TRAIN distribution only."""
    return np.percentile(values, list(percentiles))


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False)


# ==========================================================================
# Statistics: clip-clustered paired bootstrap
# ==========================================================================


def clip_clustered_bootstrap(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    clip_ids: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Paired AUC difference (a - b) with a CI from resampling whole CLIPS.

    Resampling clips rather than samples is the whole point. Adjacent t
    within a window share nearly all their causal frames and most of their
    target, and windows within a clip share a participant, a scene, a glove
    calibration and an activity, so per-sample resampling would report an
    interval several times too narrow and turn ordinary correlation into
    apparent significance. The unit of independence here is the clip.

    Paired: both conditions are scored on the SAME resampled clips every
    draw, so the shared difficulty of a given clip cancels in the difference
    and the interval reflects only the contrast being tested.
    """
    from sklearn.metrics import roc_auc_score

    unique_clips = np.unique(clip_ids)
    if len(unique_clips) < 2:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_clips": int(len(unique_clips))}

    clip_to_rows = {clip: np.flatnonzero(clip_ids == clip) for clip in unique_clips}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        drawn = rng.choice(unique_clips, size=len(unique_clips), replace=True)
        rows = np.concatenate([clip_to_rows[c] for c in drawn])
        y_boot = y_true[rows]
        if len(np.unique(y_boot)) < 2:
            continue
        deltas.append(roc_auc_score(y_boot, score_a[rows]) - roc_auc_score(y_boot, score_b[rows]))

    if not deltas:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_clips": int(len(unique_clips))}

    deltas_arr = np.asarray(deltas)
    observed = roc_auc_score(y_true, score_a) - roc_auc_score(y_true, score_b)
    return {
        "delta": float(observed),
        "ci_low": float(np.percentile(deltas_arr, 2.5)),
        "ci_high": float(np.percentile(deltas_arr, 97.5)),
        "n_clips": int(len(unique_clips)),
        "n_boot_effective": int(len(deltas_arr)),
    }


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """(M,) p-values -> (M,) bool, BH step-up at level alpha.

    The grid here is (families x bins x axes x horizons), which is large
    enough that an uncorrected 'this cell looks significant' is close to
    meaningless. BH controls the false discovery rate, which is the right
    error notion for a screen whose output is 'these cells deserve a
    confirmation run', not a single decisive test.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    rejected = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.flatnonzero(passed).max()
        rejected[order[: cutoff + 1]] = True
    return rejected


def bootstrap_pvalue(ci: dict) -> float:
    """Two-sided p for 'delta == 0' read off the bootstrap distribution's
    normal approximation. Approximate by design -- BH here selects cells for
    a confirmation run, and the confirmation run is what decides."""
    if not np.isfinite(ci.get("delta", float("nan"))):
        return 1.0
    spread = (ci["ci_high"] - ci["ci_low"]) / (2 * 1.96)
    if spread <= 0:
        return 0.0 if ci["delta"] != 0 else 1.0
    from math import erfc, sqrt

    return float(erfc(abs(ci["delta"] / spread) / sqrt(2)))


# ==========================================================================
# Probe fitting
# ==========================================================================


def fit_probe_scores(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray
) -> "np.ndarray | None":
    """Fit on train rows, return decision scores on eval rows.

    Returns SCORES rather than an AUC so the caller can bootstrap the paired
    difference on identical resampled rows -- computing AUCs here and
    differencing them afterward would lose the pairing that makes the
    interval tight and correct.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y_train)) < 2 or len(y_train) < 50:
        return None
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(scaler.transform(X_train), y_train)
    return clf.predict_proba(scaler.transform(X_eval))[:, 1]


@dataclass
class Features:
    """The three conditions compared inside every subset cell. `tactile` and
    `shuffled` are both pose+64 dims, so any difference between them is
    tactile CONTENT, not feature count."""

    pose: np.ndarray
    tactile: np.ndarray
    shuffled: np.ndarray


@dataclass
class SplitData:
    window_idx: np.ndarray
    t_values: np.ndarray
    clip_ids: np.ndarray
    targets: dict[str, torch.Tensor]
    moving_mask: np.ndarray
    features: Features
    subset_values: dict[str, np.ndarray] = field(default_factory=dict)


def build_features(pose_emb: torch.Tensor, tactile: torch.Tensor, shuffled: torch.Tensor) -> Features:
    pose_np = pose_emb.numpy()
    return Features(
        pose=pose_np,
        tactile=np.hstack([pose_np, tactile.numpy()]),
        shuffled=np.hstack([pose_np, shuffled.numpy()]),
    )


def cluster_ids_for_windows(
    dataset: PoseTransitionDataset, window_idx: np.ndarray, unit: str
) -> np.ndarray:
    """Per-sample resampling-cluster identity, at clip or scene granularity.

    `PoseTransitionDataset._materialize_pose_and_tactile` already records
    per-window `_scenes` and `_clip_ids`, so this reads them directly rather
    than re-deriving window->clip bookkeeping that could drift.

    Choice of unit matters. CLIP is the unit the train/val split is disjoint
    on, so it is the right default for "would this hold on new clips". SCENE
    is coarser and encodes location + participant: the split is NOT
    scene-disjoint, so scene-level clustering answers the stricter and more
    honest question, "would this hold on a new participant". Any effect that
    is significant under clip clustering but not under scene clustering is
    riding on within-participant correlation and should be reported as such.
    """
    if unit not in ("clip", "scene"):
        raise ValueError(f"cluster unit must be 'clip' or 'scene', got {unit!r}")

    scenes = getattr(dataset, "_scenes", None)
    clips = getattr(dataset, "_clip_ids", None)
    if scenes is None or clips is None:
        base = getattr(dataset, "base_dataset", None)
        keys = getattr(base, "clip_keys", None)
        windows = getattr(base, "windows", None)
        if keys is None or windows is None:
            raise RuntimeError(
                "Cannot recover clip/scene identity from the dataset. Clustered bootstrap "
                "intervals would be wrong (too narrow), so this refuses to guess -- expose "
                "_scenes/_clip_ids or base_dataset.windows/clip_keys."
            )
        scenes = np.asarray([keys[windows[w][0]][0] for w in range(len(windows))])
        clips = np.asarray([keys[windows[w][0]][1] for w in range(len(windows))])

    scenes = np.asarray(scenes)
    clips = np.asarray(clips)
    if unit == "scene":
        return scenes[window_idx]
    return np.char.add(np.char.add(scenes[window_idx].astype(str), "::"), clips[window_idx].astype(str))


# ==========================================================================
# Main
# ==========================================================================


def prepare_split(
    split_name: str,
    base_split,
    args,
    causal_window: int,
    tactile_encoder,
    pose_encoder,
    device: torch.device,
    motion_threshold: "float | None",
) -> tuple[SplitData, float]:
    base = VideoTactilePoseDataset(
        split=split_name,
        _preloaded=base_split,
        hf_dataset_path=args.data,
        sequence_length=args.sequence_length,
        include_tactile=True,
        include_visual=False,
        include_pose=True,
    )
    dataset = PoseTransitionDataset(base, args.horizon_k, shuffle_tactile=False, causal=False)
    assert dataset._tactile is not None

    if motion_threshold is None:
        motion_threshold = compute_motion_threshold(dataset, percentile=25.0)
        log.info(f"motion threshold fitted on {split_name}: {motion_threshold:.6f}")

    pose_all = dataset._pose
    n_windows = pose_all.shape[0]
    valid_t = dataset.valid_t_per_window
    window_idx = np.repeat(np.arange(n_windows), valid_t)
    t_values = np.tile(np.arange(valid_t), n_windows)

    keep = min_history_mask(t_values, causal_window, args.min_history)
    raise_if_no_samples_survive_min_history(
        split_name, len(t_values), int(keep.sum()),
        args.horizon_k, args.sequence_length, causal_window, args.min_history,
    )
    window_idx, t_values = window_idx[keep], t_values[keep]
    log.info(f"[{split_name}] {int(keep.sum())} samples retained of {len(keep)} after min-history")

    pose_t = pose_all[:, :valid_t].reshape(-1, NUM_KEYPOINTS, COORD_DIM)[keep]
    pose_future = pose_all[:, args.horizon_k:].reshape(-1, NUM_KEYPOINTS, COORD_DIM)[keep]
    targets = all_target_variants(pose_t, pose_future)

    # Drop samples whose palm frame is undefined -- a hand-frame target is
    # meaningless there, and keeping them would silently mix frames.
    degenerate = hand_frame_degenerate(pose_t).numpy()
    if degenerate.any():
        log.info(f"[{split_name}] dropping {int(degenerate.sum())} samples with a degenerate palm frame")
        good = ~degenerate
        window_idx, t_values = window_idx[good], t_values[good]
        pose_t, pose_future = pose_t[good], pose_future[good]
        targets = {name: value[good] for name, value in targets.items()}

    rigid_share = rigid_fraction(pose_t, pose_future)
    log.info(
        f"[{split_name}] share of 'articulation' energy explained by whole-hand ROTATION: "
        f"median {rigid_share.median():.3f}, mean {rigid_share.mean():.3f}. "
        "(High values mean the existing world-axis target was largely NOT articulation.)"
    )

    moving_mask = (
        fingertip_displacement(targets["wrist_translation_removed"]) >= motion_threshold
    ).numpy()

    causal_frame_idx = _causal_frame_indices(valid_t, causal_window)
    perm = _make_derangement(n_windows, seed=args.split_seed)

    tactile_causal = encode_causal(
        tactile_encoder, dataset._tactile, window_idx, t_values, causal_frame_idx, args.batch_size, device
    )
    tactile_shuffled = encode_causal(
        tactile_encoder, dataset._tactile, perm[window_idx], t_values, causal_frame_idx, args.batch_size, device
    )
    pose_emb = encode_pose_causal(
        pose_encoder, dataset._pose, window_idx, t_values, causal_frame_idx, args.batch_size, device
    )

    tactile_slice = gather_causal_slices(dataset._tactile, window_idx, t_values, causal_frame_idx)
    pose_slice = gather_causal_slices(dataset._pose, window_idx, t_values, causal_frame_idx)

    data = SplitData(
        window_idx=window_idx,
        t_values=t_values,
        clip_ids=cluster_ids_for_windows(dataset, window_idx, args.cluster_unit),
        targets=targets,
        moving_mask=moving_mask,
        features=build_features(pose_emb, tactile_causal, tactile_shuffled),
    )
    for family in SUBSET_FAMILIES:
        assert_causal_subset_definition(family, tactile_slice[:64], pose_slice[:64])
        data.subset_values[family.name] = family.statistic(tactile_slice, pose_slice).numpy()
    return data, motion_threshold


def run_cells(
    train: SplitData,
    evaluation: SplitData,
    families: Sequence[SubsetFamily],
    target_variant: str,
    axis_names: Sequence[str],
    args,
) -> list[dict]:
    results: list[dict] = []
    joints = [j for j in range(NUM_KEYPOINTS) if j != WRIST_INDEX]

    for family in families:
        edges = fit_bin_edges(train.subset_values[family.name], family.bin_edges_percentiles)
        train_bins = assign_bins(train.subset_values[family.name], edges)
        eval_bins = assign_bins(evaluation.subset_values[family.name], edges)
        log.info(f"\n=== family '{family.name}' edges (train-fitted): {np.round(edges, 6).tolist()}")

        for bin_index, bin_label in enumerate(family.bin_labels):
            train_rows = np.flatnonzero(train_bins == bin_index)
            eval_rows = np.flatnonzero((eval_bins == bin_index) & evaluation.moving_mask)
            log.info(f"  [{family.name}/{bin_label}] train n={len(train_rows)} eval n={len(eval_rows)}")
            if len(train_rows) < args.min_cell_size or len(eval_rows) < args.min_cell_size:
                log.info("    skipped: cell below --min-cell-size")
                continue

            for axis, axis_name in enumerate(axis_names):
                axis_deltas = []
                for joint in joints:
                    y_train = (train.targets[target_variant][train_rows, joint, axis] > 0).numpy().astype(int)
                    y_eval = (evaluation.targets[target_variant][eval_rows, joint, axis] > 0).numpy().astype(int)
                    if len(np.unique(y_eval)) < 2:
                        continue

                    scores = {}
                    for condition, matrix in (
                        ("pose", (train.features.pose, evaluation.features.pose)),
                        ("tactile", (train.features.tactile, evaluation.features.tactile)),
                        ("shuffled", (train.features.shuffled, evaluation.features.shuffled)),
                    ):
                        scores[condition] = fit_probe_scores(
                            matrix[0][train_rows], y_train, matrix[1][eval_rows]
                        )
                    if any(value is None for value in scores.values()):
                        continue

                    clips = evaluation.clip_ids[eval_rows]
                    axis_deltas.append(
                        {
                            "joint": JOINT_LABELS[joint],
                            "vs_pose": clip_clustered_bootstrap(
                                y_eval, scores["tactile"], scores["pose"], clips, args.n_boot, args.split_seed
                            ),
                            "vs_shuffled": clip_clustered_bootstrap(
                                y_eval, scores["tactile"], scores["shuffled"], clips, args.n_boot, args.split_seed
                            ),
                        }
                    )

                if not axis_deltas:
                    continue
                results.append(
                    {
                        "family": family.name,
                        "bin": bin_label,
                        "axis": axis_name,
                        "target_variant": target_variant,
                        "horizon_k": args.horizon_k,
                        "n_train": int(len(train_rows)),
                        "n_eval": int(len(eval_rows)),
                        "mean_delta_vs_pose": float(np.mean([d["vs_pose"]["delta"] for d in axis_deltas])),
                        "mean_delta_vs_shuffled": float(
                            np.mean([d["vs_shuffled"]["delta"] for d in axis_deltas])
                        ),
                        "per_joint": axis_deltas,
                    }
                )
    return results


def summarize(results: list[dict], alpha: float) -> list[dict]:
    """Aggregate each cell to one clip-clustered interval and apply BH across
    the WHOLE pre-registered grid."""
    for cell in results:
        deltas = [d["vs_shuffled"] for d in cell["per_joint"]]
        cell["cell_delta_vs_shuffled"] = float(np.mean([d["delta"] for d in deltas]))
        cell["cell_ci_low"] = float(np.mean([d["ci_low"] for d in deltas]))
        cell["cell_ci_high"] = float(np.mean([d["ci_high"] for d in deltas]))
        cell["p_value"] = bootstrap_pvalue(
            {
                "delta": cell["cell_delta_vs_shuffled"],
                "ci_low": cell["cell_ci_low"],
                "ci_high": cell["cell_ci_high"],
            }
        )
    flags = benjamini_hochberg(np.array([c["p_value"] for c in results]), alpha=alpha)
    for cell, survives in zip(results, flags):
        cell["survives_bh"] = bool(survives)
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--horizon-k", type=int, default=8)
    p.add_argument("--sequence-length", type=int, default=36)
    p.add_argument("--causal-window", type=int, default=20)
    p.add_argument("--min-history", type=int, default=10)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--tactile-emb-dim", type=int, default=64)
    p.add_argument("--pose-emb-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-boot", type=int, default=2000, help="Clustered bootstrap draws.")
    p.add_argument(
        "--cluster-unit", choices=["clip", "scene"], default="clip",
        help="Resampling unit for the bootstrap. 'clip' matches the train/val split. "
             "'scene' is stricter (location + participant) and is the honest test of "
             "generalization, since the split is NOT scene-disjoint.",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--min-cell-size", type=int, default=500,
        help="Cells smaller than this are skipped and REPORTED as skipped, never silently dropped.",
    )
    p.add_argument(
        "--target-variant", default="rigid_removed_handframe",
        choices=[
            "world", "wrist_translation_removed", "wrist_translation_removed_handframe",
            "rigid_removed", "rigid_removed_handframe",
        ],
        help="Which target definition to probe. 'wrist_translation_removed' reproduces the "
             "existing published target; the default here removes whole-hand rotation and uses "
             "palm axes, which is the physically meaningful articulation direction.",
    )
    p.add_argument("--subsets", nargs="*", default=None, help="Restrict to named families (never adds new ones).")
    p.add_argument(
        "--stage", choices=["discover", "confirm"], default="discover",
        help="discover: fit on train, score on val. confirm: score the surviving cells on the "
             "untouched TEST split.",
    )
    p.add_argument("--confirm-from", default=None, help="discover-stage JSON whose surviving cells to confirm.")
    p.add_argument("--output", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    causal_window = args.causal_window or args.sequence_length

    families = SUBSET_FAMILIES
    if args.subsets:
        unknown = set(args.subsets) - set(FAMILIES_BY_NAME)
        if unknown:
            raise ValueError(f"unknown subset families {sorted(unknown)}; pre-registered: {sorted(FAMILIES_BY_NAME)}")
        families = tuple(FAMILIES_BY_NAME[name] for name in args.subsets)
    if args.stage == "confirm":
        if not args.confirm_from:
            raise ValueError("--stage confirm requires --confirm-from pointing at a discover-stage JSON")
        discovered = json.loads(Path(args.confirm_from).read_text())
        surviving = {(c["family"], c["bin"], c["axis"]) for c in discovered["cells"] if c["survives_bh"]}
        if not surviving:
            log.info(
                "The discover stage produced NO cell surviving multiplicity correction. There is "
                "nothing to confirm: that is itself the result -- the regime hypothesis found no "
                "candidate regime, which is a far stronger null than a single population average."
            )
            return []
        log.info(f"confirming {len(surviving)} cell(s) on the held-out TEST split: {sorted(surviving)}")
        families = tuple(FAMILIES_BY_NAME[name] for name in {f for f, _, _ in surviving})

    tactile_encoder = load_tactile_encoder(args.checkpoint, args.tactile_emb_dim, device)
    pose_encoder = load_pose_encoder(args.checkpoint, args.pose_emb_dim, device)

    splits = _load_and_split_dataset(args.data, args.val_ratio, args.test_ratio, args.split_seed)
    eval_split_name = "val" if args.stage == "discover" else "test"
    if eval_split_name not in splits:
        raise ValueError(f"split '{eval_split_name}' not present in the dataset")

    train_data, threshold = prepare_split(
        "train", splits["train"], args, causal_window, tactile_encoder, pose_encoder, device, None
    )
    eval_data, _ = prepare_split(
        eval_split_name, splits[eval_split_name], args, causal_window,
        tactile_encoder, pose_encoder, device, threshold,
    )

    axis_names = AXIS_NAMES_HAND if args.target_variant.endswith("handframe") else AXIS_NAMES_WORLD
    cells = run_cells(train_data, eval_data, families, args.target_variant, axis_names, args)
    cells = summarize(cells, args.alpha)

    log.info(f"\n{'=' * 100}")
    log.info(
        f"{args.stage.upper()} | target={args.target_variant} | k={args.horizon_k} | "
        f"eval split={eval_split_name} | grid={len(cells)} cells"
    )
    log.info(f"{'family/bin':<40} {'axis':<8} {'n_eval':>8} {'d vs shuf':>11} {'95% CI':>22} {'BH':>5}")
    for cell in sorted(cells, key=lambda c: -c["cell_delta_vs_shuffled"]):
        log.info(
            f"{cell['family'] + '/' + cell['bin']:<40} {cell['axis']:<8} {cell['n_eval']:>8} "
            f"{cell['cell_delta_vs_shuffled']:>+11.4f} "
            f"[{cell['cell_ci_low']:>+.4f}, {cell['cell_ci_high']:>+.4f}]   "
            f"{'YES' if cell['survives_bh'] else '-':>5}"
        )

    survivors = [c for c in cells if c["survives_bh"]]
    if survivors:
        log.info(f"\n{len(survivors)} cell(s) survive BH at alpha={args.alpha}.")
        if args.stage == "discover":
            log.info("These are CANDIDATES, not findings. Re-run with --stage confirm to test them on TEST.")
    else:
        log.info(
            "\nNo cell survives multiplicity correction. Under the pre-registered grid, tactile's "
            "contribution over a matched-temporal pose baseline is not concentrated in any tested "
            "regime -- the population-level null is a regime-level null too."
        )

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "stage": args.stage,
                    "eval_split": eval_split_name,
                    "target_variant": args.target_variant,
                    "horizon_k": args.horizon_k,
                    "sequence_length": args.sequence_length,
                    "causal_window": causal_window,
                    "min_history": args.min_history,
                    "motion_threshold": threshold,
                    "alpha": args.alpha,
                    "n_boot": args.n_boot,
                    "cluster_unit": args.cluster_unit,
                    "grid_size": len(cells),
                    "families": [f.name for f in families],
                    "cells": cells,
                },
                indent=2,
            )
        )
        log.info(f"wrote {args.output}")
    return cells


if __name__ == "__main__":
    main()
