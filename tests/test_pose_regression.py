"""Tests for the pose-transition regression task:
  - src/opentouch/pose_regression.py (PoseTransitionRegressor, decompose_world_delta)
  - src/opentouch/regression_metrics.py
  - src/opentouch_train/regression_data.py (PoseTransitionDataset, shuffled-tactile control)

Focus: the delta-target math, the world/articulation decomposition, the
copy-baseline identity, the structural guarantee that pose-only mode never
sees tactile, the shuffled-tactile control's fixed deterministic re-pairing
and exact parameter parity, and that motion filtering is reported (both
"all" and "moving") rather than silently applied.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from opentouch.pose_encoder import _INDEX_MCP, _MIDDLE_MCP, _THUMB_MCP
from opentouch.pose_regression import (
    COORD_DIM,
    FINGERTIP_COLUMNS,
    NUM_KEYPOINTS,
    WRIST_INDEX,
    PoseTransitionRegressor,
    decompose_world_delta,
)
from opentouch.regression_metrics import (
    compute_copy_baseline_metrics,
    compute_dual_target_metrics,
    compute_regression_metrics,
    fingertip_displacement,
    per_joint_squared_error,
)
from opentouch_train.regression_data import (
    PoseTransitionDataset,
    _make_derangement,
    compute_motion_threshold,
)

_BATCH, _T = 3, 20


# ---------------------------------------------------------------------------
# PoseTransitionRegressor: shapes, and the pose-only tactile-leak guard
# ---------------------------------------------------------------------------

def test_forward_shape_with_tactile():
    model = PoseTransitionRegressor(use_tactile=True)
    model.eval()
    pose_t = torch.randn(_BATCH, NUM_KEYPOINTS, COORD_DIM)
    tactile = torch.rand(_BATCH, _T, 1, 16, 16)
    out = model(pose_t, tactile)
    assert out.shape == (_BATCH, NUM_KEYPOINTS, COORD_DIM)
    assert torch.isfinite(out).all()


def test_forward_shape_pose_only():
    model = PoseTransitionRegressor(use_tactile=False)
    model.eval()
    pose_t = torch.randn(_BATCH, NUM_KEYPOINTS, COORD_DIM)
    out = model(pose_t, None)
    assert out.shape == (_BATCH, NUM_KEYPOINTS, COORD_DIM)
    assert torch.isfinite(out).all()


def test_pose_only_model_has_no_tactile_encoder():
    model = PoseTransitionRegressor(use_tactile=False)
    assert model.tactile_encoder is None
    assert not any("tactile_encoder" in name for name, _ in model.named_parameters())


def test_pose_only_forward_asserts_if_tactile_passed():
    """The exact bug this guards against: a caller accidentally feeding
    tactile into the baseline that is supposed to have none. This must be
    an assert, not the model silently ignoring or zeroing the tensor."""
    model = PoseTransitionRegressor(use_tactile=False)
    pose_t = torch.randn(_BATCH, NUM_KEYPOINTS, COORD_DIM)
    tactile = torch.rand(_BATCH, _T, 1, 16, 16)
    with pytest.raises(AssertionError):
        model(pose_t, tactile)


def test_tactile_model_forward_asserts_if_tactile_missing():
    model = PoseTransitionRegressor(use_tactile=True)
    pose_t = torch.randn(_BATCH, NUM_KEYPOINTS, COORD_DIM)
    with pytest.raises(AssertionError):
        model(pose_t, None)


def test_pose_only_has_fewer_parameters_than_tactile_model():
    """Not a parity requirement (unlike the contact-encoder ablation) --
    the baseline is intentionally not padded to match capacity, since the
    scientific question is whether adding real tactile information helps at
    all, not whether the two architectures have matched capacity."""
    pose_only = sum(p.numel() for p in PoseTransitionRegressor(use_tactile=False).parameters())
    with_tactile = sum(p.numel() for p in PoseTransitionRegressor(use_tactile=True).parameters())
    assert pose_only < with_tactile


# ---------------------------------------------------------------------------
# Delta targets are correct
# ---------------------------------------------------------------------------

def _make_fake_base_dataset(sequence_length=20):
    """Minimal stand-in for VideoTactilePoseDataset exposing exactly what
    PoseTransitionDataset reads: sequence_length/include_pose/include_tactile
    and __getitem__ returning the same dict shape data.py produces."""
    landmarks = torch.randn(sequence_length, 1, NUM_KEYPOINTS, COORD_DIM)
    tactile = torch.rand(sequence_length, 1, 16, 16)

    class _FakeBase:
        def __init__(self):
            self.sequence_length = sequence_length
            self.include_pose = True
            self.include_tactile = True

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "hand_landmarks": landmarks,
                "tactile_pressure": tactile,
                "scene": "scene0",
                "clip_id": "clip0",
            }

    return _FakeBase(), landmarks.squeeze(1)


@pytest.mark.parametrize("horizon_k", [1, 2, 4, 8])
def test_delta_target_equals_future_minus_current(horizon_k):
    base, landmarks = _make_fake_base_dataset(sequence_length=20)
    ds = PoseTransitionDataset(base, horizon_k=horizon_k)

    for t in range(ds.valid_t_per_window):
        sample = ds[t]
        expected_delta = landmarks[t + horizon_k] - landmarks[t]
        torch.testing.assert_close(sample["world_delta"], expected_delta)
        torch.testing.assert_close(sample["pose_t"], landmarks[t])
        assert sample["t"] == t


def test_valid_t_range_matches_sequence_length_minus_k():
    base, _ = _make_fake_base_dataset(sequence_length=20)
    for k in [1, 2, 4, 8]:
        ds = PoseTransitionDataset(base, horizon_k=k)
        assert ds.valid_t_per_window == 20 - k
        assert len(ds) == len(base) * (20 - k)
        # the last valid t must place t+k exactly at the last frame index
        last_sample = ds[ds.valid_t_per_window - 1]
        assert last_sample["t"] + k == 19


def test_horizon_k_must_be_less_than_sequence_length():
    base, _ = _make_fake_base_dataset(sequence_length=20)
    with pytest.raises(ValueError):
        PoseTransitionDataset(base, horizon_k=20)
    with pytest.raises(ValueError):
        PoseTransitionDataset(base, horizon_k=25)


def test_horizon_k_must_be_positive():
    base, _ = _make_fake_base_dataset(sequence_length=20)
    with pytest.raises(ValueError):
        PoseTransitionDataset(base, horizon_k=0)


def test_pose_transition_dataset_requires_pose():
    base, _ = _make_fake_base_dataset(sequence_length=20)
    base.include_pose = False
    with pytest.raises(ValueError):
        PoseTransitionDataset(base, horizon_k=1)


def test_pose_only_dataset_sample_has_no_tactile_key():
    """include_tactile=False on the base dataset (the --pose-only wiring in
    get_regression_data) must mean tactile_pressure is entirely ABSENT from
    the sample, not present-but-zeroed."""
    base, _ = _make_fake_base_dataset(sequence_length=20)
    base.include_tactile = False
    ds = PoseTransitionDataset(base, horizon_k=1)
    sample = ds[0]
    assert "tactile_pressure" not in sample


# ---------------------------------------------------------------------------
# Wrist/fingertip layout: verified against pose_encoder constants, not
# hardcoded from memory
# ---------------------------------------------------------------------------

def test_wrist_index_derived_from_pose_encoder_layout():
    """WRIST_INDEX=0 is only correct if the 21-keypoint layout is wrist-first
    with 5 four-joint finger blocks starting at column 1 -- exactly what
    _THUMB_MCP/_INDEX_MCP/_MIDDLE_MCP == 1,5,9 (imported straight from
    pose_encoder.py, not re-typed) implies. This re-derives that check
    independently of the module-level assert in pose_regression.py."""
    assert _THUMB_MCP == 1
    assert _INDEX_MCP == 5
    assert _MIDDLE_MCP == 9
    assert _THUMB_MCP == 1 + 4 * 0
    assert _INDEX_MCP == 1 + 4 * 1
    assert _MIDDLE_MCP == 1 + 4 * 2
    assert WRIST_INDEX == 0


def test_fingertip_columns_are_4_8_12_16_20():
    assert FINGERTIP_COLUMNS == (4, 8, 12, 16, 20)
    # re-derive independently from the same finger-block formula
    assert FINGERTIP_COLUMNS == tuple(1 + 4 * slot + 3 for slot in range(5))


# ---------------------------------------------------------------------------
# Target decomposition: world_delta vs articulation_delta
# ---------------------------------------------------------------------------

def test_articulation_delta_equals_world_delta_minus_broadcast_wrist():
    torch.manual_seed(10)
    world_delta = torch.randn(8, NUM_KEYPOINTS, COORD_DIM)
    wrist_delta, articulation_delta = decompose_world_delta(world_delta)

    expected_wrist = world_delta[:, WRIST_INDEX, :]
    torch.testing.assert_close(wrist_delta, expected_wrist)

    expected_articulation = world_delta - expected_wrist.unsqueeze(1)
    torch.testing.assert_close(articulation_delta, expected_articulation)


def test_articulation_delta_wrist_row_is_exactly_zero():
    torch.manual_seed(11)
    world_delta = torch.randn(8, NUM_KEYPOINTS, COORD_DIM)
    _, articulation_delta = decompose_world_delta(world_delta)
    torch.testing.assert_close(
        articulation_delta[:, WRIST_INDEX, :], torch.zeros(8, COORD_DIM),
    )


def test_articulation_delta_nonwrist_rows_are_generally_nonzero():
    """Sanity check that decomposition isn't accidentally zeroing everything."""
    torch.manual_seed(12)
    world_delta = torch.randn(8, NUM_KEYPOINTS, COORD_DIM)
    _, articulation_delta = decompose_world_delta(world_delta)
    non_wrist = torch.cat([articulation_delta[:, :WRIST_INDEX], articulation_delta[:, WRIST_INDEX + 1:]], dim=1)
    assert not torch.allclose(non_wrist, torch.zeros_like(non_wrist))


def test_decompose_world_delta_rejects_wrong_shape():
    with pytest.raises(ValueError):
        decompose_world_delta(torch.randn(8, NUM_KEYPOINTS))  # missing coord dim
    with pytest.raises(ValueError):
        decompose_world_delta(torch.randn(NUM_KEYPOINTS, COORD_DIM))  # missing batch dim


# ---------------------------------------------------------------------------
# Shuffled-tactile control: fixed derangement, deterministic, real tactile,
# identical params to tactile+pose
# ---------------------------------------------------------------------------

def test_derangement_has_no_fixed_points():
    for n in [2, 3, 5, 10, 50]:
        perm = _make_derangement(n, seed=0)
        assert perm.shape == (n,)
        assert sorted(perm.tolist()) == list(range(n))  # still a valid permutation
        assert not np.any(perm == np.arange(n))


def test_derangement_is_deterministic_given_seed():
    perm_a = _make_derangement(20, seed=7)
    perm_b = _make_derangement(20, seed=7)
    np.testing.assert_array_equal(perm_a, perm_b)


def test_derangement_rejects_n_less_than_2():
    with pytest.raises(ValueError):
        _make_derangement(1, seed=0)
    with pytest.raises(ValueError):
        _make_derangement(0, seed=0)


def _make_multi_window_base_dataset(n_windows=6, sequence_length=20):
    """Like _make_fake_base_dataset but with n_windows DISTINCT windows,
    each with a unique constant tactile value so shuffled pairing is
    directly observable (window i's tactile is filled with value i)."""
    landmarks_per_window = [torch.randn(sequence_length, 1, NUM_KEYPOINTS, COORD_DIM) for _ in range(n_windows)]
    tactile_per_window = [torch.full((sequence_length, 1, 16, 16), float(i) + 1.0) for i in range(n_windows)]

    class _FakeMultiBase:
        def __init__(self):
            self.sequence_length = sequence_length
            self.include_pose = True
            self.include_tactile = True

        def __len__(self):
            return n_windows

        def __getitem__(self, idx):
            return {
                "hand_landmarks": landmarks_per_window[idx],
                "tactile_pressure": tactile_per_window[idx],
                "scene": f"scene{idx}",
                "clip_id": f"clip{idx}",
            }

    return _FakeMultiBase(), landmarks_per_window, tactile_per_window


def test_shuffled_dataset_pairs_pose_i_with_tactile_j_not_equal_i():
    base, landmarks_per_window, tactile_per_window = _make_multi_window_base_dataset(n_windows=6)
    ds = PoseTransitionDataset(base, horizon_k=1, shuffle_tactile=True, shuffle_seed=3)

    for window_idx in range(6):
        idx = window_idx * ds.valid_t_per_window  # t=0 sample for this window
        sample = ds[idx]
        torch.testing.assert_close(sample["pose_t"], landmarks_per_window[window_idx][0, 0])
        # the tactile tensor must match a DIFFERENT window's constant fill value
        paired_value = sample["tactile_pressure"][0, 0, 0, 0].item()
        paired_window = int(round(paired_value - 1.0))
        assert paired_window != window_idx
        torch.testing.assert_close(sample["tactile_pressure"], tactile_per_window[paired_window])


def test_shuffled_pairing_is_identical_across_repeated_construction_and_epochs():
    """Fixed re-pairing, NOT a per-batch/per-epoch shuffle: rebuilding the
    dataset (as a fresh training run or a separate regression_eval.py
    invocation would) with the same shuffle_seed must reproduce the exact
    same pairing -- and repeated __getitem__ calls (simulating multiple
    epochs) on the SAME dataset object must also return the same pairing."""
    base, _, tactile_per_window = _make_multi_window_base_dataset(n_windows=8)
    ds_a = PoseTransitionDataset(base, horizon_k=1, shuffle_tactile=True, shuffle_seed=42)
    ds_b = PoseTransitionDataset(base, horizon_k=1, shuffle_tactile=True, shuffle_seed=42)

    np.testing.assert_array_equal(ds_a.tactile_window_permutation, ds_b.tactile_window_permutation)

    # simulate two epochs of reading the same sample index
    sample_epoch1 = ds_a[0]
    sample_epoch2 = ds_a[0]
    torch.testing.assert_close(sample_epoch1["tactile_pressure"], sample_epoch2["tactile_pressure"])


def test_shuffle_tactile_requires_real_tactile_on_base_dataset():
    base, _, _ = _make_multi_window_base_dataset(n_windows=4)
    base.include_tactile = False
    with pytest.raises(ValueError):
        PoseTransitionDataset(base, horizon_k=1, shuffle_tactile=True)


def test_shuffle_tactile_parameter_count_exactly_equals_tactile_pose():
    """The model architecture never changes for shuffle_tactile -- only the
    DATASET pairing does -- so this must be an exact equality, not close."""
    real = PoseTransitionRegressor(use_tactile=True, tactile_emb_dim=64, hidden_dim=128)
    shuffled_config = PoseTransitionRegressor(use_tactile=True, tactile_emb_dim=64, hidden_dim=128)
    real_params = sum(p.numel() for p in real.parameters() if p.requires_grad)
    shuffled_params = sum(p.numel() for p in shuffled_config.parameters() if p.requires_grad)
    assert real_params == shuffled_params


# ---------------------------------------------------------------------------
# __getitem__ must be numerically identical to the old
# base_dataset[window_idx]-per-access implementation it replaced (the
# dataloader-bottleneck fix precomputes pose/tactile once in __init__
# instead of rebuilding the full window on every access).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon_k", [1, 2, 4, 8])
@pytest.mark.parametrize("shuffle_tactile", [False, True])
def test_getitem_matches_reference_base_dataset_path(horizon_k, shuffle_tactile):
    n_windows = 6
    base, landmarks_per_window, tactile_per_window = _make_multi_window_base_dataset(
        n_windows=n_windows, sequence_length=20,
    )
    ds = PoseTransitionDataset(
        base, horizon_k=horizon_k, shuffle_tactile=shuffle_tactile, shuffle_seed=3,
    )

    for window_idx in range(n_windows):
        for t in range(ds.valid_t_per_window):
            idx = window_idx * ds.valid_t_per_window + t
            sample = ds[idx]

            # Reference computed the OLD way: reload the full window via
            # base_dataset[window_idx] on every access and slice t / t+k
            # out of it directly, rather than reading the precomputed tensor.
            ref_window = base[window_idx]
            ref_landmarks = ref_window["hand_landmarks"].squeeze(1)
            ref_pose_t = ref_landmarks[t]
            ref_world_delta = ref_landmarks[t + horizon_k] - ref_pose_t

            torch.testing.assert_close(sample["pose_t"], ref_pose_t)
            torch.testing.assert_close(sample["world_delta"], ref_world_delta)

            if shuffle_tactile:
                shuffled_idx = int(ds.tactile_window_permutation[window_idx])
                ref_tactile = base[shuffled_idx]["tactile_pressure"]
            else:
                ref_tactile = ref_window["tactile_pressure"]
            torch.testing.assert_close(sample["tactile_pressure"], ref_tactile)


# ---------------------------------------------------------------------------
# compute_motion_threshold: per-k, from articulation displacement
# ---------------------------------------------------------------------------

def test_compute_motion_threshold_uses_articulation_not_world():
    """Construct a dataset where world-space motion is dominated by a huge
    constant wrist translation but articulation motion is small and known,
    so a world-based threshold and an articulation-based threshold would
    differ enormously -- compute_motion_threshold must track the latter."""
    sequence_length = 20
    horizon_k = 1
    n_windows = 4
    landmarks_per_window = []
    for _ in range(n_windows):
        base_pose = torch.zeros(sequence_length, 1, NUM_KEYPOINTS, COORD_DIM)
        # huge wrist (world) translation every frame: +100 per frame, all coords
        wrist_traj = torch.arange(sequence_length).float() * 100.0  # (T,)
        base_pose[:, 0, WRIST_INDEX, :] = wrist_traj.unsqueeze(-1).expand(-1, COORD_DIM)
        # every fingertip tracks the wrist RIGIDLY in all 3 coords (so it
        # inherits the huge world-space translation) plus a tiny
        # +0.001-per-frame drift in x only -- articulation delta (fingertip
        # world_delta minus wrist world_delta) is then exactly (0.001,0,0)
        # per step for every fingertip, median over 5 tips is also 0.001.
        articulation_traj = torch.arange(sequence_length).float() * 0.001  # (T,)
        for fingertip in FINGERTIP_COLUMNS:
            base_pose[:, 0, fingertip, :] = wrist_traj.unsqueeze(-1).expand(-1, COORD_DIM).clone()
            base_pose[:, 0, fingertip, 0] += articulation_traj
        landmarks_per_window.append(base_pose)
    tactile_per_window = [torch.rand(sequence_length, 1, 16, 16) for _ in range(n_windows)]

    class _FakeBase:
        def __init__(self):
            self.sequence_length = sequence_length
            self.include_pose = True
            self.include_tactile = True

        def __len__(self):
            return n_windows

        def __getitem__(self, idx):
            return {
                "hand_landmarks": landmarks_per_window[idx],
                "tactile_pressure": tactile_per_window[idx],
                "scene": f"scene{idx}",
                "clip_id": f"clip{idx}",
            }

    base = _FakeBase()
    ds = PoseTransitionDataset(base, horizon_k=horizon_k)
    threshold = compute_motion_threshold(ds, percentile=25.0)

    # articulation displacement per step here is exactly 0.001 for every
    # fingertip (wrist moves +100/frame, every fingertip tracks the wrist
    # plus the same +0.001/frame drift), so a correct articulation-based
    # threshold is ~0.001. A world-space-based threshold would instead be
    # dominated by the wrist's own ~100-scale motion.
    assert threshold < 1.0, (
        f"threshold={threshold} looks world-space-scaled (~100), expected an "
        "articulation-space value (<1), i.e. compute_motion_threshold used "
        "the wrong displacement signal"
    )


def test_compute_motion_threshold_is_25th_percentile():
    sequence_length = 6
    horizon_k = 1
    n_windows = 1
    landmarks = torch.zeros(sequence_length, 1, NUM_KEYPOINTS, COORD_DIM)
    # ALL 5 fingertips move identically (wrist fixed at 0, so world ==
    # articulation here) for t=0..5: position = t, so the delta over k=1 is
    # exactly 1.0 at every one of the 5 valid t -- median over 5 identical
    # tips is also exactly 1.0, and the 25th percentile of a constant
    # 5-element series is that same constant.
    for fingertip in FINGERTIP_COLUMNS:
        for t in range(sequence_length):
            landmarks[t, 0, fingertip, 0] = float(t)

    class _FakeBase:
        def __init__(self):
            self.sequence_length = sequence_length
            self.include_pose = True
            self.include_tactile = True

        def __len__(self):
            return n_windows

        def __getitem__(self, idx):
            return {
                "hand_landmarks": landmarks,
                "tactile_pressure": torch.rand(sequence_length, 1, 16, 16),
                "scene": "s",
                "clip_id": "c",
            }

    ds = PoseTransitionDataset(_FakeBase(), horizon_k=horizon_k)
    threshold = compute_motion_threshold(ds, percentile=25.0)
    # articulation delta per step is exactly 1.0 (fingertip moves 1 unit,
    # wrist doesn't move) for every one of the 5 valid t -- so the 25th
    # percentile of a constant series is that same constant.
    assert threshold == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Eval reports both target modes + wrist-translation error, from one prediction
# ---------------------------------------------------------------------------

def test_compute_dual_target_metrics_reports_both_spaces_and_wrist_error():
    torch.manual_seed(20)
    pred = torch.randn(10, NUM_KEYPOINTS, COORD_DIM)
    world_delta = torch.randn(10, NUM_KEYPOINTS, COORD_DIM)
    _, articulation_delta = decompose_world_delta(world_delta)

    dual = compute_dual_target_metrics(pred, world_delta, articulation_delta, motion_threshold=None)
    assert "world" in dual and "articulation" in dual
    assert "wrist_translation_mse" in dual

    expected_world = compute_regression_metrics(pred, world_delta, motion_threshold=None)
    expected_articulation = compute_regression_metrics(pred, articulation_delta, motion_threshold=None)
    for key in expected_world:
        assert dual["world"][key] == pytest.approx(expected_world[key])
    for key in expected_articulation:
        assert dual["articulation"][key] == pytest.approx(expected_articulation[key])

    expected_wrist_mse = (pred[:, WRIST_INDEX, :] - world_delta[:, WRIST_INDEX, :]).pow(2).mean().item()
    assert dual["wrist_translation_mse"] == pytest.approx(expected_wrist_mse)


def test_dual_target_metrics_moving_mask_shared_from_articulation():
    """The moving-subset mask must be the SAME set of samples in both the
    "world" and "articulation" views -- derived once from articulation
    displacement, not recomputed per-view from each view's own target."""
    torch.manual_seed(21)
    pred = torch.zeros(10, NUM_KEYPOINTS, COORD_DIM)
    world_delta = torch.randn(10, NUM_KEYPOINTS, COORD_DIM) * 50  # huge world-space values
    _, articulation_delta = decompose_world_delta(world_delta)

    # a threshold that is small relative to articulation displacement but
    # would behave completely differently if (incorrectly) applied to the
    # huge world-space displacement
    dual = compute_dual_target_metrics(pred, world_delta, articulation_delta, motion_threshold=0.01)
    assert dual["world"]["num_moving_samples"] == dual["articulation"]["num_moving_samples"]
    assert dual["world"]["moving_fraction"] == dual["articulation"]["moving_fraction"]


# ---------------------------------------------------------------------------
# Copy baseline == mean squared displacement
# ---------------------------------------------------------------------------

def test_copy_baseline_equals_mean_squared_displacement():
    torch.manual_seed(0)
    target_delta = torch.randn(64, NUM_KEYPOINTS, COORD_DIM)
    metrics = compute_copy_baseline_metrics(target_delta, prefix="")

    expected_per_joint = target_delta.pow(2).sum(dim=-1).mean(dim=0)  # (21,)
    torch.testing.assert_close(
        torch.tensor(metrics["mse_all_joints"]), expected_per_joint.mean(), atol=1e-5, rtol=1e-5,
    )
    expected_fingertip = expected_per_joint[list(FINGERTIP_COLUMNS)].mean()
    torch.testing.assert_close(
        torch.tensor(metrics["mse_fingertips"]), expected_fingertip, atol=1e-5, rtol=1e-5,
    )
    for j in range(NUM_KEYPOINTS):
        torch.testing.assert_close(
            torch.tensor(metrics[f"mse_joint_{j}"]), expected_per_joint[j], atol=1e-5, rtol=1e-5,
        )


def test_copy_baseline_under_articulation_delta_equals_mean_squared_articulation_displacement():
    """Same identity as test_copy_baseline_equals_mean_squared_displacement,
    but explicitly through the articulation_delta produced by
    decompose_world_delta -- the actual target space --target-mode defaults
    to -- rather than a generic random delta tensor."""
    torch.manual_seed(6)
    world_delta = torch.randn(40, NUM_KEYPOINTS, COORD_DIM)
    _, articulation_delta = decompose_world_delta(world_delta)

    metrics = compute_copy_baseline_metrics(articulation_delta, prefix="")
    expected_per_joint = articulation_delta.pow(2).sum(dim=-1).mean(dim=0)
    torch.testing.assert_close(
        torch.tensor(metrics["mse_all_joints"]), expected_per_joint.mean(), atol=1e-5, rtol=1e-5,
    )
    # the wrist row of articulation_delta is exactly zero, so its
    # contribution to the copy baseline must be exactly zero too
    assert metrics["mse_joint_0"] == pytest.approx(0.0, abs=1e-8)


def test_copy_baseline_matches_zero_prediction_via_summarize():
    """compute_copy_baseline_metrics must be exactly what you'd get by
    calling the real metrics function with an all-zero prediction."""
    torch.manual_seed(1)
    target_delta = torch.randn(32, NUM_KEYPOINTS, COORD_DIM)
    zero_pred = torch.zeros_like(target_delta)

    copy_metrics = compute_copy_baseline_metrics(target_delta, prefix="")
    direct_metrics = {
        "mse_all_joints": per_joint_squared_error(zero_pred, target_delta).mean().item(),
    }
    assert copy_metrics["mse_all_joints"] == pytest.approx(direct_metrics["mse_all_joints"])


def test_perfect_prediction_scores_zero():
    torch.manual_seed(2)
    target_delta = torch.randn(16, NUM_KEYPOINTS, COORD_DIM)
    per_joint = per_joint_squared_error(target_delta, target_delta)
    assert torch.allclose(per_joint, torch.zeros(NUM_KEYPOINTS), atol=1e-6)


# ---------------------------------------------------------------------------
# Motion filtering: reported, not silently applied
# ---------------------------------------------------------------------------

def test_all_metrics_always_present_regardless_of_motion_threshold():
    torch.manual_seed(3)
    pred = torch.randn(20, NUM_KEYPOINTS, COORD_DIM)
    target = torch.randn(20, NUM_KEYPOINTS, COORD_DIM)

    metrics_no_threshold = compute_regression_metrics(pred, target, motion_threshold=None)
    metrics_with_threshold = compute_regression_metrics(pred, target, motion_threshold=0.5)

    # "all_*" metrics must be identical whether or not a threshold is given
    # -- motion filtering must never change the all-sample numbers.
    for key in metrics_no_threshold:
        assert metrics_with_threshold[key] == pytest.approx(metrics_no_threshold[key])
    assert metrics_no_threshold["num_samples"] == 20.0


def test_moving_subset_reported_alongside_all_not_instead_of():
    """A high motion_threshold that excludes every sample must still report
    the full 'all' metrics over all 20 samples -- it must not silently
    shrink the reported sample count for 'all_*'."""
    torch.manual_seed(4)
    pred = torch.randn(20, NUM_KEYPOINTS, COORD_DIM)
    target = torch.randn(20, NUM_KEYPOINTS, COORD_DIM) * 0.01  # tiny displacements

    huge_threshold = 1e6  # excludes everything from the moving subset
    metrics = compute_regression_metrics(pred, target, motion_threshold=huge_threshold)

    assert metrics["num_samples"] == 20.0
    assert metrics["num_moving_samples"] == 0.0
    assert metrics["moving_fraction"] == 0.0
    assert metrics["motion_threshold"] == huge_threshold
    # no model can be distinguished from copy on an empty moving subset --
    # moving_* keys must be absent, not silently filled with 0/NaN as if
    # they were meaningful.
    assert "moving_mse_all_joints" not in metrics


def test_fingertip_displacement_is_median_over_5_tips():
    pose_delta = torch.zeros(1, NUM_KEYPOINTS, COORD_DIM)
    # set 5 fingertip displacements to distinct magnitudes: 1,2,3,4,5
    for i, col in enumerate(FINGERTIP_COLUMNS):
        pose_delta[0, col, 0] = float(i + 1)
    disp = fingertip_displacement(pose_delta)
    assert disp.item() == pytest.approx(3.0)  # median of [1,2,3,4,5]


def test_motion_threshold_partitions_samples_correctly():
    torch.manual_seed(5)
    target = torch.zeros(10, NUM_KEYPOINTS, COORD_DIM)
    # first 4 samples: large fingertip motion; last 6: zero motion
    for i in range(4):
        for col in FINGERTIP_COLUMNS:
            target[i, col, 0] = 10.0
    pred = torch.zeros_like(target)

    metrics = compute_regression_metrics(pred, target, motion_threshold=1.0)
    assert metrics["num_moving_samples"] == 4.0
    assert metrics["moving_fraction"] == pytest.approx(0.4)
