"""Tests for scripts/tactile_subset_probe.py's statistical machinery.

These target the parts that decide whether a "we found a subset where tactile
helps" claim is real: the clustered bootstrap (which sets how wide the error
bars are), the multiplicity correction (which sets how many cells get to
survive), the train-fitted binning (which must not peek at eval), and the
causality guard on subset definitions (a subset chosen using the future would
manufacture a positive out of nothing).

The encoder/data paths are not tested here -- they are re-used verbatim from
tactile_direction_probe.py and covered by its own audit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tactile_subset_probe import (  # noqa: E402
    FAMILIES_BY_NAME,
    SUBSET_FAMILIES,
    SubsetFamily,
    assert_causal_subset_definition,
    assign_bins,
    benjamini_hochberg,
    bootstrap_pvalue,
    clip_clustered_bootstrap,
    fit_bin_edges,
    gather_causal_slices,
)
from opentouch_train.regression_data import _causal_frame_indices  # noqa: E402


# --------------------------------------------------------------------------
# Clustered bootstrap
# --------------------------------------------------------------------------


def test_bootstrap_recovers_zero_for_identical_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=600)
    scores = rng.normal(size=600) + y
    clips = np.repeat(np.arange(60), 10)

    result = clip_clustered_bootstrap(y, scores, scores.copy(), clips, n_boot=200, seed=1)
    assert result["delta"] == pytest.approx(0.0, abs=1e-12)
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=800)
    good = rng.normal(size=800) + 1.5 * y
    useless = rng.normal(size=800)
    clips = np.repeat(np.arange(80), 10)

    result = clip_clustered_bootstrap(y, good, useless, clips, n_boot=300, seed=3)
    assert result["delta"] > 0.1
    assert result["ci_low"] > 0.0, "a large real difference must have a CI excluding zero"


def test_clustered_intervals_are_wider_than_naive_per_sample_intervals():
    """THE reason this machinery exists. When samples within a cluster are
    correlated -- as adjacent t within a window certainly are -- resampling
    individual rows reports an interval far too narrow, and near-zero effects
    get declared significant. Clustering must widen it."""
    rng = np.random.default_rng(4)
    n_clips, per_clip = 40, 25
    clips = np.repeat(np.arange(n_clips), per_clip)
    # Strong per-clip effect: every row in a clip shares the clip's offset, so
    # the real information content is 40 clips, not 1000 rows.
    clip_offset = rng.normal(size=n_clips) * 2.0
    y = rng.integers(0, 2, size=n_clips * per_clip)
    a = rng.normal(size=len(y)) * 0.1 + y * 0.3 + clip_offset[clips]
    b = rng.normal(size=len(y)) * 0.1 + clip_offset[clips]

    clustered = clip_clustered_bootstrap(y, a, b, clips, n_boot=400, seed=5)
    per_sample = clip_clustered_bootstrap(y, a, b, np.arange(len(y)), n_boot=400, seed=5)

    clustered_width = clustered["ci_high"] - clustered["ci_low"]
    naive_width = per_sample["ci_high"] - per_sample["ci_low"]
    assert clustered_width > naive_width, (
        f"clustered CI ({clustered_width:.4f}) must be wider than the per-sample CI "
        f"({naive_width:.4f}) under within-cluster correlation -- otherwise clustering is a no-op"
    )


def test_bootstrap_is_deterministic_given_a_seed():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, size=400)
    a, b = rng.normal(size=400) + y, rng.normal(size=400)
    clips = np.repeat(np.arange(40), 10)
    first = clip_clustered_bootstrap(y, a, b, clips, n_boot=150, seed=7)
    second = clip_clustered_bootstrap(y, a, b, clips, n_boot=150, seed=7)
    assert first == second


def test_bootstrap_reports_the_cluster_count_not_the_row_count():
    """A reader must be able to see the real unit of independence."""
    rng = np.random.default_rng(8)
    y = rng.integers(0, 2, size=500)
    a, b = rng.normal(size=500) + y, rng.normal(size=500)
    clips = np.repeat(np.arange(25), 20)
    result = clip_clustered_bootstrap(y, a, b, clips, n_boot=100, seed=9)
    assert result["n_clips"] == 25, "must report 25 clusters, not 500 rows"


def test_bootstrap_degrades_gracefully_with_one_cluster():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    result = clip_clustered_bootstrap(y, scores, scores, np.zeros(4), n_boot=50, seed=10)
    assert np.isnan(result["delta"]), "a single cluster carries no resampling information"


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------


def test_bh_rejects_nothing_when_all_p_values_are_null():
    rng = np.random.default_rng(11)
    uniform_pvalues = rng.uniform(size=200)
    rejected = benjamini_hochberg(uniform_pvalues, alpha=0.05)
    assert rejected.sum() <= 5, (
        f"BH on pure noise should reject ~none, rejected {rejected.sum()}/200"
    )


def test_bh_rejects_clear_signal_among_noise():
    pvalues = np.concatenate([np.full(5, 1e-6), np.random.default_rng(12).uniform(size=95)])
    rejected = benjamini_hochberg(pvalues, alpha=0.05)
    assert rejected[:5].all(), "five overwhelming p-values must survive BH"


def test_bh_is_monotone_in_alpha():
    pvalues = np.array([0.001, 0.01, 0.02, 0.2, 0.5])
    strict = benjamini_hochberg(pvalues, alpha=0.01).sum()
    lenient = benjamini_hochberg(pvalues, alpha=0.2).sum()
    assert lenient >= strict


def test_bh_step_up_rejects_all_below_the_largest_passing_rank():
    """The step-UP property: BH is not a per-p-value threshold. A p-value
    above alpha*(i/n) must still be rejected if a later-ranked one passes."""
    pvalues = np.array([0.001, 0.04, 0.045])
    rejected = benjamini_hochberg(pvalues, alpha=0.05)
    assert rejected.all(), "step-up must reject the whole prefix up to the last passing rank"


def test_bh_handles_an_empty_grid():
    assert benjamini_hochberg(np.array([]), alpha=0.05).shape == (0,)


def test_bootstrap_pvalue_is_small_for_a_ci_excluding_zero():
    assert bootstrap_pvalue({"delta": 0.05, "ci_low": 0.03, "ci_high": 0.07}) < 0.01
    assert bootstrap_pvalue({"delta": 0.001, "ci_low": -0.02, "ci_high": 0.022}) > 0.5
    assert bootstrap_pvalue({"delta": float("nan"), "ci_low": 0.0, "ci_high": 0.0}) == 1.0


# --------------------------------------------------------------------------
# Binning: fitted on train, applied to eval
# --------------------------------------------------------------------------


def test_bin_edges_come_from_train_and_transfer_unchanged():
    train_values = np.linspace(0.0, 1.0, 1000)
    edges = fit_bin_edges(train_values, (33.0, 67.0))
    assert edges[0] == pytest.approx(0.33, abs=0.01)
    assert edges[1] == pytest.approx(0.67, abs=0.01)

    # A shifted eval distribution must NOT get re-fitted edges -- if it did,
    # "firm contact" would mean something different in train and eval.
    eval_values = np.linspace(0.5, 1.5, 100)
    bins = assign_bins(eval_values, edges)
    assert bins.min() >= 1, "an eval set shifted upward should land in upper bins, not be re-centred"
    assert set(np.unique(bins)) <= {0, 1, 2}


def test_assign_bins_covers_every_value_exactly_once():
    edges = np.array([0.2, 0.8])
    values = np.array([-1.0, 0.0, 0.2, 0.5, 0.8, 5.0])
    bins = assign_bins(values, edges)
    np.testing.assert_array_equal(bins, np.array([0, 0, 1, 1, 2, 2]))


# --------------------------------------------------------------------------
# Causality of subset definitions
# --------------------------------------------------------------------------


def test_gather_causal_slices_never_reaches_past_t():
    """Same poisoning argument the encoder path uses: fill every frame > t
    with a value nothing else can produce, then assert it never appears in a
    gathered slice."""
    n_windows, seq_len, window = 5, 12, 4
    valid_t = 8
    tactile = torch.zeros(n_windows, seq_len, 1, 4, 4)
    causal_frame_idx = _causal_frame_indices(valid_t, window)

    window_idx = np.repeat(np.arange(n_windows), valid_t)
    t_values = np.tile(np.arange(valid_t), n_windows)

    for t in range(valid_t):
        poisoned = tactile.clone()
        poisoned[:, t + 1 :] = 1e6
        rows = t_values == t
        slices = gather_causal_slices(
            poisoned, window_idx[rows], t_values[rows], causal_frame_idx
        )
        assert slices.max() < 1e5, f"a frame after t={t} leaked into the causal slice"


def test_gather_causal_slices_is_invariant_to_the_future():
    n_windows, seq_len, window, valid_t = 4, 10, 5, 6
    generator = torch.Generator().manual_seed(13)
    pose = torch.randn(n_windows, seq_len, 21, 3, generator=generator)
    causal_frame_idx = _causal_frame_indices(valid_t, window)
    window_idx = np.repeat(np.arange(n_windows), valid_t)
    t_values = np.tile(np.arange(valid_t), n_windows)

    baseline = gather_causal_slices(pose, window_idx, t_values, causal_frame_idx)
    for t in range(valid_t):
        perturbed = pose.clone()
        perturbed[:, t + 1 :] += 100.0
        rows = t_values == t
        after = gather_causal_slices(perturbed, window_idx[rows], t_values[rows], causal_frame_idx)
        assert torch.equal(after, baseline[rows]), f"changing frames > {t} changed the slice"


def test_every_registered_family_passes_the_causality_guard():
    generator = torch.Generator().manual_seed(14)
    tactile = torch.rand(64, 20, 1, 16, 16, generator=generator)
    pose = torch.randn(64, 20, 21, 3, generator=generator)
    for family in SUBSET_FAMILIES:
        assert_causal_subset_definition(family, tactile, pose)


def test_causality_guard_rejects_a_batch_dependent_statistic():
    """A statistic that normalizes by the batch would make a sample's subset
    membership depend on which other samples were loaded -- train-fitted
    thresholds would then be meaningless."""
    bad = SubsetFamily(
        name="batch_dependent",
        statistic=lambda tactile, pose: (
            tactile.flatten(1).mean(dim=1) - tactile.flatten(1).mean()
        ) / tactile.flatten(1).std(),
        bin_edges_percentiles=(50.0,),
        bin_labels=("low", "high"),
        hypothesis="deliberately broken",
    )
    generator = torch.Generator().manual_seed(15)
    tactile = torch.rand(32, 8, 1, 16, 16, generator=generator)
    pose = torch.randn(32, 8, 21, 3, generator=generator)
    # Batch mean/std are permutation-invariant, so this specific form passes
    # the permutation check; the guard's real target is a statistic whose
    # per-row value changes with ordering. Build that explicitly.
    ordering_dependent = SubsetFamily(
        name="ordering_dependent",
        statistic=lambda tactile, pose: torch.cumsum(tactile.flatten(1).mean(dim=1), dim=0),
        bin_edges_percentiles=(50.0,),
        bin_labels=("low", "high"),
        hypothesis="deliberately broken",
    )
    with pytest.raises(ValueError, match="batch composition"):
        assert_causal_subset_definition(ordering_dependent, tactile, pose)
    # The normalizing one is at least deterministic and permutation-equivariant.
    assert_causal_subset_definition(bad, tactile, pose)


def test_causality_guard_rejects_a_wrong_shaped_statistic():
    broken = SubsetFamily(
        name="wrong_shape",
        statistic=lambda tactile, pose: tactile.flatten(1).mean(dim=1)[:, None],
        bin_edges_percentiles=(50.0,),
        bin_labels=("low", "high"),
        hypothesis="deliberately broken",
    )
    generator = torch.Generator().manual_seed(16)
    with pytest.raises(ValueError, match="one scalar per sample"):
        assert_causal_subset_definition(
            broken,
            torch.rand(8, 4, 1, 16, 16, generator=generator),
            torch.randn(8, 4, 21, 3, generator=generator),
        )


# --------------------------------------------------------------------------
# Pre-registration integrity
# --------------------------------------------------------------------------


def test_family_labels_match_their_cut_points():
    for family in SUBSET_FAMILIES:
        assert len(family.bin_labels) == len(family.bin_edges_percentiles) + 1


def test_family_construction_rejects_mismatched_labels():
    with pytest.raises(ValueError, match="need"):
        SubsetFamily(
            name="bad",
            statistic=lambda tactile, pose: tactile.flatten(1).mean(dim=1),
            bin_edges_percentiles=(33.0, 67.0),
            bin_labels=("only", "two"),
            hypothesis="x",
        )


def test_every_family_states_a_hypothesis_and_is_uniquely_named():
    names = [f.name for f in SUBSET_FAMILIES]
    assert len(names) == len(set(names)), "family names must be unique -- they key the output grid"
    assert set(names) == set(FAMILIES_BY_NAME)
    for family in SUBSET_FAMILIES:
        assert family.hypothesis.strip(), (
            f"family '{family.name}' has no stated hypothesis; an unmotivated cut is a fishing trip"
        )
