"""Tests for bootstrap_eval.bootstrap_map.

The claim this function backs is the retrieval headline, so the two properties
that make its interval honest are pinned here rather than assumed:

  1. Resampling CLIPS, not windows. Sliding windows overlap by 19 of 20 frames,
     so windows within a clip are near-duplicates. A per-window bootstrap
     treats them as independent evidence and reports an interval that is far
     too tight -- the same error HANDOFF 2.3 documents for the direction probe.
  2. A FIXED gallery. mAP is mean(1/rank) against the whole split, so it
     depends on gallery size; resampling the gallery would mix metric artifact
     into the interval, and duplicated gallery rows tie with the correct target.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bootstrap_eval import bootstrap_map


def _correlated_split(n_clips: int, per_clip: int, dim: int, seed: int = 0,
                      heterogeneous: bool = True):
    """Embeddings with the two structures that make clip clustering matter.

    WITHIN a clip, windows are near-identical -- that is what a 19-of-20-frame
    overlap produces, and it is why a per-window bootstrap over-counts evidence.

    ACROSS clips, difficulty varies: alternating clips are cleanly retrievable
    or not retrievable at all. Real clips differ by participant and activity,
    and this between-clip spread is precisely the variance a per-window
    bootstrap is blind to. With `heterogeneous=False` every clip is equally
    hard, and clustering then has nothing to find -- the two estimators agree,
    which is the correct behaviour, not a bug.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clips, dim))
    q, t, clip_ids = [], [], []
    for c in range(n_clips):
        easy = (c % 2 == 0) if heterogeneous else True
        # One decoy centre per hard clip, not one per window: difficulty is a
        # property OF THE CLIP. Drawing a fresh decoy per window instead would
        # scatter ranks inside the clip, and that within-clip variance is
        # exactly what swamps the between-clip signal clustering exists to
        # capture.
        decoy = rng.normal(size=dim)
        for _ in range(per_clip):
            query = centers[c] + rng.normal(scale=0.01, size=dim)
            anchor = centers[c] if easy else decoy
            q.append(query)
            t.append(anchor + rng.normal(scale=0.01, size=dim))
            clip_ids.append(c)
    return (
        torch.tensor(np.array(q), dtype=torch.float32),
        torch.tensor(np.array(t), dtype=torch.float32),
        np.array(clip_ids),
    )


def test_clustered_interval_is_wider_than_the_naive_one():
    """The whole reason for clustering. If this ever inverts, the interval
    being reported is the too-tight one."""
    q, t, clip_ids = _correlated_split(n_clips=24, per_clip=12, dim=32)

    clustered = bootstrap_map(q, t, 300, np.random.default_rng(0), clip_ids=clip_ids)
    naive = bootstrap_map(q, t, 300, np.random.default_rng(0), clip_ids=None)

    # Observed ratio is ~2.0 on this construction; the homogeneous control
    # below sits at ~0.9, so 1.5 separates them with room for RNG drift.
    assert clustered.std() > naive.std() * 1.5, (
        f"clip-clustered std {clustered.std():.5f} should be materially wider "
        f"than per-window std {naive.std():.5f} when clips differ in difficulty"
    )


def test_estimators_agree_when_clips_are_homogeneous():
    """The converse guard: clustering widens the interval because clips differ,
    not as a blanket inflation. Equally-hard clips must give similar widths, so
    a bug that simply scaled the interval up would fail here."""
    q, t, clip_ids = _correlated_split(
        n_clips=24, per_clip=12, dim=32, heterogeneous=False
    )
    clustered = bootstrap_map(q, t, 300, np.random.default_rng(0), clip_ids=clip_ids)
    naive = bootstrap_map(q, t, 300, np.random.default_rng(0), clip_ids=None)

    assert clustered.std() < naive.std() * 1.3, (
        f"clustered {clustered.std():.5f} vs naive {naive.std():.5f}: with "
        f"equally-hard clips the two estimators should agree"
    )


def test_map_values_are_in_range():
    q, t, clip_ids = _correlated_split(n_clips=10, per_clip=5, dim=8)
    maps = bootstrap_map(q, t, 50, np.random.default_rng(1), clip_ids=clip_ids)
    assert maps.shape == (50,)
    assert np.all(maps > 0.0) and np.all(maps <= 1.0)


def test_is_deterministic_given_a_seed():
    q, t, clip_ids = _correlated_split(n_clips=8, per_clip=4, dim=8)
    a = bootstrap_map(q, t, 30, np.random.default_rng(7), clip_ids=clip_ids)
    b = bootstrap_map(q, t, 30, np.random.default_rng(7), clip_ids=clip_ids)
    np.testing.assert_allclose(a, b)


def test_gallery_stays_fixed_so_perfect_retrieval_scores_one():
    """With orthogonal targets the correct match always ranks first, so every
    draw must be exactly 1.0. If the gallery were resampled with replacement,
    duplicate rows would tie with the correct target under `sim >=
    correct_sims`, inflate ranks, and push this below 1.0."""
    n = 12
    emb = torch.eye(n)
    clip_ids = np.arange(n)
    maps = bootstrap_map(emb, emb, 25, np.random.default_rng(3), clip_ids=clip_ids)
    np.testing.assert_allclose(maps, np.ones(25))


@pytest.mark.parametrize("clustered", [True, False])
def test_handles_a_single_cluster(clustered):
    q, t, clip_ids = _correlated_split(n_clips=1, per_clip=6, dim=8)
    maps = bootstrap_map(
        q, t, 10, np.random.default_rng(0),
        clip_ids=clip_ids if clustered else None,
    )
    assert maps.shape == (10,)
