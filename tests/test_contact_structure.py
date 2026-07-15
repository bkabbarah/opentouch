"""Tests for src/opentouch/contact_structure.py (taxel-to-joint B matrix).

Requires assets/MANO_RIGHT.pkl and the OBJ/layout files described in
scripts/diag_layout.py; these tests build the real B matrices from those
files rather than mocking them, since the whole point of the module is
that the numbers are anatomically grounded.
"""

from __future__ import annotations

import numpy as np
import pytest

from opentouch.contact_structure import (
    LAYOUT_LOWRES,
    NUM_KEYPOINTS,
    NUM_REGIONS,
    NUM_TAXELS,
    _kp21_col_to_region,
    build_region_B,
    build_skinning_B,
    build_taxel_index,
    to_attention_bias,
)


@pytest.fixture(scope="module")
def taxel_order():
    return build_taxel_index()


@pytest.fixture(scope="module")
def skinning_B(taxel_order):
    return build_skinning_B(taxel_order)


@pytest.fixture(scope="module")
def region_B(skinning_B):
    return build_region_B(skinning_B)


# ---------------------------------------------------------------------------
# Taxel index
# ---------------------------------------------------------------------------

def test_taxel_count(taxel_order):
    assert len(taxel_order) == NUM_TAXELS == 169


def test_taxel_order_is_deterministic():
    assert build_taxel_index() == build_taxel_index()


def test_taxel_order_sorted_by_row_then_col(taxel_order):
    keys = [tuple(int(x) for x in k.split("-")) for k in taxel_order]
    assert keys == sorted(keys)


def test_taxel_order_matches_across_layout_files():
    # lowres and highres share an identical valid-taxel key set (verified
    # in scripts/diag_layout.py); the canonical order must not depend on
    # which one build_taxel_index reads.
    assert build_taxel_index() == build_taxel_index(LAYOUT_LOWRES)


def test_taxel_keys_unique(taxel_order):
    assert len(set(taxel_order)) == len(taxel_order)


# ---------------------------------------------------------------------------
# skinning_B
# ---------------------------------------------------------------------------

def test_skinning_B_shape(skinning_B):
    assert skinning_B.shape == (NUM_TAXELS, NUM_KEYPOINTS)


def test_skinning_B_row_sums_to_one(skinning_B):
    np.testing.assert_allclose(skinning_B.sum(axis=1), 1.0, atol=1e-8)


def test_skinning_B_no_nan_or_inf(skinning_B):
    assert np.isfinite(skinning_B).all()


def test_skinning_B_nonnegative(skinning_B):
    assert (skinning_B >= 0).all()


def test_skinning_B_deterministic():
    a = build_skinning_B()
    b = build_skinning_B()
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# region_B
# ---------------------------------------------------------------------------

def test_region_B_shape(region_B):
    assert region_B.shape == (NUM_TAXELS, NUM_REGIONS)


def test_region_B_row_sums_to_one(region_B):
    np.testing.assert_allclose(region_B.sum(axis=1), 1.0, atol=1e-8)


def test_region_B_no_nan(region_B):
    assert np.isfinite(region_B).all()


def test_region_B_is_one_hot(region_B):
    assert np.all((region_B == 0.0) | (region_B == 1.0))


def test_region_B_is_strict_coarsening_of_skinning_B(skinning_B, region_B):
    """region_B must be derivable purely by taking argmax(skinning_B) per
    taxel and bucketing the dominant keypoint column into its finger's
    region -- not an independently constructed labeling."""
    dominant_kp_col = np.argmax(skinning_B, axis=1)
    expected_region = np.array([_kp21_col_to_region(int(c)) for c in dominant_kp_col])
    actual_region = np.argmax(region_B, axis=1)
    np.testing.assert_array_equal(actual_region, expected_region)


def test_thumb_taxels_route_to_thumb_columns(skinning_B, region_B):
    thumb_rows = np.where(np.argmax(region_B, axis=1) == 1)[0]
    assert thumb_rows.size > 0
    dominant_cols = np.argmax(skinning_B[thumb_rows], axis=1)
    assert set(int(c) for c in dominant_cols).issubset({1, 2, 3, 4})


# ---------------------------------------------------------------------------
# to_attention_bias
# ---------------------------------------------------------------------------

def test_bias_skinning_shape(skinning_B):
    bias = to_attention_bias(skinning_B)
    assert bias.shape == (NUM_KEYPOINTS, NUM_TAXELS)


def test_bias_region_shape(region_B):
    bias = to_attention_bias(region_B)
    assert bias.shape == (NUM_REGIONS, NUM_TAXELS)


def test_bias_is_finite_and_floored(skinning_B):
    bias = to_attention_bias(skinning_B, floor=-10.0)
    assert np.isfinite(bias).all()
    assert bias.min() >= -10.0


def test_bias_zero_correspondence_maps_to_floor():
    B = np.zeros((NUM_TAXELS, 3), dtype=np.float64)
    B[:, 0] = 1.0
    bias = to_attention_bias(B, floor=-7.5)
    assert np.all(bias[1] == -7.5)
    assert np.all(bias[2] == -7.5)
    assert np.all(bias[0] == 0.0)  # log(1) == 0
