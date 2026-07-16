"""Tests for src/opentouch/tactile_contact_encoder.py (taxel-token tactile
encoder with contact-structured cross-attention, plus the plain-attention
baseline). Uses the real assets/B_matrices.npz built by contact_structure.py
-- the whole point of the anatomical bias is that the numbers come from
MANO skinning weights, so a mock would test nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from opentouch.contact_structure import (
    LAYOUT_HIGHRES,
    NUM_KEYPOINTS,
    NUM_TAXELS,
    build_taxel_index,
)
from opentouch.tactile_contact_encoder import (
    TactileContactEncoder,
    TaxelTokenizer,
    _region_bias_to_kp21,
    count_parameters,
)

_CONTACT_MODES = ("skinning", "region", "plain")
_BATCH, _T = 2, 20


def _random_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(_BATCH, _T, 1, 16, 16)


# ---------------------------------------------------------------------------
# Forward pass shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _CONTACT_MODES)
def test_forward_shape_contact_modes(mode):
    model = TactileContactEncoder(emb_dim=64, mode=mode)
    model.eval()
    out = model(_random_input())
    assert out.shape == (_BATCH, 64)
    assert torch.isfinite(out).all()


def test_forward_shape_cnn_gru_baseline_unaffected():
    """The pre-existing CNN+biGRU encoder is untouched by this change."""
    from opentouch.tactile_encoder import CNNetEmbedding

    model = CNNetEmbedding(emb_dim=64)
    model.eval()
    out = model(_random_input())
    assert out.shape == (_BATCH, 64)


def test_forward_accepts_4d_input_without_channel_dim():
    model = TactileContactEncoder(emb_dim=64, mode="plain")
    model.eval()
    out = model(_random_input().squeeze(2))
    assert out.shape == (_BATCH, 64)


# ---------------------------------------------------------------------------
# Token order / count -- build_taxel_index() is the single source of truth
# ---------------------------------------------------------------------------

def test_taxel_count_is_169():
    model = TactileContactEncoder(mode="skinning")
    assert len(model.taxel_order) == NUM_TAXELS == 169


def test_token_order_matches_build_taxel_index():
    model = TactileContactEncoder(mode="skinning")
    assert model.taxel_order == build_taxel_index()


def test_gather_indices_match_taxel_order():
    """taxel_rows/taxel_cols must decode the same keys as taxel_order, in
    the same order -- this is the exact assert that catches a transposed B."""
    model = TactileContactEncoder(mode="skinning")
    canonical = build_taxel_index()
    for i, key in enumerate(canonical):
        r, c = (int(x) for x in key.split("-"))
        assert int(model.taxel_rows[i]) == r
        assert int(model.taxel_cols[i]) == c


# ---------------------------------------------------------------------------
# Temporal aggregator must be order-SENSITIVE (regression test for the
# mean-pool bug: T->P mAP was 5.18/6.46/7.04 vs cnn_gru's 45.46 and the
# published 14.33 baseline, because mean-pooling over T cannot distinguish a
# grasp from a release -- exactly the direction-in-time signal the
# tactile-predicts-pose-transitions claim depends on).
# ---------------------------------------------------------------------------

def test_temporal_aggregator_is_order_sensitive():
    """Reversing a taxel's pressure trace over T must change its token. A
    permutation-invariant aggregator (e.g. the old Conv1d + mean-pool) would
    produce an IDENTICAL token here -- this is the exact bug this fix
    addresses, caught directly rather than inferred from downstream mAP."""
    torch.manual_seed(3)
    taxel_order = build_taxel_index()
    tokenizer = TaxelTokenizer(d_model=32, taxel_order=taxel_order, layout_path=LAYOUT_HIGHRES)
    tokenizer.eval()

    pressure = torch.rand(2, _T, NUM_TAXELS)
    pressure_reversed = pressure.flip(dims=[1])

    with torch.no_grad():
        tokens = tokenizer(pressure)
        tokens_reversed = tokenizer(pressure_reversed)

    assert not torch.allclose(tokens, tokens_reversed, atol=1e-4)


def test_full_encoder_output_is_order_sensitive():
    """End-to-end version of the same check: reversing the whole (B,T,1,16,16)
    tactile window must change the encoder's output embedding."""
    torch.manual_seed(4)
    model = TactileContactEncoder(mode="plain")
    model.eval()

    x = torch.rand(2, _T, 1, 16, 16)
    x_reversed = x.flip(dims=[1])

    with torch.no_grad():
        out = model(x)
        out_reversed = model(x_reversed)

    assert not torch.allclose(out, out_reversed, atol=1e-4)


# ---------------------------------------------------------------------------
# "plain" mode has no anatomical bias, but DOES have the learnable residual
# ---------------------------------------------------------------------------

def test_plain_mode_has_no_anat_bias_but_has_residual():
    model = TactileContactEncoder(mode="plain")
    assert not hasattr(model, "anat_bias") or model.anat_bias is None
    assert "anat_bias" not in dict(model.named_buffers())

    # anat_residual must be present in plain mode -- this is the parameter-
    # parity fix. A plain baseline that cannot learn ANY attention bias would
    # confound "no anatomical prior" with "no learnable bias at all".
    assert model.anat_residual.shape == (NUM_KEYPOINTS, NUM_TAXELS)
    assert torch.equal(model.anat_residual, torch.zeros(NUM_KEYPOINTS, NUM_TAXELS))
    assert "anat_residual" in dict(model.named_parameters())
    assert model.anat_residual.requires_grad


@pytest.mark.parametrize("mode", ["skinning", "region"])
def test_structured_modes_register_bias_and_residual(mode):
    model = TactileContactEncoder(mode=mode)
    assert model.anat_bias.shape == (NUM_KEYPOINTS, NUM_TAXELS)
    assert model.anat_residual.shape == (NUM_KEYPOINTS, NUM_TAXELS)
    assert "anat_bias" in dict(model.named_buffers())
    assert "anat_residual" in dict(model.named_parameters())


# ---------------------------------------------------------------------------
# Parameter counts across modes: EXACT parity, not a tolerance
# ---------------------------------------------------------------------------

def test_parameter_counts_are_exactly_equal_across_contact_modes():
    counts = {
        mode: count_parameters(TactileContactEncoder(mode=mode))
        for mode in _CONTACT_MODES
    }
    for mode, n in counts.items():
        print(f"{mode:10s}: {n:,} trainable parameters")

    # skinning/region/plain must be architecturally identical in every
    # trainable parameter: same tokenizer, same q/k/v/out projections, same
    # joint_queries, and now the same (21,169) anat_residual in all three.
    # anat_bias differs only in VALUES (or is absent, in plain), and it is a
    # non-trainable buffer either way, so it contributes 0 to this count.
    # A skinning-vs-plain comparison that isn't exactly parameter-matched
    # would confound the anatomical prior with extra free capacity -- the
    # exact question this ablation exists to answer -- so this is an exact
    # equality assert, not a tolerance.
    assert counts["skinning"] == counts["region"] == counts["plain"], counts


# ---------------------------------------------------------------------------
# Bias enters BEFORE softmax
# ---------------------------------------------------------------------------

def test_attention_weights_are_a_valid_softmax_distribution():
    """If the bias were (incorrectly) added after softmax, attn would not
    sum to 1 along the key dimension once a nonzero bias is present."""
    model = TactileContactEncoder(mode="skinning")
    model.eval()
    with torch.no_grad():
        model(_random_input())
    attn = model.last_attn_weights
    assert attn is not None
    row_sums = attn.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=0)


def test_large_bias_dominates_attention_pre_softmax():
    """Directly demonstrates pre-softmax injection: an artificially extreme
    anat_bias column overwhelms the learned q@k logits (which are small by
    comparison) and the softmax concentrates almost all mass there -- this
    can only happen if the bias is added to logits before normalization."""
    model = TactileContactEncoder(mode="skinning")
    model.eval()
    with torch.no_grad():
        model.anat_bias.fill_(-50.0)
        model.anat_bias[:, 3] = 50.0
        model.anat_residual.zero_()
        model(_random_input())
    attn = model.last_attn_weights  # (B,H,21,169)
    assert (attn[..., 3] > 0.99).all()


def test_zero_bias_and_residual_reduces_to_plain_style_logits():
    """With anat_bias and anat_residual both zeroed, a structured-mode
    forward pass produces the same attention as plain mode given identical
    q/k -- confirms the bias is a pure additive term on the logits, not
    applied through some other path."""
    torch.manual_seed(1)
    structured = TactileContactEncoder(mode="skinning")
    plain = TactileContactEncoder(mode="plain")

    # Copy every parameter plain has onto structured so q/k/v/tokenizer match.
    plain_state = dict(plain.named_parameters())
    plain_buffers = dict(plain.named_buffers())
    with torch.no_grad():
        for name, param in structured.named_parameters():
            if name in plain_state:
                param.copy_(plain_state[name])
        for name, buf in structured.named_buffers():
            if name in plain_buffers:
                buf.copy_(plain_buffers[name])
        structured.anat_bias.zero_()
        structured.anat_residual.zero_()
        plain.anat_residual.zero_()  # already zero at init, explicit for clarity

    structured.eval()
    plain.eval()
    x = _random_input()
    with torch.no_grad():
        structured(x)
        plain(x)
    torch.testing.assert_close(structured.last_attn_weights, plain.last_attn_weights)


def test_plain_at_init_is_flat_bias_and_matches_no_bias_reference():
    """plain's anat_residual inits to zeros, so at construction time (before
    any training) its logits are exactly q@k/scale -- numerically identical
    to a hypothetical implementation with no additive bias term at all. This
    is what makes "plain" a fair, capacity-matched, but anatomically-blank
    starting point rather than a different architecture."""
    torch.manual_seed(2)
    plain = TactileContactEncoder(mode="plain")
    assert torch.equal(plain.anat_residual, torch.zeros(NUM_KEYPOINTS, NUM_TAXELS))
    plain.eval()
    x = _random_input()

    with torch.no_grad():
        plain(x)
    attn_with_module_bias_path = plain.last_attn_weights

    # Independently recompute q@k/scale -> softmax with no bias term of any
    # kind, using the same tokenizer/projections plain already has.
    from opentouch.tactile_encoder import _normalize_input
    flat, b, t = _normalize_input(x)
    grid = flat.view(b, t, 16, 16)
    pressure_per_taxel = grid[:, :, plain.taxel_rows, plain.taxel_cols]
    with torch.no_grad():
        tokens = plain.tokenizer(pressure_per_taxel)
        q = plain.q_proj(plain.joint_queries)[None].expand(b, -1, -1)
        k = plain.k_proj(tokens)
        q = q.view(b, NUM_KEYPOINTS, plain.num_heads, plain.head_dim).transpose(1, 2)
        k = k.view(b, NUM_TAXELS, plain.num_heads, plain.head_dim).transpose(1, 2)
        logits_no_bias = (q @ k.transpose(-1, -2)) / plain.scale
        expected_attn = logits_no_bias.softmax(dim=-1)

    torch.testing.assert_close(attn_with_module_bias_path, expected_attn)


# ---------------------------------------------------------------------------
# Gradients: anat_residual yes, anat_bias no
# ---------------------------------------------------------------------------

def test_gradients_flow_to_residual_not_to_bias():
    model = TactileContactEncoder(mode="skinning")
    model.train()
    out = model(_random_input())
    out.sum().backward()

    assert model.anat_residual.grad is not None
    assert torch.any(model.anat_residual.grad != 0)
    assert not model.anat_bias.requires_grad
    assert model.anat_bias.grad is None


def test_gradients_flow_to_residual_in_plain_mode_too():
    """plain has no anat_bias to check, but its anat_residual must be a real
    learnable parameter -- this is what lets plain learn taxel-to-joint
    routing from data instead of being structurally stuck at uniform bias."""
    model = TactileContactEncoder(mode="plain")
    assert not hasattr(model, "anat_bias") or model.anat_bias is None
    model.train()
    out = model(_random_input())
    out.sum().backward()

    assert model.anat_residual.grad is not None
    assert torch.any(model.anat_residual.grad != 0)


# ---------------------------------------------------------------------------
# Region bias broadcast
# ---------------------------------------------------------------------------

def test_region_bias_broadcast_shape_and_grouping():
    from opentouch.contact_structure import (
        NUM_REGIONS,
        _kp21_col_to_region,
    )

    bias_region = np.random.default_rng(0).normal(size=(NUM_REGIONS, NUM_TAXELS))
    bias_21 = _region_bias_to_kp21(bias_region)
    assert bias_21.shape == (NUM_KEYPOINTS, NUM_TAXELS)
    for c in range(NUM_KEYPOINTS):
        np.testing.assert_array_equal(bias_21[c], bias_region[_kp21_col_to_region(c)])
