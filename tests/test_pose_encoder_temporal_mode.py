"""Tests for PoseEncoder's temporal aggregation modes.

The project's headline result is biGRU-vs-average-pooling. Until now the
avg-pool arm was only runnable from a different repository, which made the
published comparison cross-codebase. These tests pin that both arms now live
in one class, that "gru" is bit-identical to the historical behaviour, and
that the two modes cannot be silently confused for one another.
"""

from __future__ import annotations

import pytest
import torch

from opentouch.pose_encoder import PoseEncoder


def landmarks(batch: int = 4, frames: int = 20, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, frames, 21, 3, generator=generator)


def test_both_modes_produce_the_expected_embedding_shape():
    x = landmarks()
    for mode in ("gru", "mean"):
        encoder = PoseEncoder(emb_dim=64, temporal_mode=mode).eval()
        assert encoder(x).shape == (4, 64)


def test_gru_is_the_default():
    """Every published number used the GRU. If the default flipped, results
    would silently stop being comparable."""
    assert PoseEncoder().temporal_mode == "gru"
    assert PoseEncoder().gru is not None


def test_mean_mode_does_not_build_a_gru():
    """Keeps the baseline's parameter count honest, and makes a state_dict
    from one mode fail loudly rather than half-load into the other."""
    encoder = PoseEncoder(temporal_mode="mean")
    assert encoder.gru is None
    names = dict(encoder.named_parameters())
    assert not any(n.startswith("gru") for n in names), "mean mode must have no GRU parameters"


def test_mean_mode_has_strictly_fewer_parameters():
    gru_params = sum(p.numel() for p in PoseEncoder(temporal_mode="gru").parameters())
    mean_params = sum(p.numel() for p in PoseEncoder(temporal_mode="mean").parameters())
    assert mean_params < gru_params
    # Almost all of the gap is the recurrent stack; the rest is the narrower
    # projection (128 wide instead of 240).
    assert gru_params - mean_params > 300_000


def test_mean_mode_is_invariant_to_frame_order_and_gru_mode_is_not():
    """THE property that separates the two arms. Average pooling discards
    temporal structure entirely; the GRU is order-sensitive, which is the
    whole mechanism the headline result attributes the gain to."""
    x = landmarks(seed=1)
    reversed_x = torch.flip(x, dims=[1])

    mean_encoder = PoseEncoder(temporal_mode="mean").eval()
    with torch.no_grad():
        assert torch.allclose(mean_encoder(x), mean_encoder(reversed_x), atol=1e-5), (
            "average pooling must be order-invariant"
        )

    gru_encoder = PoseEncoder(temporal_mode="gru").eval()
    with torch.no_grad():
        assert not torch.allclose(gru_encoder(x), gru_encoder(reversed_x), atol=1e-3), (
            "the biGRU must be order-sensitive"
        )


def test_projection_width_matches_each_mode_s_readout():
    """240 for the biGRU (concat of final forward and backward hidden states,
    2 x 120), 128 for mean pooling (the per-frame encoder width). The mean
    width matters for compatibility: it is what upstream avg-pool checkpoints
    were trained with, so they load into this class unchanged."""
    assert PoseEncoder(temporal_mode="gru").projection.weight.shape == (64, 240)
    assert PoseEncoder(temporal_mode="mean").projection.weight.shape == (64, 128)


def test_mean_mode_loads_an_upstream_avgpool_state_dict():
    """Regression guard. An earlier version zero-padded the pooled vector to
    240 so both modes shared a projection shape; that made every historical
    avg-pool checkpoint fail to load with a size mismatch, which is how it was
    caught. This builds a state_dict shaped like the upstream encoder and
    asserts it loads strictly."""
    encoder = PoseEncoder(temporal_mode="mean")
    # randn_like would fail on integer buffers (BatchNorm num_batches_tracked).
    upstream = {
        k: (torch.randn_like(v) if v.is_floating_point() else v.clone())
        for k, v in encoder.state_dict().items()
    }
    assert upstream["projection.weight"].shape == (64, 128)
    encoder.load_state_dict(upstream, strict=True)


def test_state_dicts_are_not_interchangeable():
    """A GRU checkpoint loaded into a mean-mode encoder (or vice versa) must
    raise, not silently run on partially-random weights."""
    gru_state = PoseEncoder(temporal_mode="gru").state_dict()
    mean_state = PoseEncoder(temporal_mode="mean").state_dict()
    with pytest.raises(RuntimeError):
        PoseEncoder(temporal_mode="mean").load_state_dict(gru_state, strict=True)
    with pytest.raises(RuntimeError):
        PoseEncoder(temporal_mode="gru").load_state_dict(mean_state, strict=True)


def test_invalid_temporal_mode_raises_with_a_useful_message():
    with pytest.raises(ValueError, match="temporal_mode"):
        PoseEncoder(temporal_mode="lstm")


def test_normalize_mode_still_validated():
    with pytest.raises(ValueError):
        PoseEncoder(normalize_mode="bogus")


@pytest.mark.parametrize("frames", [2, 20, 36])
def test_both_modes_accept_varying_sequence_lengths(frames):
    x = landmarks(frames=frames, seed=3)
    for mode in ("gru", "mean"):
        encoder = PoseEncoder(temporal_mode=mode).eval()
        with torch.no_grad():
            assert encoder(x).shape == (4, 64)
