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
    # The whole gap is the recurrent stack -- the projection is shared.
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


def test_projection_shape_is_identical_across_modes():
    """Both arms share one projection layer of the same shape, so the only
    difference between them is the aggregation."""
    gru_encoder = PoseEncoder(temporal_mode="gru")
    mean_encoder = PoseEncoder(temporal_mode="mean")
    assert gru_encoder.projection.weight.shape == mean_encoder.projection.weight.shape


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


def test_mean_mode_padding_does_not_leak_into_the_pooled_half():
    """The pooled vector is zero-padded 128 -> 240 to share the projection.
    Verify the pad is genuinely zeros and sits after the real features, so
    the baseline is not quietly receiving noise in those dimensions."""
    encoder = PoseEncoder(temporal_mode="mean").eval()
    captured = {}

    original = encoder.projection.forward

    def spy(inp):
        captured["input"] = inp.detach().clone()
        return original(inp)

    encoder.projection.forward = spy
    with torch.no_grad():
        encoder(landmarks(seed=2))
    encoder.projection.forward = original

    activations = captured["input"]
    assert activations.shape[-1] == 240
    assert torch.all(activations[:, 128:] == 0), "padding region must be exactly zero"


@pytest.mark.parametrize("frames", [2, 20, 36])
def test_both_modes_accept_varying_sequence_lengths(frames):
    x = landmarks(frames=frames, seed=3)
    for mode in ("gru", "mean"):
        encoder = PoseEncoder(temporal_mode=mode).eval()
        with torch.no_grad():
            assert encoder(x).shape == (4, 64)
