"""Tests for the tactile_reduce ablation on PoseTransitionRegressor.

WHAT THIS ABLATION IS FOR. Three separate observations in this project are all
consistent with the tactile benefit being a single scalar -- contact magnitude
-- rather than a spatial representation: it is learnable within 2-4 epochs
(HANDOFF 2.22), a random frozen encoder captures all of it (2.32), and
deranging the temporal pairing destroys it. `tactile_reduce="scalar"` keeps the
per-frame TOTAL pressure exactly and destroys every spatial pattern, so
comparing it against "none" isolates whether WHERE the hand touches matters at
all.

The load-bearing test here is the first one: "none" must be bit-identical to
the behaviour before this flag existed, or every historical result silently
shifts underneath the comparison.
"""

from __future__ import annotations

import pytest
import torch

from opentouch.pose_regression import POSE_DIM, PoseTransitionRegressor

B, T, H, W = 4, 6, 16, 16  # 16x16 is the taxel grid the encoder requires


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    pose = torch.randn(B, 21, 3, generator=g)
    tactile = torch.rand(B, T, H, W, generator=g)
    return pose, tactile


def _model(reduce_mode, seed=0, **kw):
    torch.manual_seed(seed)
    return PoseTransitionRegressor(
        use_tactile=True, tactile_emb_dim=16, hidden_dim=32,
        tactile_correction_input="tactile_only", tactile_reduce=reduce_mode, **kw
    ).eval()


def test_none_is_bit_identical_to_the_default():
    """The default must not change behaviour for any existing checkpoint."""
    pose, tactile = _inputs()
    explicit, implicit = _model("none"), _model("none")
    torch.manual_seed(0)
    default = PoseTransitionRegressor(
        use_tactile=True, tactile_emb_dim=16, hidden_dim=32,
        tactile_correction_input="tactile_only",
    ).eval()
    with torch.inference_mode():
        a = explicit(pose, tactile)
        b = implicit(pose, tactile)
        c = default(pose, tactile)
    assert torch.equal(a, b)
    assert torch.equal(a, c), "adding tactile_reduce changed default behaviour"


def test_scalar_preserves_per_frame_total_pressure():
    _, tactile = _inputs()
    m = _model("scalar")
    reduced = m._reduce_tactile(tactile)
    assert reduced.shape == tactile.shape
    torch.testing.assert_close(
        reduced.sum(dim=(-2, -1)), tactile.sum(dim=(-2, -1)), rtol=1e-5, atol=1e-6)


def test_scalar_destroys_all_spatial_structure():
    _, tactile = _inputs()
    reduced = _model("scalar")._reduce_tactile(tactile)
    # Every taxel within a frame is now the same value.
    per_frame = reduced.reshape(B, T, -1)
    assert float(per_frame.std(dim=-1).max()) < 1e-6
    # ...and the input genuinely had structure to destroy.
    assert float(tactile.reshape(B, T, -1).std(dim=-1).min()) > 1e-3


def test_scalar_keeps_temporal_structure():
    """Only spatial pattern is removed; frame-to-frame variation must survive,
    or the ablation would confound 'no spatial structure' with 'no signal'."""
    _, tactile = _inputs()
    reduced = _model("scalar")._reduce_tactile(tactile)
    per_frame_value = reduced[..., 0, 0]
    assert float(per_frame_value.std(dim=-1).min()) > 0


def test_scalar_changes_the_prediction():
    pose, tactile = _inputs()
    none_m, scalar_m = _model("none"), _model("scalar")
    # Force the gate open; at init it is zero, which would make both identical
    # for reasons unrelated to the ablation.
    with torch.no_grad():
        none_m.gate.fill_(1.0)
        scalar_m.gate.fill_(1.0)
    with torch.inference_mode():
        a, b = none_m(pose, tactile), scalar_m(pose, tactile)
    assert not torch.allclose(a, b), "scalar reduction had no effect on output"


def test_parameter_count_is_unchanged():
    """The comparison is only fair if capacity is identical."""
    n_none = sum(p.numel() for p in _model("none").parameters())
    n_scalar = sum(p.numel() for p in _model("scalar").parameters())
    assert n_none == n_scalar


def test_state_dicts_are_interchangeable():
    """A scalar-reduced run must be loadable into a 'none' model and vice
    versa, so checkpoints stay comparable across the ablation."""
    a, b = _model("none", seed=1), _model("scalar", seed=2)
    b.load_state_dict(a.state_dict(), strict=True)


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="tactile_reduce must be"):
        PoseTransitionRegressor(use_tactile=True, tactile_reduce="mean")


def test_reduce_is_inert_for_pose_only():
    """pose_only never receives tactile at all; the flag must not change it."""
    pose, _ = _inputs()
    torch.manual_seed(0)
    m = PoseTransitionRegressor(use_tactile=False, hidden_dim=32,
                                tactile_reduce="scalar").eval()
    torch.manual_seed(0)
    ref = PoseTransitionRegressor(use_tactile=False, hidden_dim=32).eval()
    with torch.inference_mode():
        assert torch.equal(m(pose), ref(pose))


def test_film_path_also_reduces():
    """Both encoder call sites must apply the ablation, not just the gate one.

    film's output layer is zero-initialised so the model starts identical to
    pose-only, exactly as the gate starts at zero. Both have to be opened
    before tactile can reach the output at all, or this would pass vacuously.
    """
    pose, tactile = _inputs()
    none_m = _model("none", fusion="film")
    scalar_m = _model("scalar", fusion="film")
    g = torch.Generator().manual_seed(3)
    w = torch.randn(none_m.film[-1].weight.shape, generator=g) * 0.1
    b_ = torch.randn(none_m.film[-1].bias.shape, generator=g) * 0.1
    for m in (none_m, scalar_m):
        with torch.no_grad():
            m.film[-1].weight.copy_(w)
            m.film[-1].bias.copy_(b_)
    with torch.inference_mode():
        a, b = none_m(pose, tactile), scalar_m(pose, tactile)
    assert not torch.allclose(a, b), "film path ignored tactile_reduce"


def test_output_shape_unchanged():
    pose, tactile = _inputs()
    with torch.inference_mode():
        out = _model("scalar")(pose, tactile)
    assert out.shape == (B, 21, 3)
    with torch.inference_mode():
        scalar_out = _model("scalar", output_dim=1)(pose, tactile)
    assert scalar_out.reshape(B, -1).shape == (B, 1)
    assert POSE_DIM == 63
