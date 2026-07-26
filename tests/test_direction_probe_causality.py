"""By-construction leak tests for the direction probe's OWN encoding path.

The existing causality test (`tests/test_pose_regression.py::
test_causal_getitem_never_contains_a_future_frame_by_construction`) covers
`PoseTransitionDataset.__getitem__`. The probe never calls `__getitem__` --
it has its own gather in `encode_causal` / `encode_pose_causal` -- so until
now the code every published probe number actually ran through had no leak
test of its own, and the module docstring cited a verification report that is
not in the repository.

These tests close that gap the same way the dataset test does: poison every
frame after t with a value nothing else can produce, then assert it never
reaches the encoder; and separately assert the embedding is bit-identical
when only frames after t change. That second check is the stronger one --
it covers indirect paths such as a normalization statistic averaged over the
encoder's own input, which an input-inspection test alone would miss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tactile_direction_probe import encode_causal, encode_pose_causal  # noqa: E402
from opentouch.pose_encoder import PoseEncoder  # noqa: E402
from opentouch_train.regression_data import _causal_frame_indices  # noqa: E402

POISON = 1e6

N_WINDOWS = 6
SEQ_LEN = 24
CAUSAL_WINDOW = 8
VALID_T = 12


class SpyEncoder(torch.nn.Module):
    """Records the largest value it is ever handed, and returns a cheap
    deterministic function of its input so callers still get real embeddings."""

    def __init__(self, emb_dim: int = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seen = -float("inf")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.max_seen = max(self.max_seen, float(x.max()))
        flat = x.reshape(x.shape[0], -1)
        return flat[:, : self.emb_dim].clone()


def sample_index_arrays():
    window_idx = np.repeat(np.arange(N_WINDOWS), VALID_T)
    t_values = np.tile(np.arange(VALID_T), N_WINDOWS)
    causal_frame_idx = _causal_frame_indices(VALID_T, CAUSAL_WINDOW)
    return window_idx, t_values, causal_frame_idx


def tactile_tensor(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(N_WINDOWS, SEQ_LEN, 1, 16, 16, generator=generator)


def pose_tensor(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(N_WINDOWS, SEQ_LEN, 21, 3, generator=generator)


# --------------------------------------------------------------------------
# The frames after t never reach the encoder
# --------------------------------------------------------------------------


def test_encode_causal_never_assembles_a_frame_after_t():
    window_idx, t_values, causal_frame_idx = sample_index_arrays()
    device = torch.device("cpu")

    for t in range(VALID_T):
        poisoned = tactile_tensor()
        poisoned[:, t + 1 :] = POISON
        rows = t_values == t
        spy = SpyEncoder()
        encode_causal(
            spy, poisoned, window_idx[rows], t_values[rows],
            causal_frame_idx, batch_size=16, device=device,
        )
        assert spy.max_seen < POISON / 2, (
            f"a frame after t={t} was assembled into the tactile encoder's input"
        )


def test_encode_pose_causal_never_assembles_a_frame_after_t():
    window_idx, t_values, causal_frame_idx = sample_index_arrays()
    device = torch.device("cpu")

    for t in range(VALID_T):
        poisoned = pose_tensor()
        poisoned[:, t + 1 :] = POISON
        rows = t_values == t
        spy = SpyEncoder()
        encode_pose_causal(
            spy, poisoned, window_idx[rows], t_values[rows],
            causal_frame_idx, batch_size=16, device=device,
        )
        assert spy.max_seen < POISON / 2, (
            f"a frame after t={t} was assembled into the pose encoder's input"
        )


# --------------------------------------------------------------------------
# The embedding is invariant to the future -- the stronger property
# --------------------------------------------------------------------------


def test_tactile_embedding_is_invariant_to_frames_after_t():
    window_idx, t_values, causal_frame_idx = sample_index_arrays()
    device = torch.device("cpu")
    base = tactile_tensor(seed=1)
    spy = SpyEncoder()

    reference = encode_causal(
        spy, base, window_idx, t_values, causal_frame_idx, batch_size=32, device=device,
    )
    for t in range(VALID_T):
        perturbed = base.clone()
        perturbed[:, t + 1 :] = torch.rand_like(perturbed[:, t + 1 :]) * 50.0
        rows = t_values == t
        after = encode_causal(
            SpyEncoder(), perturbed, window_idx[rows], t_values[rows],
            causal_frame_idx, batch_size=32, device=device,
        )
        assert torch.equal(after, reference[rows]), (
            f"changing frames after t={t} changed the tactile embedding"
        )


def test_real_pose_encoder_embedding_is_invariant_to_frames_after_t():
    """Uses the ACTUAL PoseEncoder, not a spy, because `_normalize_pose`
    averages a scale statistic over its input's time axis. That is exactly the
    kind of indirect path an input-inspection test cannot see -- if the causal
    gather were wrong, the future would enter through the normalizer."""
    window_idx, t_values, causal_frame_idx = sample_index_arrays()
    device = torch.device("cpu")
    encoder = PoseEncoder(emb_dim=8).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    base = pose_tensor(seed=2)
    with torch.no_grad():
        reference = encode_pose_causal(
            encoder, base, window_idx, t_values, causal_frame_idx,
            batch_size=32, device=device,
        )
    for t in range(VALID_T):
        perturbed = base.clone()
        perturbed[:, t + 1 :] = perturbed[:, t + 1 :] * 100.0 + 7.0
        rows = t_values == t
        with torch.no_grad():
            after = encode_pose_causal(
                encoder, perturbed, window_idx[rows], t_values[rows],
                causal_frame_idx, batch_size=32, device=device,
            )
        assert torch.allclose(after, reference[rows], atol=1e-6), (
            f"changing frames after t={t} changed the pose embedding "
            "(likely via _normalize_pose's time-averaged scale)"
        )


# --------------------------------------------------------------------------
# The shuffled-control path obeys the same guarantee
# --------------------------------------------------------------------------


def test_deranged_window_path_is_also_causal():
    """The shuffled-tactile control reads the SAME t from a DIFFERENT window.
    If that path leaked, the control would be contaminated rather than the
    treatment -- a failure mode that would quietly weaken the null instead of
    faking a positive, and is therefore easy to miss."""
    window_idx, t_values, causal_frame_idx = sample_index_arrays()
    device = torch.device("cpu")
    permutation = np.array([(i + 3) % N_WINDOWS for i in range(N_WINDOWS)])
    assert not np.any(permutation == np.arange(N_WINDOWS))

    for t in range(VALID_T):
        poisoned = tactile_tensor(seed=3)
        poisoned[:, t + 1 :] = POISON
        rows = t_values == t
        spy = SpyEncoder()
        encode_causal(
            spy, poisoned, permutation[window_idx[rows]], t_values[rows],
            causal_frame_idx, batch_size=16, device=device,
        )
        assert spy.max_seen < POISON / 2, (
            f"the deranged-window path leaked a frame after t={t}"
        )


# --------------------------------------------------------------------------
# Sanity: the encoders do vary with the past, so the tests above are not
# passing merely because nothing is connected
# --------------------------------------------------------------------------


def test_embeddings_do_change_when_the_past_changes():
    """Guards against a vacuous suite: if `encode_causal` returned a constant,
    every invariance test above would pass for the wrong reason."""
    window_idx, t_values, causal_frame_idx = sample_index_arrays()
    device = torch.device("cpu")
    base = tactile_tensor(seed=4)

    reference = encode_causal(
        SpyEncoder(), base, window_idx, t_values, causal_frame_idx,
        batch_size=32, device=device,
    )
    perturbed = base.clone()
    perturbed[:, :3] += 5.0  # frames 0-2, inside every t's causal window for t>=2
    after = encode_causal(
        SpyEncoder(), perturbed, window_idx, t_values, causal_frame_idx,
        batch_size=32, device=device,
    )
    assert not torch.equal(after, reference), (
        "changing PAST frames must change the embedding"
    )


def test_different_t_within_a_window_give_different_embeddings():
    """The pre-fix bug was one embedding per window broadcast to every t. If
    that regressed, every invariance test would still pass."""
    window_idx, t_values, causal_frame_idx = sample_index_arrays()
    device = torch.device("cpu")
    embeddings = encode_causal(
        SpyEncoder(), tactile_tensor(seed=5), window_idx, t_values,
        causal_frame_idx, batch_size=32, device=device,
    )
    first_window = embeddings[window_idx == 0]
    distinct = {tuple(row.tolist()) for row in first_window}
    assert len(distinct) > 1, (
        "all t in a window produced the SAME embedding -- the pre-fix "
        "broadcast leak has regressed"
    )


@pytest.mark.parametrize("batch_size", [1, 7, 1024])
def test_batching_does_not_change_the_result(batch_size):
    """Row order and batch boundaries must not affect any embedding, or the
    (window, t) -> row alignment the probe relies on would be unstable."""
    window_idx, t_values, causal_frame_idx = sample_index_arrays()
    device = torch.device("cpu")
    tactile = tactile_tensor(seed=6)
    reference = encode_causal(
        SpyEncoder(), tactile, window_idx, t_values, causal_frame_idx,
        batch_size=13, device=device,
    )
    other = encode_causal(
        SpyEncoder(), tactile, window_idx, t_values, causal_frame_idx,
        batch_size=batch_size, device=device,
    )
    assert torch.equal(reference, other)
