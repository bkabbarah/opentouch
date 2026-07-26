"""Tests for opentouch.articulation_frames.

The load-bearing claim these tests defend is the one the direction probe's
interpretation now rests on: `wrist_translation_removed` (the codebase's
existing "articulation delta") still contains whole-hand ROTATION, and
`rigid_removed` does not. Several tests below construct motion that is
provably pure rigid rotation and assert exactly that asymmetry, so the
distinction is verified by construction rather than asserted in a docstring.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from opentouch.articulation_frames import (
    ARTICULATING_JOINTS,
    all_target_variants,
    decompose_rigid,
    hand_frame_basis,
    hand_frame_degenerate,
    kabsch_rotation,
    rigid_fraction,
    to_hand_frame,
    wrist_centered,
)
from opentouch.pose_regression import (
    COORD_DIM,
    NUM_KEYPOINTS,
    WRIST_INDEX,
    decompose_world_delta,
)

TOL = 1e-4


def random_hand(batch: int = 8, seed: int = 0) -> torch.Tensor:
    """A batch of plausible, non-degenerate hands: a fixed template with
    per-sample jitter, so the palm anchors never become collinear."""
    generator = torch.Generator().manual_seed(seed)
    template = torch.zeros(NUM_KEYPOINTS, COORD_DIM)
    for slot in range(5):
        # Fingers fan out along x, extend along y, with a little z spread so
        # the keypoint cloud is genuinely 3D rather than planar.
        base_x = (slot - 2) * 0.3
        for joint in range(4):
            idx = 1 + 4 * slot + joint
            template[idx] = torch.tensor(
                [base_x, 0.4 + 0.25 * joint, 0.05 * (slot - 2) + 0.02 * joint]
            )
    hands = template.unsqueeze(0).repeat(batch, 1, 1)
    hands = hands + 0.02 * torch.randn(hands.shape, generator=generator)
    # Give each sample a distinct wrist position so translation is nontrivial.
    hands = hands + torch.randn(batch, 1, COORD_DIM, generator=generator)
    return hands


def random_rotations(batch: int, seed: int = 1) -> torch.Tensor:
    """Proper rotations (det=+1) via QR of a random matrix."""
    generator = torch.Generator().manual_seed(seed)
    mats = torch.randn(batch, COORD_DIM, COORD_DIM, generator=generator)
    q, r = torch.linalg.qr(mats)
    # Fix QR's sign ambiguity, then force det=+1.
    q = q * torch.sign(torch.diagonal(r, dim1=1, dim2=2)).unsqueeze(1)
    flip = torch.where(torch.linalg.det(q) < 0, -1.0, 1.0)
    q[:, :, -1] = q[:, :, -1] * flip.unsqueeze(-1)
    return q


def small_rotations(batch: int, angle: float = 0.15, seed: int = 2) -> torch.Tensor:
    """Rotations of a fixed small angle about random axes -- closer to the
    frame-to-frame wrist re-orientation this module actually has to handle
    than a uniformly random rotation is."""
    generator = torch.Generator().manual_seed(seed)
    axes = torch.nn.functional.normalize(
        torch.randn(batch, COORD_DIM, generator=generator), dim=-1
    )
    # Rodrigues
    k = torch.zeros(batch, COORD_DIM, COORD_DIM)
    k[:, 0, 1], k[:, 0, 2] = -axes[:, 2], axes[:, 1]
    k[:, 1, 0], k[:, 1, 2] = axes[:, 2], -axes[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -axes[:, 1], axes[:, 0]
    eye = torch.eye(COORD_DIM).expand(batch, COORD_DIM, COORD_DIM)
    return eye + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


# --------------------------------------------------------------------------
# Kabsch
# --------------------------------------------------------------------------


def test_kabsch_recovers_a_known_rotation():
    source = wrist_centered(random_hand(batch=16, seed=3))
    rotation_true = random_rotations(16, seed=4)
    target = source @ rotation_true.transpose(1, 2)

    recovered = kabsch_rotation(source, target)
    assert torch.allclose(recovered, rotation_true, atol=TOL), (
        "Kabsch must recover the exact rotation when the target IS a rotated source"
    )


def test_kabsch_returns_proper_rotations_never_reflections():
    """A reflection would mirror the hand -- physically impossible, and it
    would silently absorb real articulation into the 'rigid' part."""
    generator = torch.Generator().manual_seed(5)
    source = wrist_centered(random_hand(batch=32, seed=6))
    # Deliberately adversarial: target is a REFLECTED source plus noise, so an
    # unrepaired SVD would happily return det=-1.
    reflect = torch.eye(COORD_DIM).expand(32, COORD_DIM, COORD_DIM).clone()
    reflect[:, 2, 2] = -1.0
    target = source @ reflect + 0.01 * torch.randn(source.shape, generator=generator)

    rotation = kabsch_rotation(source, target)
    dets = torch.linalg.det(rotation)
    assert torch.all(dets > 0.99), f"expected det=+1 proper rotations, got {dets}"
    assert torch.allclose(
        rotation @ rotation.transpose(1, 2),
        torch.eye(COORD_DIM).expand(32, COORD_DIM, COORD_DIM),
        atol=TOL,
    ), "Kabsch output must be orthonormal"


# --------------------------------------------------------------------------
# The central claim: rotation survives wrist-translation removal
# --------------------------------------------------------------------------


def test_pure_rigid_rotation_leaves_zero_residual():
    pose_t = random_hand(batch=16, seed=7)
    rotation = small_rotations(16, angle=0.2, seed=8)
    wrist = pose_t[:, WRIST_INDEX : WRIST_INDEX + 1, :]
    pose_future = (pose_t - wrist) @ rotation.transpose(1, 2) + wrist

    _, residual = decompose_rigid(pose_t, pose_future)
    assert residual.abs().max() < TOL, (
        f"pure rigid rotation must leave zero non-rigid residual, got max {residual.abs().max():.2e}"
    )


def test_pure_rigid_rotation_plus_translation_leaves_zero_residual():
    generator = torch.Generator().manual_seed(9)
    pose_t = random_hand(batch=16, seed=10)
    rotation = small_rotations(16, angle=0.25, seed=11)
    wrist = pose_t[:, WRIST_INDEX : WRIST_INDEX + 1, :]
    translation = torch.randn(16, 1, COORD_DIM, generator=generator)
    pose_future = (pose_t - wrist) @ rotation.transpose(1, 2) + wrist + translation

    _, residual = decompose_rigid(pose_t, pose_future)
    assert residual.abs().max() < TOL, (
        "rigid rotation + whole-hand translation must both be removed"
    )


def test_rotation_survives_wrist_translation_removal_but_not_rigid_removal():
    """THE headline test. Motion that is 100% rigid rotation shows up as large
    'articulation' under the codebase's current definition, and as zero under
    the rigid-removed definition. This is precisely why a probe trained
    against the current target may be decoding whole-hand re-orientation."""
    pose_t = random_hand(batch=64, seed=12)
    rotation = small_rotations(64, angle=0.2, seed=13)
    wrist = pose_t[:, WRIST_INDEX : WRIST_INDEX + 1, :]
    pose_future = (pose_t - wrist) @ rotation.transpose(1, 2) + wrist

    _, current_target = decompose_world_delta(pose_future - pose_t)
    variants = all_target_variants(pose_t, pose_future)

    np.testing.assert_allclose(
        variants["wrist_translation_removed"].numpy(), current_target.numpy(), atol=1e-6
    )
    current_energy = current_target[:, ARTICULATING_JOINTS].abs().mean()
    residual_energy = variants["rigid_removed"][:, ARTICULATING_JOINTS].abs().mean()

    assert current_energy > 1e-3, (
        "sanity: pure rotation must produce substantial motion under the CURRENT target"
    )
    assert residual_energy < TOL, (
        "the same pure-rotation motion must vanish under the rigid-removed target"
    )
    assert residual_energy < current_energy / 100, (
        f"rigid removal must kill essentially all of it: {residual_energy:.2e} vs {current_energy:.2e}"
    )


def test_rigid_fraction_is_one_for_pure_rotation_and_low_for_pure_articulation():
    pose_t = random_hand(batch=32, seed=14)
    wrist = pose_t[:, WRIST_INDEX : WRIST_INDEX + 1, :]

    rotation = small_rotations(32, angle=0.2, seed=15)
    rotated = (pose_t - wrist) @ rotation.transpose(1, 2) + wrist
    assert rigid_fraction(pose_t, rotated).min() > 0.999, (
        "pure rigid rotation must be ~100% rigid-explained"
    )

    # Pure articulation: curl only the fingertips, leave MCPs fixed, so no
    # single rotation of the whole hand can account for it.
    generator = torch.Generator().manual_seed(16)
    articulated = pose_t.clone()
    tips = [4, 8, 12, 16, 20]
    articulated[:, tips] += 0.1 * torch.randn(32, len(tips), COORD_DIM, generator=generator)
    assert rigid_fraction(pose_t, articulated).max() < 0.9, (
        "independent fingertip motion must NOT be mostly rigid-explained"
    )


def test_rigid_removed_wrist_row_is_exactly_zero():
    pose_t = random_hand(batch=8, seed=17)
    pose_future = random_hand(batch=8, seed=18)
    _, residual = decompose_rigid(pose_t, pose_future)
    assert residual[:, WRIST_INDEX].abs().max() == 0.0, (
        "wrist row must be identically zero, matching decompose_world_delta's contract"
    )


def test_rigid_removal_is_invariant_to_the_world_frame():
    """Rotating the whole scene must not change the non-rigid residual's
    magnitude -- if it did, the 'signal lives on world y' finding could be
    reproduced simply by choosing axes."""
    pose_t = random_hand(batch=16, seed=19)
    pose_future = random_hand(batch=16, seed=20)
    scene_rotation = random_rotations(16, seed=21)

    _, residual = decompose_rigid(pose_t, pose_future)
    _, residual_rotated = decompose_rigid(
        pose_t @ scene_rotation.transpose(1, 2), pose_future @ scene_rotation.transpose(1, 2)
    )
    norms = residual.flatten(1).norm(dim=1)
    norms_rotated = residual_rotated.flatten(1).norm(dim=1)
    assert torch.allclose(norms, norms_rotated, atol=TOL), (
        "non-rigid residual magnitude must be world-frame invariant"
    )


def test_hand_frame_target_is_invariant_to_the_world_frame():
    """The whole point of the palm frame: per-AXIS values, not just
    magnitudes, must be unchanged by an arbitrary choice of lab axes. The
    world-axis target has no such property, which is what makes a
    'signal only on world y' result hard to interpret."""
    pose_t = random_hand(batch=16, seed=22)
    pose_future = random_hand(batch=16, seed=23)
    scene_rotation = random_rotations(16, seed=24)

    plain = all_target_variants(pose_t, pose_future)
    rotated = all_target_variants(
        pose_t @ scene_rotation.transpose(1, 2), pose_future @ scene_rotation.transpose(1, 2)
    )
    assert torch.allclose(
        plain["rigid_removed_handframe"], rotated["rigid_removed_handframe"], atol=TOL
    ), "hand-frame components must be world-frame invariant, axis by axis"

    # And the contrast: world-axis components are NOT invariant.
    assert not torch.allclose(
        plain["wrist_translation_removed"], rotated["wrist_translation_removed"], atol=1e-2
    ), "sanity: world-axis components SHOULD change when the lab frame rotates"


# --------------------------------------------------------------------------
# Hand frame
# --------------------------------------------------------------------------


def test_hand_frame_basis_is_orthonormal_and_right_handed():
    basis = hand_frame_basis(random_hand(batch=32, seed=25))
    eye = torch.eye(COORD_DIM).expand(32, COORD_DIM, COORD_DIM)
    assert torch.allclose(basis @ basis.transpose(1, 2), eye, atol=TOL)
    assert torch.all(torch.linalg.det(basis) > 0.99), "basis must be right-handed"


def test_hand_frame_uses_only_the_pose_at_t():
    """No future information may enter the basis -- otherwise the change of
    frame would itself be a leak."""
    pose_t = random_hand(batch=16, seed=26)
    basis_a = hand_frame_basis(pose_t)
    basis_b = hand_frame_basis(pose_t.clone())
    assert torch.equal(basis_a, basis_b)
    # Perturbing the FUTURE pose cannot change the basis, because the basis
    # never sees it.
    variants_a = all_target_variants(pose_t, random_hand(batch=16, seed=27))
    variants_b = all_target_variants(pose_t, random_hand(batch=16, seed=28))
    assert torch.equal(hand_frame_basis(pose_t), basis_a)
    assert not torch.allclose(
        variants_a["rigid_removed_handframe"], variants_b["rigid_removed_handframe"]
    ), "sanity: different futures must give different targets"


def test_to_hand_frame_preserves_vector_norms():
    delta = torch.randn(16, NUM_KEYPOINTS, COORD_DIM)
    basis = hand_frame_basis(random_hand(batch=16, seed=29))
    transformed = to_hand_frame(delta, basis)
    assert torch.allclose(delta.norm(dim=-1), transformed.norm(dim=-1), atol=TOL), (
        "a change of orthonormal basis is a rotation; per-joint magnitudes must not change"
    )


def test_degenerate_hands_are_flagged_and_do_not_produce_nans():
    pose = random_hand(batch=4, seed=30)
    # Collapse one sample's palm anchors onto the wrist.
    pose[1, [5, 9, 17]] = pose[1, WRIST_INDEX].clone()
    flags = hand_frame_degenerate(pose)
    assert bool(flags[1]), "collapsed palm anchors must be flagged degenerate"
    assert not bool(flags[0]), "a normal hand must not be flagged"
    basis = hand_frame_basis(pose)
    assert torch.isfinite(basis).all(), "degenerate rows must fall back, not emit NaNs"


# --------------------------------------------------------------------------
# Contract / shape guards
# --------------------------------------------------------------------------


def test_all_target_variants_agrees_with_the_existing_codebase_definition():
    pose_t = random_hand(batch=16, seed=31)
    pose_future = random_hand(batch=16, seed=32)
    _, existing = decompose_world_delta(pose_future - pose_t)
    variants = all_target_variants(pose_t, pose_future)
    assert torch.allclose(variants["wrist_translation_removed"], existing, atol=1e-6), (
        "the 'wrist_translation_removed' variant must be BIT-COMPATIBLE with the target "
        "every existing result was computed against, or cross-variant comparison is meaningless"
    )


@pytest.mark.parametrize(
    "bad_shape", [(4, 20, 3), (4, 21, 2), (21, 3)]
)
def test_wrong_shapes_raise(bad_shape):
    with pytest.raises(ValueError):
        all_target_variants(torch.zeros(bad_shape), torch.zeros(bad_shape))


def test_batch_size_mismatch_raises():
    with pytest.raises(ValueError):
        decompose_rigid(random_hand(batch=4), random_hand(batch=5))
