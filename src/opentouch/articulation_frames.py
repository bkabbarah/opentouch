"""Reference frames for the pose-transition target.

WHY THIS EXISTS. `pose_regression.decompose_world_delta` produces what the
codebase calls the "articulation delta":

    articulation_delta = world_delta - wrist_delta

That subtracts the wrist's TRANSLATION and nothing else. It does NOT remove
whole-hand ROTATION. If the hand rotates rigidly about the wrist -- the
fingers perfectly still relative to the palm -- every non-wrist joint still
moves in the world frame, and all of that motion lands in
`articulation_delta`. So the quantity named "articulation" is really
"wrist-translation-removed world motion", which is rigid rotation PLUS true
finger articulation, in arbitrary world/lab axes.

This matters for the direction probe (scripts/tactile_direction_probe.py).
That probe found its entire above-chance signal on the world **y** axis, with
x and z at chance at every horizon. World axes carry no anatomical meaning,
and a signal confined to a single lab axis is exactly the fingerprint you
would expect from a global effect (a dominant vertical motion direction,
gravity, or a systematic wrist re-orientation) rather than from finger
flexion, which has no reason to prefer one world axis over another. The probe
result as it stands cannot distinguish those two explanations.

This module provides the decomposition that separates them, at three
successively stricter levels:

  1. `world_delta`                 -- raw p(t+k) - p(t).
  2. `wrist_translation_removed`   -- what the codebase already calls
                                      "articulation"; rigid rotation still in.
  3. `rigid_removed_*`             -- the true non-rigid residual: the best
                                      rigid rotation about the wrist is fit
                                      (Kabsch) and removed, leaving only
                                      motion the hand could not have produced
                                      by re-orienting as a solid object.

and, orthogonally, a change of basis:

  - world axes (x, y, z)  -- arbitrary lab frame, what the probe used.
  - hand axes (`hand_frame_basis`) -- a palm-anchored orthonormal frame built
    from the pose at time t, in which the axes mean something fixed relative
    to the hand: `long` (wrist->middle-MCP), `flex` (the direction fingers
    curl toward the palm), `normal` (palm normal).

SCOPE NOTE. Defining the target's coordinate frame is measurement, not an
architectural prior -- no model here is given anatomical structure, and
nothing in this module is used to constrain a network's parameters or
attention. It only changes what quantity is being predicted and reported.

All functions are batched, pure, and take/return torch tensors shaped
(B, NUM_KEYPOINTS, COORD_DIM); they never touch a dataset.
"""

from __future__ import annotations

from typing import Tuple

import torch

from opentouch.pose_regression import COORD_DIM, NUM_KEYPOINTS, WRIST_INDEX

# Palm-frame anchors, in the 21-keypoint layout documented in
# pose_regression.FINGERTIP_COLUMNS (0=wrist, then five 4-joint finger blocks
# MCP/PIP/DIP/TIP for thumb/index/middle/ring/pinky).
_INDEX_MCP = 1 + 4 * 1  # 5
_MIDDLE_MCP = 1 + 4 * 2  # 9
_PINKY_MCP = 1 + 4 * 4  # 17
assert (_INDEX_MCP, _MIDDLE_MCP, _PINKY_MCP) == (5, 9, 17)

# Joints that actually articulate: everything but the wrist. The wrist row is
# identically zero once wrist translation is removed, so it carries no
# direction and every consumer here drops it.
ARTICULATING_JOINTS = tuple(j for j in range(NUM_KEYPOINTS) if j != WRIST_INDEX)

AXIS_NAMES_WORLD = ("x", "y", "z")
AXIS_NAMES_HAND = ("long", "flex", "normal")


def _check_pose(name: str, pose: torch.Tensor) -> None:
    if pose.dim() != 3 or pose.shape[1:] != (NUM_KEYPOINTS, COORD_DIM):
        raise ValueError(
            f"{name} must be shaped (B,{NUM_KEYPOINTS},{COORD_DIM}), got {tuple(pose.shape)}"
        )


def wrist_centered(pose: torch.Tensor) -> torch.Tensor:
    """(B,21,3) -> (B,21,3) with the wrist moved to the origin."""
    _check_pose("pose", pose)
    return pose - pose[:, WRIST_INDEX : WRIST_INDEX + 1, :]


def kabsch_rotation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Best-fit rotation R (B,3,3) taking `source` onto `target`, both
    (B,21,3) and both assumed already wrist-centered, minimizing
    ||source @ R^T - target||_F over proper rotations.

    Proper rotations only: the determinant is repaired via the standard
    sign-flip on the last singular vector, so a noisy hand can never be
    "explained" by a reflection (which would mirror the hand -- physically
    impossible and would silently absorb real articulation).
    """
    _check_pose("source", source)
    _check_pose("target", target)
    if source.shape[0] != target.shape[0]:
        raise ValueError(
            f"source and target must have the same batch size, got {source.shape[0]} and {target.shape[0]}"
        )

    # Cross-covariance H = source^T @ target, (B,3,3)
    covariance = source.transpose(1, 2) @ target
    # float64 for the SVD: hand keypoints are near-degenerate (close to
    # planar) often enough that float32 gives visibly unstable singular
    # vectors, and this runs once per sample, not in a training loop.
    u, _, vh = torch.linalg.svd(covariance.double())
    det_sign = torch.linalg.det(vh.transpose(1, 2) @ u.transpose(1, 2))
    flip = torch.ones_like(u)
    flip[:, :, -1] = det_sign.unsqueeze(-1)
    rotation = vh.transpose(1, 2) @ (u * flip).transpose(1, 2)
    return rotation.to(source.dtype)


def decompose_rigid(
    pose_t: torch.Tensor, pose_future: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(B,21,3), (B,21,3) -> (rotation (B,3,3), residual_delta (B,21,3)).

    `rotation` is the best rigid rotation about the wrist taking the hand's
    shape at t onto its shape at t+k. `residual_delta` is what that rotation
    could NOT explain, expressed in the hand's own frame at time t:

        residual = (future_centered @ R) - pose_t_centered

    i.e. the future pose is rotated BACK into t's orientation and compared to
    the pose at t. A hand that only translated and re-oriented rigidly gives
    residual == 0 (up to numerical error) at every joint; anything nonzero is
    genuine change in finger configuration.

    The wrist row is exactly zero by construction (both inputs are
    wrist-centered, and rotation fixes the origin).
    """
    _check_pose("pose_t", pose_t)
    _check_pose("pose_future", pose_future)
    source = wrist_centered(pose_t)
    target = wrist_centered(pose_future)
    rotation = kabsch_rotation(source, target)
    # target @ R un-rotates the future pose back into t's frame, since R maps
    # source onto target (source @ R^T ~ target), so target @ R ~ source.
    residual = target @ rotation - source
    return rotation, residual


def hand_frame_basis(pose: torch.Tensor) -> torch.Tensor:
    """(B,21,3) -> (B,3,3) orthonormal palm-anchored basis, ROWS are the axes
    in the order (`long`, `flex`, `normal`):

      long   = normalize(middle_MCP - wrist)          palm's long axis
      normal = normalize(long x (index_MCP - pinky_MCP))   palm normal
      flex   = normal x long                          fingers curl toward this

    Built from the pose at a single time step (call it with the pose at t),
    so re-expressing a delta in this basis is a rotation of the target only --
    it uses no future information whatsoever.

    Degenerate hands (a collapsed or exactly-planar keypoint set, where the
    cross product vanishes) would give a non-orthonormal basis; those rows are
    detected and replaced with the world identity basis so downstream code
    gets a valid rotation rather than NaNs. `hand_frame_degenerate` reports
    which rows were affected so callers can exclude them.
    """
    _check_pose("pose", pose)
    centered = wrist_centered(pose)
    long_axis = centered[:, _MIDDLE_MCP, :]
    transverse = centered[:, _INDEX_MCP, :] - centered[:, _PINKY_MCP, :]

    long_n = torch.nn.functional.normalize(long_axis, dim=-1, eps=1e-12)
    normal_n = torch.nn.functional.normalize(
        torch.cross(long_n, transverse, dim=-1), dim=-1, eps=1e-12
    )
    flex_n = torch.cross(normal_n, long_n, dim=-1)

    basis = torch.stack([long_n, flex_n, normal_n], dim=1)  # (B,3,3), rows = axes

    degenerate = hand_frame_degenerate(pose)
    if degenerate.any():
        identity = torch.eye(COORD_DIM, dtype=basis.dtype, device=basis.device)
        basis = torch.where(degenerate[:, None, None], identity.expand_as(basis), basis)
    return basis


def hand_frame_degenerate(pose: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """(B,21,3) -> (B,) bool: True where the palm anchors are too collinear or
    too collapsed to define a stable frame. Callers should drop these samples
    rather than trust a hand-frame target for them."""
    _check_pose("pose", pose)
    centered = wrist_centered(pose)
    long_axis = centered[:, _MIDDLE_MCP, :]
    transverse = centered[:, _INDEX_MCP, :] - centered[:, _PINKY_MCP, :]
    cross_norm = torch.linalg.norm(torch.cross(long_axis, transverse, dim=-1), dim=-1)
    long_norm = torch.linalg.norm(long_axis, dim=-1)
    transverse_norm = torch.linalg.norm(transverse, dim=-1)
    return (
        (long_norm < tol)
        | (transverse_norm < tol)
        | (cross_norm < tol * torch.clamp(long_norm * transverse_norm, min=tol))
    )


def to_hand_frame(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """(B,21,3) delta + (B,3,3) row-basis -> (B,21,3) delta in hand axes.

    Component i of the output is the projection of the delta onto row i of
    the basis, so output[..., 0] is motion along `long`, [..., 1] along
    `flex`, [..., 2] along `normal`.
    """
    _check_pose("delta", delta)
    if basis.shape[1:] != (COORD_DIM, COORD_DIM):
        raise ValueError(f"basis must be shaped (B,3,3), got {tuple(basis.shape)}")
    if delta.shape[0] != basis.shape[0]:
        raise ValueError(
            f"delta and basis must have the same batch size, got {delta.shape[0]} and {basis.shape[0]}"
        )
    return delta @ basis.transpose(1, 2)


def all_target_variants(
    pose_t: torch.Tensor, pose_future: torch.Tensor
) -> dict[str, torch.Tensor]:
    """One call -> every target variant the probe should be run against, so a
    caller cannot accidentally compare across inconsistent definitions.

    Keys, from loosest to strictest:
      world                    p(t+k) - p(t), raw world axes.
      wrist_translation_removed
                               the codebase's current "articulation_delta"
                               (pose_regression.decompose_world_delta), world
                               axes, rigid rotation STILL PRESENT.
      wrist_translation_removed_handframe
                               the same quantity re-expressed in palm axes.
      rigid_removed            non-rigid residual only, in t's world axes.
      rigid_removed_handframe  non-rigid residual only, in palm axes. This is
                               the strictest and the most anatomically
                               interpretable: a nonzero `flex` component here
                               is finger flexion and cannot be whole-hand
                               motion of any kind.

    Comparing probe AUC across these isolates what the signal actually is: a
    signal present in `wrist_translation_removed` but absent in
    `rigid_removed` was whole-hand rotation, not articulation.
    """
    _check_pose("pose_t", pose_t)
    _check_pose("pose_future", pose_future)

    world = pose_future - pose_t
    wrist_removed = world - world[:, WRIST_INDEX : WRIST_INDEX + 1, :]
    _, rigid_removed = decompose_rigid(pose_t, pose_future)
    basis = hand_frame_basis(pose_t)

    return {
        "world": world,
        "wrist_translation_removed": wrist_removed,
        "wrist_translation_removed_handframe": to_hand_frame(wrist_removed, basis),
        "rigid_removed": rigid_removed,
        "rigid_removed_handframe": to_hand_frame(rigid_removed, basis),
    }


def rigid_fraction(pose_t: torch.Tensor, pose_future: torch.Tensor) -> torch.Tensor:
    """(B,) in [0,1]: the share of wrist-translation-removed motion energy
    that the best rigid rotation explains, per sample.

        1 - ||rigid_removed||^2 / ||wrist_translation_removed||^2

    This is the single number that says how much of the codebase's
    "articulation delta" was never articulation. Samples with no motion at
    all return 0 (nothing to explain) rather than NaN.
    """
    variants = all_target_variants(pose_t, pose_future)
    total = variants["wrist_translation_removed"].flatten(1).pow(2).sum(dim=1)
    residual = variants["rigid_removed"].flatten(1).pow(2).sum(dim=1)
    explained = torch.where(
        total > 0, 1.0 - residual / torch.clamp(total, min=torch.finfo(total.dtype).tiny), torch.zeros_like(total)
    )
    return explained.clamp(0.0, 1.0)
