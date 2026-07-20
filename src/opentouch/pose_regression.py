"""Pose-transition regression: does tactile signal predict how the hand is
about to move, beyond what the current pose already implies?

Scientific design:
  - Target is the pose DELTA at t+k, not the absolute pose at t+k. Absolute
    targets let a copy-the-current-pose predictor score well for free, since
    pose is heavily autocorrelated at short horizons; delta targets force
    that trivial baseline to predict exactly zero, so any improvement has to
    come from actually modeling motion (see PoseTransitionRegressor and
    opentouch.regression_metrics.compute_copy_baseline_metrics).
  - The tactile-contact-encoder direction (tactile_contact_encoder.py) is
    dead: its shuffled-anatomy control matched the real anatomical prior. So
    this uses the plain CNN+biGRU tactile encoder (tactile_encoder.py,
    T->P retrieval mAP 45.46), not the contact encoder.
  - One class with a use_tactile switch, not two divergent copies, so the
    pose-only baseline and the tactile+pose model share the exact same
    `head` module and training path.

FUSION -- RESIDUAL, NOT CONCATENATION:
  An earlier version concatenated pose_flat (63d) with the tactile embedding
  (64d) and fed both into one BatchNorm1d(input_dim) + MLP head. That is a
  fusion artifact, not evidence about tactile: a ~500k-param tactile encoder
  output enters the head as an equal partner to the 63-dim pose vector, the
  joint BatchNorm couples their scales, and the small pose signal that
  actually predicts articulation gets swamped. Measured effect at k=16:
  pose-only val loss 0.002761 vs. concat-fused tactile+pose 0.008024 -- 3x
  WORSE than pose-only, from fusion, not from tactile content.

  The model instead predicts:
      output = delta_pose + gate * delta_correction
  where:
    - delta_pose = self.head(pose_flat) -- the exact same module, with the
      exact same architecture (BatchNorm1d(63) -> Linear -> GELU -> Dropout
      -> Linear -> GELU -> Dropout -> Linear), that the pose-only baseline
      uses. This path is always computed, in both modes, from pose alone.
    - delta_correction = self.tactile_head([pose_flat, tactile_embed]) --
      a SEPARATE head with its own BatchNorm1d(63+tactile_emb_dim), only
      constructed when use_tactile=True. No BatchNorm is shared between the
      two branches, so tactile can never rescale the pose path's inputs.
    - gate is a learnable scalar initialized to ZERO. At init, gate=0 makes
      the tactile+pose model IDENTICAL to the pose-only baseline (see
      test_residual_gate_zero_matches_pose_only_head); tactile can only earn
      its way in via gradient descent, never corrupt the pose signal by
      construction. A gate that trains to ~0 is therefore itself a valid
      null result -- logged at eval, not silently discarded.

TARGET DECOMPOSITION -- world_delta vs. articulation_delta:
  Pose coordinates in this dataset are WORLD-SPACE (coordinate range spans
  ~2.2 units while a hand is only ~0.19 across). Measured on the val split,
  decomposing the raw delta pose[t+k]-pose[t] into the wrist's own
  translation vs. wrist-relative articulation:

    k=1: full tip 0.00811 | wrist translation 0.00591 | articulation tip 0.00355 | 73% translation
    k=2: full tip 0.01814 | wrist translation 0.01355 | articulation tip 0.00793 | 75% translation
    k=4: full tip 0.03496 | wrist translation 0.02643 | articulation tip 0.01521 | 76% translation
    k=8: full tip 0.06473 | wrist translation 0.04940 | articulation tip 0.02734 | 76% translation

  ~76% of the raw ("world_delta") displacement is the whole hand translating
  through space -- arm motion. Tactile cannot predict where an arm is going
  and there is no reason it should; training on world_delta asks tactile to
  predict mostly-unpredictable translation and would produce a null result
  that says nothing about the actual scientific question. --target-mode
  defaults to "articulation_delta" (see decompose_world_delta below); both
  target spaces are always reported at eval regardless of which was trained
  on (opentouch.regression_metrics.compute_dual_target_metrics), so the
  wrist-translation-dominated view is never silently thrown away either.

SENSOR NOISE FLOOR (interpret k=1 accordingly, do not "fix" it in code):
  Pose comes from a Rokoko Smartglove with ~1 degree rotational accuracy, so
  a fingertip ~10cm from the wrist carries roughly 0.0017 of positional
  uncertainty from sensor noise alone. At k=1 the median articulation
  displacement is 0.00355 -- only ~2x the noise floor. k=1 is therefore
  close to unmeasurable and should be read as a noise check (does the
  pipeline behave sanely near the floor?) rather than a real motion-
  prediction horizon; k=2/4/8 have progressively more headroom above noise.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .pose_encoder import _INDEX_MCP, _MIDDLE_MCP, _THUMB_MCP
from .tactile_encoder import CNNetEmbedding

NUM_KEYPOINTS = 21
COORD_DIM = 3
POSE_DIM = NUM_KEYPOINTS * COORD_DIM  # 63

# VERIFY, not assumed: pose_encoder.py's 21-keypoint layout is wrist-first
# with 5 four-joint finger blocks (MCP/PIP/DIP/TIP) starting at column 1 --
# _THUMB_MCP/_INDEX_MCP/_MIDDLE_MCP == 1,5,9 == 1+4*slot for slot=0,1,2 is
# only consistent with a layout where columns 1..20 are entirely covered by
# the 5 finger blocks, leaving column 0 as the one keypoint outside any
# finger block. In a hand skeleton that unclaimed keypoint is the wrist --
# corroborated independently by pose_encoder.py's own PoseEncoder._normalize_pose,
# which centers every pose on keypoint index 0 (x - x[:, :, 0:1, :]), i.e.
# the codebase already treats index 0 as the hand's reference/root point.
assert _THUMB_MCP == 1 + 4 * 0 == 1, "pose_encoder.py's keypoint layout changed; WRIST_INDEX=0 assumption is stale"
assert _INDEX_MCP == 1 + 4 * 1 == 5, "pose_encoder.py's keypoint layout changed; WRIST_INDEX=0 assumption is stale"
assert _MIDDLE_MCP == 1 + 4 * 2 == 9, "pose_encoder.py's keypoint layout changed; WRIST_INDEX=0 assumption is stale"
WRIST_INDEX = 0

# Fingertip columns in the 21-keypoint layout: 0=wrist; for finger slot i in
# [thumb,index,middle,ring,pinky], columns 1+4i/2+4i/3+4i/4+4i =
# MCP/PIP/DIP/TIP. This is the same convention documented in
# src/opentouch/pose_encoder.py (_THUMB_MCP/_INDEX_MCP/_MIDDLE_MCP = 1,5,9)
# and derived identically in contact_structure.py's
# _skinning_to_kp21_map docstring and build_skinning_B's tip_kp21_cols --
# the single source of truth for this layout, not an independently guessed
# index list.
FINGERTIP_COLUMNS = tuple(1 + 4 * slot + 3 for slot in range(5))  # (4, 8, 12, 16, 20)
assert FINGERTIP_COLUMNS == (4, 8, 12, 16, 20)


def decompose_world_delta(world_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(B,21,3) world-space delta -> (wrist_delta (B,3), articulation_delta (B,21,3)).

    articulation_delta = world_delta - wrist_delta broadcast to all 21
    joints -- wrist-relative motion with the whole-hand translation removed.
    The wrist row of articulation_delta is therefore EXACTLY zero by
    construction (see test_articulation_delta_wrist_row_is_zero).
    """
    if world_delta.dim() != 3 or world_delta.shape[1:] != (NUM_KEYPOINTS, COORD_DIM):
        raise ValueError(
            f"world_delta must be shaped (B,{NUM_KEYPOINTS},{COORD_DIM}), got {tuple(world_delta.shape)}"
        )
    wrist_delta = world_delta[:, WRIST_INDEX, :]  # (B,3)
    articulation_delta = world_delta - wrist_delta.unsqueeze(1)
    return wrist_delta, articulation_delta


class PoseTransitionRegressor(nn.Module):
    """Predicts the pose delta at t+k from the pose at t, optionally with a
    RESIDUAL correction from a whole-T=20-frame-window tactile embedding
    (see module docstring, "FUSION -- RESIDUAL, NOT CONCATENATION").

    forward(pose_t, tactile_pressure=None) -> (B, 21, 3) predicted delta.
    output = self.head(pose_flat) [+ self.gate * self.tactile_head(...) if
    use_tactile].

    use_tactile=False is the pose-only baseline this task exists to beat. It
    must never receive a tactile tensor: this is enforced with a runtime
    assert, not by silently ignoring or zeroing a tactile input, so it is
    structurally impossible -- not just a matter of the caller remembering
    -- to leak tactile information into the baseline that tactile has to
    outperform for the tactile-predicts-transitions claim to mean anything.

    self.head is constructed IDENTICALLY (same BatchNorm1d(63), same layer
    sizes, same nonlinearity) regardless of use_tactile, and is the only
    thing that produces the pose-path delta in both modes -- so
    use_tactile=False is not merely similar to but numerically the same
    computation as the pose-only baseline this replaces (see
    test_residual_gate_zero_matches_pose_only_head).
    """

    def __init__(
        self,
        use_tactile: bool = True,
        tactile_emb_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.use_tactile = use_tactile
        self.tactile_emb_dim = tactile_emb_dim
        self.hidden_dim = hidden_dim

        # Keep it simple: a small MLP head, not a new deep backbone -- the
        # tactile side already has its own encoder (CNNetEmbedding), and the
        # OpenTouch ablations show deep backbones underperform on this data.
        # This exact module, with this exact input_dim, is what the
        # pose-only baseline uses -- it must not change shape based on
        # use_tactile, or use_tactile=False would stop being numerically
        # identical to that baseline.
        self.head = nn.Sequential(
            nn.BatchNorm1d(POSE_DIM),
            nn.Linear(POSE_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, POSE_DIM),
        )

        if use_tactile:
            self.tactile_encoder = CNNetEmbedding(emb_dim=tactile_emb_dim)
            # Correction head sees pose_flat too (so the correction can be
            # conditioned on the current articulation, not tactile alone),
            # but through its OWN BatchNorm over its OWN input_dim -- never
            # the pose-only head's BatchNorm(63), so tactile can never
            # rescale the pose path's inputs (see module docstring).
            tactile_head_input_dim = POSE_DIM + tactile_emb_dim
            self.tactile_head = nn.Sequential(
                nn.BatchNorm1d(tactile_head_input_dim),
                nn.Linear(tactile_head_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, POSE_DIM),
            )
            # Scalar gate, zero-initialized: at init this model computes
            # exactly self.head(pose_flat), i.e. the pose-only baseline.
            # Gradient descent is the only way tactile's correction can
            # start contributing -- it cannot corrupt the pose path by
            # construction. Logged at eval; a gate that stays ~0 means
            # tactile carries no transition signal beyond what pose implies.
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.tactile_encoder = None
            self.tactile_head = None
            self.gate = None

    def forward(
        self,
        pose_t: torch.Tensor,
        tactile_pressure: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pose_t.dim() != 3 or pose_t.shape[1:] != (NUM_KEYPOINTS, COORD_DIM):
            raise ValueError(
                f"pose_t must be shaped (B,{NUM_KEYPOINTS},{COORD_DIM}), got {tuple(pose_t.shape)}"
            )
        b = pose_t.shape[0]
        pose_flat = pose_t.reshape(b, POSE_DIM)
        delta_pose = self.head(pose_flat).view(b, NUM_KEYPOINTS, COORD_DIM)

        if self.use_tactile:
            assert tactile_pressure is not None, (
                "use_tactile=True requires a tactile_pressure tensor, got None"
            )
            tactile_embed = self.tactile_encoder(tactile_pressure)
            correction_input = torch.cat([pose_flat, tactile_embed], dim=-1)
            delta_correction = self.tactile_head(correction_input).view(b, NUM_KEYPOINTS, COORD_DIM)
            return delta_pose + self.gate * delta_correction
        else:
            assert tactile_pressure is None, (
                "pose-only mode (use_tactile=False) must not receive a tactile "
                "tensor at all -- this baseline exists specifically to measure "
                "what pose alone can predict; tactile information must be "
                "structurally impossible to leak in, not merely unused"
            )
            return delta_pose
