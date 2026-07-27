from __future__ import annotations
import torch.nn.functional as F
import torch
import torch.nn as nn

_THUMB_MCP, _INDEX_MCP, _MIDDLE_MCP = 1, 5, 9
_NUM_JOINTS = 21
_COORD_DIM = 3
_INPUT_DIM = _NUM_JOINTS * _COORD_DIM
_VALID_NORMALIZE_MODES = {"none", "simple"}
_VALID_TEMPORAL_MODES = {"gru", "mean"}


class PoseEncoder(nn.Module):
    """Per-frame MLP over 21 hand keypoints, then a temporal aggregation.

    `temporal_mode` selects the aggregation, and is the single knob for the
    project's headline ablation:

      "gru"  (default) -- 2-layer bidirectional GRU, readout = concat of the
             final forward and final backward hidden states (240 dims).
      "mean" -- temporal average pooling, the original OpenTouch baseline.
             Projects the pooled 128-dim vector directly, matching the
             upstream architecture exactly, so historical avg-pool
             checkpoints load into this class unchanged, and reproduce their
             published numbers. Two things had to match for that: the
             projection width (128, not a padded 240) and the ABSENCE of the
             ReLU that arrived alongside the GRU. Getting either wrong scores
             an upstream checkpoint at roughly 10 mAP instead of 16.8.

    Having both in one class matters for provenance: the avg-pool control was
    previously only runnable from a different repository, which made the
    published comparison a cross-codebase one. Now a single flag switches
    them and every other component is bit-identical.
    """

    def __init__(
        self,
        emb_dim: int = 64,
        normalize_mode: str = "simple",
        temporal_mode: str = "gru",
    ):
        super().__init__()
        if normalize_mode not in _VALID_NORMALIZE_MODES:
            raise ValueError("Invalid model configuration.")
        if temporal_mode not in _VALID_TEMPORAL_MODES:
            raise ValueError(
                f"temporal_mode must be one of {sorted(_VALID_TEMPORAL_MODES)}, got {temporal_mode!r}"
            )
        self.normalize_mode = normalize_mode
        self.temporal_mode = temporal_mode
        # Only built for "gru": leaving it out of "mean" keeps the baseline's
        # parameter count honest and makes a state_dict from one mode fail
        # loudly rather than silently load into the other.
        self.gru = (
            nn.GRU(128, 120, num_layers=2, bidirectional=True, batch_first=True)
            if temporal_mode == "gru"
            else None
        )
        self.encoder = nn.Sequential(
            nn.Linear(_INPUT_DIM, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
        )
        # 240 = concat of the biGRU's final forward and backward hidden
        # states (2 x 120); 128 = the per-frame encoder width, pooled.
        self.projection = nn.Linear(240 if temporal_mode == "gru" else 128, emb_dim)

    @torch.no_grad()
    def _normalize_pose(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - x[:, :, 0:1, :]
        d1 = torch.norm(x_centered[:, :, _INDEX_MCP] - x_centered[:, :, _THUMB_MCP], dim=-1, keepdim=True)
        d2 = torch.norm(x_centered[:, :, _MIDDLE_MCP] - x_centered[:, :, _THUMB_MCP], dim=-1, keepdim=True)
        scale = (0.5 * (d1 + d2)).mean(dim=1, keepdim=True).clamp_min(1e-6)
        return x_centered / scale.unsqueeze(-1)

    def _prepare_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        if landmarks.dim() == 5:
            landmarks = landmarks.squeeze(2)
        expected_shape = (_NUM_JOINTS, _COORD_DIM)
        if landmarks.dim() != 4 or landmarks.shape[2:] != expected_shape:
            raise ValueError("Invalid modality input.")
        return landmarks

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Encode hand landmarks to normalized embeddings."""
        landmarks = self._prepare_landmarks(landmarks)
        b, t = landmarks.shape[:2]
        if self.normalize_mode == "simple":
            landmarks = self._normalize_pose(landmarks)
        encoded = self.encoder(landmarks.reshape(b * t, _INPUT_DIM))
        seq = encoded.view(b, t, -1)
        if self.temporal_mode == "gru":
            self.gru.flatten_parameters()
            _, h_n = self.gru(seq)
            combined = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # fwd last + bwd last
            # The ReLU arrived with the GRU (commit 73dc799) and was never part
            # of the avg-pool encoder, so it stays inside this branch.
            return self.projection(F.relu(combined))

        # Average pooling discards temporal order entirely -- this is the
        # baseline the GRU is measured against. No ReLU: upstream projects the
        # pooled vector directly, and applying one here both diverges from the
        # published architecture and handicaps the baseline by zeroing every
        # negative pooled feature.
        return self.projection(seq.mean(dim=1))

