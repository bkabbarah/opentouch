from __future__ import annotations
import torch.nn.functional as F
import torch
import torch.nn as nn

_THUMB_MCP, _INDEX_MCP, _MIDDLE_MCP = 1, 5, 9
_NUM_JOINTS = 21
_COORD_DIM = 3
_INPUT_DIM = _NUM_JOINTS * _COORD_DIM
_VALID_NORMALIZE_MODES = {"none", "simple"}


class PoseEncoder(nn.Module):
    def __init__(self, emb_dim: int = 64, normalize_mode: str = "simple"):
        super().__init__()
        self.gru = nn.GRU(128, 120, num_layers=2, bidirectional=True, batch_first=True)
        if normalize_mode not in _VALID_NORMALIZE_MODES:
            raise ValueError("Invalid model configuration.")
        self.normalize_mode = normalize_mode
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
        self.projection = nn.Linear(240, emb_dim)  # replaces existing projection

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
        self.gru.flatten_parameters()
        _, h_n = self.gru(seq)
        combined = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # fwd last + bwd last
        return self.projection(F.relu(combined))

