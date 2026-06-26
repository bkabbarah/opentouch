from __future__ import annotations

import torch
import torch.nn as nn
from .tactile_encoder import _normalize_input


class TactileDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, 2, 2)),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class TactileAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.decoder = TactileDecoder()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat, b, t = _normalize_input(x)
        features = self.encoder(flat)
        reconstruction = self.decoder(features)
        return reconstruction, flat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        flat, b, t = _normalize_input(x)
        return self.encoder(flat)