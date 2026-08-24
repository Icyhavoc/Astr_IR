"""Compact residual CNN chosen to limit hallucination risk on a small FITS dataset."""

from __future__ import annotations

import torch
from torch import nn


class ResidualDnCNN(nn.Module):
    """Predict normalized noise and subtract it from the science-image channel."""

    def __init__(self, depth: int = 8, features: int = 32) -> None:
        super().__init__()
        if depth < 3 or features < 4:
            raise ValueError("ResidualDnCNN requires depth >= 3 and features >= 4")
        layers: list[nn.Module] = [
            nn.Conv2d(2, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(features, 1, kernel_size=3, padding=1))
        self.noise_predictor = nn.Sequential(*layers)
        self.depth = int(depth)
        self.features = int(features)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        if model_input.ndim != 4 or model_input.shape[1] != 2:
            raise ValueError("ResidualDnCNN input must have shape (batch, 2, height, width)")
        image = model_input[:, :1]
        predicted_noise = self.noise_predictor(model_input)
        return image - predicted_noise

    def predicted_noise(self, model_input: torch.Tensor) -> torch.Tensor:
        return self.noise_predictor(model_input)
