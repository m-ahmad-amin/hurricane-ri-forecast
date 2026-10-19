from __future__ import annotations

import torch
import torch.nn as nn

from .tabular import TabularRIModel


class ImageryRIModel(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, tabular: torch.Tensor | None, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(image)).squeeze(-1)


class FusionRIModel(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.2):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
        )
        self.tab_encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, tabular: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.tab_encoder(tabular), self.image_encoder(image)], dim=1)
        return self.head(x).squeeze(-1)


def build_model(name: str, n_features: int) -> nn.Module:
    name = name.lower()
    if name == "tabular":
        return TabularRIModel(n_features=n_features)
    if name == "imagery":
        return ImageryRIModel()
    if name == "fusion":
        return FusionRIModel(n_features=n_features)
    raise ValueError(f"unknown model: {name}")
