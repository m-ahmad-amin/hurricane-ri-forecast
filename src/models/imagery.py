from __future__ import annotations

import torch
import torch.nn as nn

from .tabular import TabularRIModel


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageryEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256, pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


class ImageryRIModel(nn.Module):
    def __init__(self, emb_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = ImageryEncoder(out_dim=emb_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, 1),
        )

    def forward(self, tabular: torch.Tensor | None, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(image)).squeeze(-1)


class FusionRIModel(nn.Module):
    def __init__(self, n_features: int, emb_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.image_encoder = ImageryEncoder(out_dim=emb_dim, dropout=dropout)
        self.tab_encoder = nn.Sequential(
            nn.Linear(n_features, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, 2),
        )
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, 1),
        )

    def forward(self, tabular: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        tab = self.tab_encoder(tabular)
        img = self.image_encoder(image)
        weights = torch.softmax(self.gate(torch.cat([tab, img], dim=1)), dim=1)
        fused = weights[:, :1] * tab + weights[:, 1:] * img
        return self.head(fused).squeeze(-1)


def build_model(name: str, n_features: int) -> nn.Module:
    name = name.lower()
    if name == "tabular":
        return TabularRIModel(n_features=n_features)
    if name == "imagery":
        return ImageryRIModel()
    if name == "fusion":
        return FusionRIModel(n_features=n_features)
    raise ValueError(f"unknown model: {name}")
