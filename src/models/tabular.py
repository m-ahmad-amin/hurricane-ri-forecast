from __future__ import annotations

import torch
import torch.nn as nn


class TabularRIModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, tabular: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(tabular).squeeze(-1)
