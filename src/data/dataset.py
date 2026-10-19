from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .prepare import TABULAR_FEATURES


class RIDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        images: np.ndarray,
        feature_cols: list[str],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ):
        self.table = table.reset_index(drop=True)
        self.images = images
        self.feature_cols = feature_cols
        feats = self.table[feature_cols].to_numpy(dtype=np.float32)
        feats = np.where(np.isfinite(feats), feats, feature_mean)
        self.features = ((feats - feature_mean) / feature_std).astype(np.float32)
        self.labels = self.table["ri_label"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "tabular": torch.from_numpy(self.features[idx]),
            "image": torch.from_numpy(self.images[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def load_processed(processed_dir: Path, tag: str) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    processed_dir = Path(processed_dir)
    table = pd.read_parquet(processed_dir / f"samples_{tag}.parquet")
    images = np.load(processed_dir / f"images_{tag}.npy")
    feature_path = processed_dir / f"features_{tag}.json"
    if feature_path.exists():
        with open(feature_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        feature_cols = TABULAR_FEATURES
    with open(processed_dir / f"meta_{tag}.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return table, images, feature_cols, meta


def fit_feature_scaler(
    table: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train = table[table["split"] == "train"]
    arr = train[feature_cols].to_numpy(dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def make_loaders(
    table: pd.DataFrame,
    images: np.ndarray,
    feature_cols: list[str],
    batch_size: int,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader], np.ndarray, np.ndarray]:
    mean, std = fit_feature_scaler(table, feature_cols)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        mask = table["split"].to_numpy() == split
        if mask.sum() == 0:
            continue
        ds = RIDataset(
            table.loc[mask],
            images[mask],
            feature_cols=feature_cols,
            feature_mean=mean,
            feature_std=std,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders, mean, std
