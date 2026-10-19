from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_paths(cfg: dict[str, Any], root: Path | None = None) -> dict[str, Path]:
    root = Path(root) if root is not None else Path(cfg.get("project_root", ".")).resolve()
    paths = {k: root / v for k, v in cfg["paths"].items()}
    paths["root"] = root
    return paths


def ensure_dirs(paths: dict[str, Path]) -> None:
    for key in ("processed", "interim", "checkpoints", "metrics", "figures"):
        if key in paths:
            paths[key].mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
