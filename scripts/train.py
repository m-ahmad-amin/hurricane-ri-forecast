from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed, make_loaders
from src.models.imagery import build_model
from src.train.loop import train_model
from src.utils import ensure_dirs, get_device, load_config, project_paths, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RI models")
    parser.add_argument("--config", default="configs/al062018.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--model", default=None, help="tabular, imagery, fusion, or all")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    ensure_dirs(paths)
    set_seed(cfg["train"]["seed"])

    tag = args.tag or Path(args.config).stem
    table, images, feature_cols, meta = load_processed(paths["processed"], tag)

    loaders, mean, std = make_loaders(
        table,
        images,
        feature_cols=feature_cols,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 0),
    )
    if "train" not in loaders:
        raise RuntimeError("no train split available")

    np.save(paths["processed"] / f"scaler_mean_{tag}.npy", mean)
    np.save(paths["processed"] / f"scaler_std_{tag}.npy", std)

    device = get_device()
    print(f"device={device} samples={meta}")

    model_names = cfg["train"]["models"]
    if args.model and args.model != "all":
        model_names = [args.model]

    results = {}
    for name in model_names:
        print(f"training {name}")
        model = build_model(name, n_features=len(feature_cols))
        ckpt = paths["checkpoints"] / f"{name}_{tag}.pt"
        result = train_model(
            model,
            loaders,
            device=device,
            epochs=cfg["train"]["epochs"],
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
            checkpoint_path=ckpt,
        )
        results[name] = result

    save_json(results, paths["metrics"] / f"train_{tag}.json")
    print(f"saved metrics to {paths['metrics'] / f'train_{tag}.json'}")


if __name__ == "__main__":
    main()
