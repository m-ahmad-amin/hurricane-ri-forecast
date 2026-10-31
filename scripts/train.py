from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import fit_feature_scaler, load_processed, make_loaders
from src.data.prepare import TABULAR_FEATURES
from src.models.imagery import build_model
from src.train.loop import train_model
from src.utils import ensure_dirs, get_device, load_config, project_paths, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RI models")
    parser.add_argument("--config", default="configs/al062018.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--model", default=None, help="tabular, imagery, fusion, or all")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override train.seed; checkpoints saved as {model}_{tag}_seed{N}.pt",
    )
    parser.add_argument(
        "--no-pos-weight",
        action="store_true",
        help="use unweighted BCEWithLogitsLoss (default: positive-class weight n_neg/n_pos)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="optional checkpoint/metrics suffix, e.g. noweight -> tabular_{tag}_noweight_seedN.pt",
    )
    parser.add_argument(
        "--tabular-only",
        action="store_true",
        help="skip loading images_*.npy; use zero image tensors (tabular pathway only)",
    )
    parser.add_argument(
        "--images-path",
        default=os.environ.get("IMAGE_NPY"),
        help="optional path to images_*.npy (e.g. local hydrated copy); default processed dir",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    ensure_dirs(paths)
    seed = int(args.seed if args.seed is not None else cfg["train"]["seed"])
    set_seed(seed)

    tag = args.tag or Path(args.config).stem
    use_pos_weight = not args.no_pos_weight
    suffix = args.suffix.strip("_")
    name_tag = f"{tag}_{suffix}" if suffix else tag

    if args.tabular_only:
        table = pd.read_parquet(paths["processed"] / f"samples_{tag}.parquet")
        feat_path = paths["processed"] / f"features_{tag}.json"
        if feat_path.exists():
            with open(feat_path, "r", encoding="utf-8") as f:
                feature_cols = json.load(f)
        else:
            feature_cols = [c for c in TABULAR_FEATURES if c in table.columns]
        images = None
        meta = {"n": len(table), "tabular_only": True}
        print("tabular-only mode: on-the-fly zero image tensors")
    else:
        table, images, feature_cols, meta = load_processed(paths["processed"], tag)
        if args.images_path:
            img_path = Path(args.images_path)
            print(f"loading images from {img_path}")
            images = np.load(img_path, mmap_mode="r")

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
    print(f"device={device} seed={seed} samples={meta}")
    print(
        f"train config: batch={cfg['train']['batch_size']} epochs={cfg['train']['epochs']} "
        f"lr={cfg['train']['lr']} patience={cfg['train'].get('patience', 8)} "
        f"pos_weight={'on' if use_pos_weight else 'off'} name_tag={name_tag}"
    )
    if args.tabular_only and args.model not in (None, "tabular"):
        raise SystemExit("--tabular-only requires --model tabular")

    model_names = cfg["train"]["models"]
    if args.model and args.model != "all":
        model_names = [args.model]

    results = {}
    for name in model_names:
        print(f"\n=== training {name} (seed={seed}, pos_weight={use_pos_weight}) ===")
        model = build_model(name, n_features=len(feature_cols))
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"parameters: {n_params:,}")
        if args.seed is not None or suffix:
            ckpt = paths["checkpoints"] / f"{name}_{name_tag}_seed{seed}.pt"
        else:
            ckpt = paths["checkpoints"] / f"{name}_{tag}.pt"
        result = train_model(
            model,
            loaders,
            device=device,
            epochs=cfg["train"]["epochs"],
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
            checkpoint_path=ckpt,
            patience=cfg["train"].get("patience", 8),
            min_epochs=cfg["train"].get("min_epochs", 5),
            use_pos_weight=use_pos_weight,
        )
        result["seed"] = seed
        results[name] = result
        print(f"best epoch={result['best_epoch']} best_score={result['best_score']:.4f}")

    metrics_name = (
        f"train_{name_tag}_seed{seed}.json"
        if (args.seed is not None or suffix)
        else f"train_{tag}.json"
    )
    save_json(results, paths["metrics"] / metrics_name)
    print(f"\nsaved metrics to {paths['metrics'] / metrics_name}")


if __name__ == "__main__":
    main()
