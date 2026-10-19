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
from src.eval.cases import pick_cases, save_case_figure, save_reliability_figure
from src.eval.ships_rii import compare_to_ships_rii
from src.models.imagery import build_model
from src.train.loop import predict_loader
from src.train.metrics import bootstrap_bss, summarize_predictions
from src.utils import ensure_dirs, get_device, load_config, project_paths, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RI models")
    parser.add_argument("--config", default="configs/al062018.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--model", default="fusion")
    parser.add_argument("--ships-csv", default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    ensure_dirs(paths)

    tag = args.tag or Path(args.config).stem
    table, images, feature_cols, _ = load_processed(paths["processed"], tag)
    loaders, _, _ = make_loaders(
        table,
        images,
        feature_cols=feature_cols,
        batch_size=cfg["train"]["batch_size"],
        num_workers=0,
    )
    if "test" not in loaders:
        raise RuntimeError("no test split available")

    device = get_device()
    model = build_model(args.model, n_features=len(feature_cols))
    ckpt_path = paths["checkpoints"] / f"{args.model}_{tag}.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    probs, labels, indices = predict_loader(model, loaders["test"], device)
    test_table = table[table["split"] == "test"].reset_index(drop=True)
    test_images = images[table["split"].to_numpy() == "test"]

    summary = summarize_predictions(labels, probs)
    summary["bootstrap_bss"] = bootstrap_bss(labels, probs)
    summary["model"] = args.model
    summary["tag"] = tag

    cases = pick_cases(test_table, probs)
    summary["cases"] = cases
    save_case_figure(test_images, cases, paths["figures"] / f"cases_{args.model}_{tag}.png")
    save_reliability_figure(
        summary["reliability"],
        paths["figures"] / f"reliability_{args.model}_{tag}.png",
        title=f"{args.model} reliability ({tag})",
    )

    if args.ships_csv:
        import pandas as pd

        ships = pd.read_csv(args.ships_csv)
        merged = test_table.merge(ships, on=["storm_id", "time"], how="left")
        summary["ships_rii_comparison"] = compare_to_ships_rii(
            labels,
            probs,
            merged["ships_rii_prob"].to_numpy(dtype=float),
        )

    pred_path = paths["metrics"] / f"predictions_{args.model}_{tag}.csv"
    out_pred = test_table.copy()
    out_pred["prob"] = probs
    out_pred["label"] = labels
    out_pred.to_csv(pred_path, index=False)

    save_json(summary, paths["metrics"] / f"eval_{args.model}_{tag}.json")
    print(summary)


if __name__ == "__main__":
    main()
