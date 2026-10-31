from __future__ import annotations

"""Add Platt probs for imagery/fusion; write unweighted MLP reliability figure."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import RIDataset, fit_feature_scaler, load_processed, make_loaders
from src.eval.calibration import apply_platt, fit_platt, fit_isotonic, apply_isotonic, fit_temperature, temperature_scale, probs_to_logits
from src.eval.cases import save_reliability_figure
from src.models.imagery import build_model
from src.train.loop import predict_loader
from src.train.metrics import reliability_curve, summarize_predictions
from src.utils import get_device, load_config, project_paths, save_json


def _ckpt(paths, model, name_tag, seed):
    p = paths["checkpoints"] / f"{model}_{name_tag}_seed{seed}.pt"
    if p.exists():
        return p
    if name_tag == "default" and seed == 42:
        alt = paths["checkpoints"] / f"{model}_default.pt"
        if alt.exists():
            return alt
    raise FileNotFoundError(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--images-path", default=os.environ.get("IMAGE_NPY"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    tag = Path(args.config).stem
    device = get_device()

    table, images, feature_cols, _ = load_processed(paths["processed"], tag)
    if args.images_path:
        images = np.load(args.images_path, mmap_mode="r")
    loaders, _, _ = make_loaders(table, images, feature_cols, cfg["train"]["batch_size"], 0)
    test = table[table["split"] == "test"].reset_index(drop=True)

    platt_rows = []
    for model in ("imagery", "fusion"):
        for seed in args.seeds:
            for name_tag in ("default",):  # weighted originals; noweight filled when ready
                try:
                    ckpt = _ckpt(paths, model, name_tag, seed)
                except FileNotFoundError:
                    continue
                net = build_model(model, n_features=len(feature_cols))
                try:
                    state = torch.load(ckpt, map_location=device, weights_only=False)
                except TypeError:
                    state = torch.load(ckpt, map_location=device)
                net.load_state_dict(state["model"])
                net.to(device)
                val_p, val_y, _ = predict_loader(net, loaders["val"], device)
                test_p, test_y, _ = predict_loader(net, loaders["test"], device)
                platt = apply_platt(fit_platt(val_y, val_p), test_p)
                iso = apply_isotonic(fit_isotonic(val_y, val_p), test_p)
                T = fit_temperature(val_y, val_p)
                temp = temperature_scale(probs_to_logits(test_p), T)
                for cal, p in (("raw", test_p), ("temperature", temp), ("platt", platt), ("isotonic", iso)):
                    s = summarize_predictions(test_y, p)
                    platt_rows.append(
                        {
                            "model": model,
                            "name_tag": name_tag,
                            "seed": seed,
                            "calibrator": cal,
                            "bss": s["brier_skill_vs_climatology"],
                            "mean_pred": float(np.mean(p)),
                            "pod": s["pod"],
                            "roc_auc": s["roc_auc"],
                            "pr_auc": s["pr_auc"],
                        }
                    )
                    print(
                        f"{model}/{name_tag}/s{seed}/{cal}: BSS={s['brier_skill_vs_climatology']:+.3f} "
                        f"pbar={np.mean(p):.3f} AUC={s['roc_auc']:.3f}"
                    )
                # write augmented prediction CSV
                pred_path = paths["metrics"] / f"predictions_{model}_{name_tag}_seed{seed}.csv"
                if not pred_path.exists() and seed == 42 and name_tag == "default":
                    pred_path = paths["metrics"] / f"predictions_{model}_{name_tag}.csv"
                if pred_path.exists():
                    df = pd.read_csv(pred_path)
                    # align
                    m = test[["storm_id", "time"]].merge(df, on=["storm_id", "time"], how="left")
                    m["prob_platt"] = platt
                    out = paths["metrics"] / f"predictions_{model}_{name_tag}_seed{seed}_with_platt.csv"
                    m.to_csv(out, index=False)

    save_json({"rows": platt_rows}, paths["metrics"] / "platt_imagery_fusion_default.json")

    # Unweighted MLP reliability figure
    for seed in args.seeds:
        pred = paths["metrics"] / f"predictions_tabular_default_noweight_seed{seed}.csv"
        if not pred.exists():
            continue
        df = pd.read_csv(pred)
        for cal, col in (("raw", "prob_raw"), ("temperature", "prob_temperature"), ("isotonic", "prob_isotonic")):
            rel = reliability_curve(df["label"].to_numpy(), df[col].to_numpy())
            out_fig = paths["figures"] / f"reliability_tabular_noweight_{cal}_seed{seed}.png"
            save_reliability_figure(rel, out_fig, title=f"Unweighted MLP {cal} (seed {seed})")
            # copy seed42 raw to paper/figures
            if seed == 42 and cal == "raw":
                paper_fig = ROOT / "paper" / "figures" / "reliability_tabular_noweight_raw.png"
                save_reliability_figure(rel, paper_fig, title="Unweighted MLP (raw)")
        print(f"wrote reliability figs for unweighted MLP seed {seed}")
        break


if __name__ == "__main__":
    main()
