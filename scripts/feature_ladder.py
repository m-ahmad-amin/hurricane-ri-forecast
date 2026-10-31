from __future__ import annotations

"""Environment-only -> +IR scalars ladder (unweighted MLP + GBM)."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import RIDataset, fit_feature_scaler
from src.data.prepare import TABULAR_FEATURES
from src.eval.calibration import apply_platt, fit_platt
from src.models.imagery import build_model
from src.train.loop import train_model
from src.train.metrics import bootstrap_bss, summarize_predictions
from src.utils import get_device, load_config, project_paths, save_json, set_seed

ENV_FEATURES = [
    "sst_c",
    "shear_low_0_800",
    "shear_high_0_800",
    "potential_intensity_kt",
    "pw_mean",
    "distance_to_land_km",
    "storm_speed",
    "intensity_kt",
]
IR_SCALAR_FEATURES = [c for c in TABULAR_FEATURES if c.startswith("ir_")]


def _fill(x, med):
    out = x.copy()
    inds = np.where(np.isnan(out))
    out[inds] = np.take(med, inds[1])
    return out


def _gbm_eval(x_train, y_train, x_val, y_val, x_test, y_test, storms, seed):
    gbm = GradientBoostingClassifier(random_state=seed)
    gbm.fit(x_train, y_train)
    vr = gbm.predict_proba(x_val)[:, 1]
    tr = gbm.predict_proba(x_test)[:, 1]
    pl = fit_platt(y_val, vr)
    probs = apply_platt(pl, tr)
    s = summarize_predictions(y_test, probs)
    boot = bootstrap_bss(y_test, probs, clusters=storms)
    return {**s, "bootstrap_bss": boot, "mean_pred": float(np.mean(probs))}


def _train_mlp(table, feats, seed, cfg, paths, tag, rung_name):
    set_seed(seed)
    mean, std = fit_feature_scaler(table, feats)
    loaders = {}
    for split in ("train", "val", "test"):
        mask = table["split"].to_numpy() == split
        if mask.sum() == 0:
            continue
        ds = RIDataset(table.loc[mask], None, feats, mean, std)
        loaders[split] = DataLoader(
            ds, batch_size=cfg["train"]["batch_size"], shuffle=(split == "train")
        )
    model = build_model("tabular", n_features=len(feats))
    ckpt = paths["checkpoints"] / f"tabular_{tag}_ladder_{rung_name}_seed{seed}.pt"
    result = train_model(
        model,
        loaders,
        device=get_device(),
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        checkpoint_path=ckpt,
        patience=cfg["train"].get("patience", 8),
        min_epochs=cfg["train"].get("min_epochs", 5),
        use_pos_weight=False,
    )
    # eval raw probs on test
    device = get_device()
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    test = table[table["split"] == "test"].reset_index(drop=True)
    ds = RIDataset(test, None, feats, mean, std)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for b in loader:
            tab = b["tabular"].to(device)
            img = b["image"].to(device)
            logit = model(tab, img)
            probs.append(torch.sigmoid(logit).detach().cpu().numpy())
            labels.append(b["label"].numpy())
    p = np.concatenate(probs)
    y = np.concatenate(labels)
    storms = test["storm_id"].to_numpy()
    s = summarize_predictions(y, p)
    boot = bootstrap_bss(y, p, clusters=storms)
    return {
        **s,
        "bootstrap_bss": boot,
        "mean_pred": float(np.mean(p)),
        "best_epoch": result["best_epoch"],
        "best_val_brier": result["best_score"],
        "checkpoint": str(ckpt),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    tag = Path(args.config).stem
    table = pd.read_parquet(paths["processed"] / f"samples_{tag}.parquet")

    rungs = {
        "env_only": ENV_FEATURES,
        "env_plus_ir_scalars": ENV_FEATURES + IR_SCALAR_FEATURES,
    }
    # full tabular is env+IR scalars (same as second rung if TABULAR_FEATURES matches)
    assert set(rungs["env_plus_ir_scalars"]) == set(TABULAR_FEATURES)

    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    test = table[table["split"] == "test"]
    y_train = train["ri_label"].to_numpy(int)
    y_val = val["ri_label"].to_numpy(float)
    y_test = test["ri_label"].to_numpy(float)
    storms = test["storm_id"].to_numpy()

    out = {"rungs": {}, "note": "Unweighted MLP and Platt-calibrated GBM; no image pathway here."}
    for rung, feats in rungs.items():
        print(f"\n=== rung {rung} n_feat={len(feats)} ===")
        med = np.nanmedian(train[feats].to_numpy(np.float64), 0)
        x_train = _fill(train[feats].to_numpy(np.float64), med)
        x_val = _fill(val[feats].to_numpy(np.float64), med)
        x_test = _fill(test[feats].to_numpy(np.float64), med)
        gbm_bss, mlp_bss = [], []
        gbm_rows, mlp_rows = [], []
        for seed in args.seeds:
            g = _gbm_eval(x_train, y_train, x_val, y_val, x_test, y_test, storms, seed)
            gbm_bss.append(g["brier_skill_vs_climatology"])
            gbm_rows.append({"seed": seed, **{k: g[k] for k in ("brier", "brier_skill_vs_climatology", "pod", "roc_auc", "mean_pred")}, "bootstrap_bss": g["bootstrap_bss"]})
            print(f"  GBM seed {seed}: BSS={g['brier_skill_vs_climatology']:+.3f} pbar={g['mean_pred']:.3f}")
            m = _train_mlp(table, feats, seed, cfg, paths, tag, rung)
            mlp_bss.append(m["brier_skill_vs_climatology"])
            mlp_rows.append({"seed": seed, **{k: m[k] for k in ("brier", "brier_skill_vs_climatology", "pod", "roc_auc", "mean_pred", "best_epoch")}, "bootstrap_bss": m["bootstrap_bss"]})
            print(f"  MLP seed {seed}: BSS={m['brier_skill_vs_climatology']:+.3f} pbar={m['mean_pred']:.3f}")
        out["rungs"][rung] = {
            "features": feats,
            "gbm": {
                "per_seed": gbm_rows,
                "bss_mean": float(np.mean(gbm_bss)),
                "bss_std": float(np.std(gbm_bss, ddof=1)),
            },
            "mlp_unweighted": {
                "per_seed": mlp_rows,
                "bss_mean": float(np.mean(mlp_bss)),
                "bss_std": float(np.std(mlp_bss, ddof=1)),
            },
        }
        print(
            f"  summary GBM {out['rungs'][rung]['gbm']['bss_mean']:+.3f}±{out['rungs'][rung]['gbm']['bss_std']:.3f} "
            f"MLP {out['rungs'][rung]['mlp_unweighted']['bss_mean']:+.3f}±{out['rungs'][rung]['mlp_unweighted']['bss_std']:.3f}"
        )

    out_path = paths["metrics"] / f"feature_ladder_{tag}.json"
    save_json(out, out_path)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
