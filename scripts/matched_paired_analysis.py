from __future__ import annotations

"""Matched-calibrator paired contrasts + MLP bias audit + ranking CIs."""

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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import fit_feature_scaler
from src.data.prepare import TABULAR_FEATURES
from src.eval.calibration import (
    apply_isotonic,
    apply_platt,
    fit_isotonic,
    fit_platt,
    fit_temperature,
    probs_to_logits,
    temperature_scale,
)
from src.eval.regimes import assign_regimes
from src.models.imagery import build_model
from src.train.metrics import (
    binary_counts,
    bootstrap_delta_bss,
    brier_skill_score,
    brier_score,
)
from src.utils import get_device, load_config, project_paths, save_json


def _fill(x, med):
    out = x.copy()
    inds = np.where(np.isnan(out))
    out[inds] = np.take(med, inds[1])
    return out


def _csi(y, p, thr):
    c = binary_counts(y, p, threshold=thr)
    den = c["tp"] + c["fp"] + c["fn"]
    return (c["tp"] / den) if den else float("nan"), c


def _best_csi_threshold(y_val, p_val, n_grid: int = 201) -> float:
    cands = np.unique(
        np.concatenate(
            [
                np.linspace(1e-4, 0.999, n_grid),
                np.quantile(np.asarray(p_val, dtype=float), np.linspace(0.01, 0.99, 99)),
            ]
        )
    )
    best_t, best_c = 0.5, -1.0
    for t in cands:
        csi, _ = _csi(y_val, p_val, float(t))
        if np.isfinite(csi) and csi > best_c:
            best_c, best_t = csi, float(t)
    return best_t


def _bootstrap_auc(
    y, p, clusters, n_boot=500, seed=42
) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    roc_vals, pr_vals = [], []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(clusters == c) for c in draw])
        yt, pt = y[idx], p[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            roc_vals.append(roc_auc_score(yt, pt))
            pr_vals.append(average_precision_score(yt, pt))
        except ValueError:
            continue
    roc_vals = np.asarray(roc_vals, dtype=float)
    pr_vals = np.asarray(pr_vals, dtype=float)
    out = {}
    for name, vals in (("roc_auc", roc_vals), ("pr_auc", pr_vals)):
        if vals.size == 0:
            out[name] = {"point": float("nan"), "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
        else:
            point = float(roc_auc_score(y, p)) if name == "roc_auc" else float(average_precision_score(y, p))
            out[name] = {
                "point": point,
                "mean": float(np.mean(vals)),
                "lo": float(np.quantile(vals, 0.025)),
                "hi": float(np.quantile(vals, 0.975)),
            }
    return out


class _TabDS(Dataset):
    def __init__(self, tab, feats, mean, std):
        self.tab = tab.reset_index(drop=True)
        f = self.tab[feats].to_numpy(dtype=np.float32)
        f = np.where(np.isfinite(f), f, mean)
        self.x = ((f - mean) / std).astype(np.float32)
        self.y = self.tab["ri_label"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.tab)

    def __getitem__(self, i):
        return {
            "tabular": torch.from_numpy(self.x[i]),
            "image": torch.zeros(1, 128, 128),
            "label": torch.tensor(self.y[i]),
        }


def _predict_tab(model, loader, device):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for b in loader:
            logit = model(b["tabular"].to(device), b["image"].to(device))
            ps.append(torch.sigmoid(logit).cpu().numpy())
            ys.append(b["label"].numpy())
    return np.concatenate(ps), np.concatenate(ys)


def _load_pred(paths, model, tag, seed, default_seed):
    cands = [paths["metrics"] / f"predictions_{model}_{tag}_seed{seed}.csv"]
    if seed == default_seed:
        cands.append(paths["metrics"] / f"predictions_{model}_{tag}.csv")
    for p in cands:
        if p.exists():
            return pd.read_csv(p)
    return None


def _ckpt(paths, model, tag, seed, default_seed):
    tagged = paths["checkpoints"] / f"{model}_{tag}_seed{seed}.pt"
    if tagged.exists():
        return tagged
    if seed == default_seed:
        untagged = paths["checkpoints"] / f"{model}_{tag}.pt"
        if untagged.exists():
            return untagged
    raise FileNotFoundError(model, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    tag = args.tag or Path(args.config).stem
    default_seed = int(cfg["train"].get("seed", 42))
    device = get_device()

    table = pd.read_parquet(paths["processed"] / f"samples_{tag}.parquet")
    feat_path = paths["processed"] / f"features_{tag}.json"
    feature_cols = (
        json.load(open(feat_path, encoding="utf-8"))
        if feat_path.exists()
        else [c for c in TABULAR_FEATURES if c in table.columns]
    )
    mean, std = fit_feature_scaler(table, feature_cols)

    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    test = table[table["split"] == "test"].reset_index(drop=True)
    y_train = train["ri_label"].to_numpy(dtype=int)
    y_val = val["ri_label"].to_numpy(dtype=float)
    y_test = test["ri_label"].to_numpy(dtype=float)
    storms_test = test["storm_id"].to_numpy()

    x_train = _fill(train[feature_cols].to_numpy(np.float64), np.nanmedian(train[feature_cols].to_numpy(np.float64), 0))
    med = np.nanmedian(train[feature_cols].to_numpy(np.float64), 0)
    x_train = _fill(train[feature_cols].to_numpy(np.float64), med)
    x_val = _fill(val[feature_cols].to_numpy(np.float64), med)
    x_test = _fill(test[feature_cols].to_numpy(np.float64), med)

    framed = assign_regimes(test.assign(label=y_test))
    cat_mask = framed["regime_intensity"].to_numpy() == "Cat1-2_64-95kt"

    paired = []
    ranking = []
    bias_audit = []
    detection = []

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")

        # --- GBM calibrators ---
        gbm = GradientBoostingClassifier(random_state=seed)
        gbm.fit(x_train, y_train)
        gbm_val_raw = gbm.predict_proba(x_val)[:, 1]
        gbm_test_raw = gbm.predict_proba(x_test)[:, 1]
        gbm_cals = {
            "raw": gbm_test_raw,
            "temperature": temperature_scale(
                probs_to_logits(gbm_test_raw), fit_temperature(y_val, gbm_val_raw)
            ),
            "platt": apply_platt(fit_platt(y_val, gbm_val_raw), gbm_test_raw),
            "isotonic": apply_isotonic(fit_isotonic(y_val, gbm_val_raw), gbm_test_raw),
        }
        gbm_val_cals = {
            "raw": gbm_val_raw,
            "temperature": temperature_scale(
                probs_to_logits(gbm_val_raw), fit_temperature(y_val, gbm_val_raw)
            ),
            "platt": apply_platt(fit_platt(y_val, gbm_val_raw), gbm_val_raw),
            "isotonic": apply_isotonic(fit_isotonic(y_val, gbm_val_raw), gbm_val_raw),
        }

        # --- Neural isotonic/primary from CSVs ---
        neural_test = {}
        for model in ("tabular", "imagery", "fusion"):
            pred = _load_pred(paths, model, tag, seed, default_seed)
            if pred is None:
                continue
            m = test[["storm_id", "time"]].merge(pred, on=["storm_id", "time"], how="inner")
            assert len(m) == len(test), (model, seed, len(m), len(test))
            neural_test[model] = {
                "raw": m["prob_raw"].to_numpy(float),
                "temperature": m["prob_temperature"].to_numpy(float),
                "isotonic": m["prob_isotonic"].to_numpy(float),
                "primary": m["prob_primary"].to_numpy(float),
            }

        # --- Tabular MLP train/val/test raw + Platt via checkpoint ---
        ckpt_path = _ckpt(paths, "tabular", tag, seed, default_seed)
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        model = build_model("tabular", n_features=len(feature_cols))
        model.load_state_dict(state["model"])
        model.to(device)
        ckpt_meta = {
            "epoch": state.get("epoch"),
            "best_score": state.get("best_score"),
            "metrics": state.get("metrics"),
        }
        print(
            f"  tabular ckpt epoch={ckpt_meta['epoch']} "
            f"val_brier={ckpt_meta['metrics'].get('val_brier') if ckpt_meta['metrics'] else None} "
            f"val_bss={ckpt_meta['metrics'].get('val_bss') if ckpt_meta['metrics'] else None}"
        )

        loaders = {}
        for name, df in (("train", train), ("val", val), ("test", test)):
            loaders[name] = DataLoader(
                _TabDS(df, feature_cols, mean, std), batch_size=256, shuffle=False
            )
        p_tr, y_tr = _predict_tab(model, loaders["train"], device)
        p_va, y_va = _predict_tab(model, loaders["val"], device)
        p_te, y_te = _predict_tab(model, loaders["test"], device)
        assert np.allclose(y_te, y_test)

        T = fit_temperature(y_va, p_va)
        platt = fit_platt(y_va, p_va)
        iso = fit_isotonic(y_va, p_va)
        mlp_test = {
            "raw": p_te,
            "temperature": temperature_scale(probs_to_logits(p_te), T),
            "platt": apply_platt(platt, p_te),
            "isotonic": apply_isotonic(iso, p_te),
        }
        mlp_val = {
            "raw": p_va,
            "temperature": temperature_scale(probs_to_logits(p_va), T),
            "platt": apply_platt(platt, p_va),
            "isotonic": apply_isotonic(iso, p_va),
        }
        # prefer CSV isotonic for consistency with other neural rows when present
        if "tabular" in neural_test:
            mlp_test["isotonic"] = neural_test["tabular"]["isotonic"]
            mlp_test["temperature"] = neural_test["tabular"]["temperature"]
            mlp_test["raw"] = neural_test["tabular"]["raw"]

        bias_audit.append(
            {
                "seed": seed,
                "checkpoint": {
                    "path": str(ckpt_path.name),
                    "best_epoch": ckpt_meta["epoch"],
                    "best_val_brier": ckpt_meta["best_score"],
                    "best_val_bss": (ckpt_meta["metrics"] or {}).get("val_bss"),
                    "note": "checkpoint stores best_state by min val Brier; evaluate.py loads that file",
                },
                "ri_rates": {
                    "train": float(np.mean(y_tr)),
                    "val": float(np.mean(y_va)),
                    "test": float(np.mean(y_te)),
                },
                "mean_pred_raw": {
                    "train": float(np.mean(p_tr)),
                    "val": float(np.mean(p_va)),
                    "test": float(np.mean(p_te)),
                },
                "mean_pred_temperature_test": float(np.mean(mlp_test["temperature"])),
                "mean_pred_platt_test": float(np.mean(mlp_test["platt"])),
                "mean_pred_isotonic_test": float(np.mean(mlp_test["isotonic"])),
                "val_bss_raw": float(brier_skill_score(y_va, p_va)),
                "val_brier_raw": float(brier_score(y_va, p_va)),
                "val_climatology_brier": float(brier_score(y_va, np.full_like(y_va, float(np.mean(y_va))))),
                "pos_weight_note": (
                    "Training uses BCEWithLogitsLoss with pos_weight=n_neg/n_pos, "
                    "which induces overforecasting bias (mean p >> RI rate) even at best val Brier."
                ),
            }
        )
        print(
            f"  MLP raw pbar train/val/test="
            f"{np.mean(p_tr):.3f}/{np.mean(p_va):.3f}/{np.mean(p_te):.3f} "
            f"(rates {np.mean(y_tr):.3f}/{np.mean(y_va):.3f}/{np.mean(y_te):.3f})"
        )
        print(
            f"  temp raises test pbar {np.mean(p_te):.3f} -> {np.mean(mlp_test['temperature']):.3f} "
            f"(T={T:.3f}>1 pulls toward 0.5)"
        )

        # --- Detection: FAR/CSI at val CSI-max threshold ---
        for cal in ("raw", "temperature", "platt", "isotonic"):
            thr = _best_csi_threshold(y_va, mlp_val[cal])
            csi, counts = _csi(y_test, mlp_test[cal], thr)
            detection.append(
                {
                    "seed": seed,
                    "model": "tabular",
                    "calibrator": cal,
                    "val_csi_threshold": thr,
                    "test_csi": csi,
                    "test_pod": counts["pod"],
                    "test_far": counts["far"],
                    **{k: counts[k] for k in ("tp", "fp", "fn", "tn")},
                }
            )
            print(
                f"  MLP {cal}: val-CSI-thr={thr:.3f} test CSI={csi:.3f} "
                f"POD={counts['pod']:.3f} FAR={counts['far']:.3f}"
            )
        for cal in ("raw", "temperature", "platt", "isotonic"):
            thr = _best_csi_threshold(y_val, gbm_val_cals[cal])
            csi, counts = _csi(y_test, gbm_cals[cal], thr)
            detection.append(
                {
                    "seed": seed,
                    "model": "gbm",
                    "calibrator": cal,
                    "val_csi_threshold": thr,
                    "test_csi": csi,
                    "test_pod": counts["pod"],
                    "test_far": counts["far"],
                    **{k: counts[k] for k in ("tp", "fp", "fn", "tn")},
                }
            )

        # --- Ranking CIs (overall, raw probs = ranking) ---
        for model_name, probs in (
            ("tabular", mlp_test["raw"]),
            ("gbm", gbm_cals["raw"]),
            ("imagery", neural_test["imagery"]["raw"] if "imagery" in neural_test else None),
        ):
            if probs is None:
                continue
            auc = _bootstrap_auc(y_test, probs, storms_test, n_boot=args.n_boot, seed=seed)
            ranking.append({"seed": seed, "model": model_name, "scope": "overall", **auc})
            print(
                f"  {model_name} ROC-AUC {auc['roc_auc']['point']:.3f} "
                f"[{auc['roc_auc']['lo']:.3f},{auc['roc_auc']['hi']:.3f}] "
                f"PR-AUC {auc['pr_auc']['point']:.3f} "
                f"[{auc['pr_auc']['lo']:.3f},{auc['pr_auc']['hi']:.3f}]"
            )

        # --- Paired BSS contrasts ---
        def _pair(name_a, p_a, name_b, p_b, scope_mask, scope_name, cal_name):
            y = y_test[scope_mask]
            pa, pb = p_a[scope_mask], p_b[scope_mask]
            st = storms_test[scope_mask]
            point = float(brier_skill_score(y, pa) - brier_skill_score(y, pb))
            boot = bootstrap_delta_bss(y, pa, pb, n_boot=args.n_boot, clusters=st, seed=seed)
            rec = {
                "seed": seed,
                "contrast": f"{name_a}_minus_{name_b}",
                "calibrator": cal_name,
                "scope": scope_name,
                "point_delta_bss": point,
                "bss_a": float(brier_skill_score(y, pa)),
                "bss_b": float(brier_skill_score(y, pb)),
                "n": int(len(y)),
                "n_storms": int(len(np.unique(st))),
                **boot,
            }
            paired.append(rec)
            print(
                f"  {cal_name} {name_a}-{name_b} {scope_name}: "
                f"point={point:+.3f} [{boot['delta_lo']:.3f},{boot['delta_hi']:.3f}]"
            )
            return rec

        # isotonic vs isotonic (the matched ask)
        if "imagery" in neural_test:
            _pair(
                "imagery",
                neural_test["imagery"]["isotonic"],
                "gbm",
                gbm_cals["isotonic"],
                cat_mask,
                "Cat1-2",
                "isotonic",
            )
            _pair(
                "imagery",
                neural_test["imagery"]["isotonic"],
                "tabular",
                mlp_test["isotonic"],
                cat_mask,
                "Cat1-2",
                "isotonic",
            )
            # also overall isotonic imagery - gbm
            _pair(
                "imagery",
                neural_test["imagery"]["isotonic"],
                "gbm",
                gbm_cals["isotonic"],
                np.ones(len(y_test), dtype=bool),
                "overall",
                "isotonic",
            )

        # MLP - GBM matched (platt and isotonic), overall
        for cal in ("platt", "isotonic"):
            _pair(
                "tabular",
                mlp_test[cal],
                "gbm",
                gbm_cals[cal],
                np.ones(len(y_test), dtype=bool),
                "overall",
                cal,
            )
            _pair(
                "tabular",
                mlp_test[cal],
                "gbm",
                gbm_cals[cal],
                cat_mask,
                "Cat1-2",
                cal,
            )

    out = {
        "tag": tag,
        "n_boot": args.n_boot,
        "bias_audit": bias_audit,
        "paired_delta_bss": paired,
        "ranking_clustered": ranking,
        "detection_val_csi_threshold": detection,
        "notes": [
            "Training uses pos_weight in BCEWithLogitsLoss, inducing overforecasting bias.",
            "Checkpoint restores min-val-Brier epoch; that epoch can still have val BSS << 0.",
            "Temperature T>1 pulls probabilities toward 0.5, which can raise mean p when raw mean < 0.5.",
            "POD@rate omitted: rank-invariant; use FAR/CSI at validation CSI-max threshold instead.",
        ],
    }
    out_path = paths["metrics"] / f"matched_paired_{tag}.json"
    save_json(out, out_path)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
