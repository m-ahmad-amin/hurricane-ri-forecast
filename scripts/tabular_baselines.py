from __future__ import annotations

"""Fit simple tabular baselines (logistic / GBM) on the same features and split."""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from src.data.prepare import TABULAR_FEATURES
from src.eval.regimes import assign_regimes
from src.train.metrics import bootstrap_bss, summarize_predictions
from src.utils import load_config, project_paths, save_json


def _eval_probs(y_test, probs, storms, test_df):
    summary = summarize_predictions(y_test, probs)
    boot = bootstrap_bss(y_test, probs, clusters=storms)
    framed = assign_regimes(test_df.assign(label=y_test))
    cat_mask = framed["regime_intensity"].to_numpy() == "Cat1-2_64-95kt"
    cat = summarize_predictions(y_test[cat_mask], probs[cat_mask])
    cat_boot = bootstrap_bss(
        y_test[cat_mask], probs[cat_mask], clusters=storms[cat_mask]
    )
    return {
        "overall": {**summary, "bootstrap_bss": boot},
        "cat12": {**cat, "bootstrap_bss": cat_boot},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular baselines for ablation sanity check")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    tag = args.tag or Path(args.config).stem
    # Table only (avoid loading images_*.npy, which is unnecessary here)
    import pandas as pd

    table = pd.read_parquet(paths["processed"] / f"samples_{tag}.parquet")
    feat_path = paths["processed"] / f"features_{tag}.json"
    if feat_path.exists():
        with open(feat_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in TABULAR_FEATURES if c in table.columns]

    train = table[table["split"] == "train"].copy()
    val = table[table["split"] == "val"].copy()
    test = table[table["split"] == "test"].copy()

    x_train = train[feature_cols].to_numpy(dtype=np.float64)
    y_train = train["ri_label"].to_numpy(dtype=int)
    x_val = val[feature_cols].to_numpy(dtype=np.float64)
    y_val = val["ri_label"].to_numpy(dtype=int)
    x_test = test[feature_cols].to_numpy(dtype=np.float64)
    y_test = test["ri_label"].to_numpy(dtype=int)

    # Fit scaler on train; impute nan with train median
    med = np.nanmedian(x_train, axis=0)
    def _fill(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        inds = np.where(np.isnan(out))
        out[inds] = np.take(med, inds[1])
        return out

    x_train, x_val, x_test = _fill(x_train), _fill(x_val), _fill(x_test)

    storms = test["storm_id"].to_numpy()
    results = {"per_seed": {}, "summary": {}}
    print("\nTabular baselines on held-out test (same features/split as NN tabular)")

    # Logistic: deterministic given data (no seed); fit once on train+val
    log_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    x_fit = np.vstack([x_train, x_val])
    y_fit = np.concatenate([y_train, y_val])
    log_model.fit(x_fit, y_fit)
    log_probs = log_model.predict_proba(x_test)[:, 1]
    results["logistic"] = _eval_probs(y_test, log_probs, storms, test)
    o = results["logistic"]["overall"]
    c = results["logistic"]["cat12"]
    print(
        f"{'logistic':<10} brier={o['brier']:.4f} bss={o['brier_skill_vs_climatology']:.4f} "
        f"[{o['bootstrap_bss']['bss_lo']:.4f}, {o['bootstrap_bss']['bss_hi']:.4f}] "
        f"| cat12 bss={c['brier_skill_vs_climatology']:.4f} "
        f"[{c['bootstrap_bss']['bss_lo']:.4f}, {c['bootstrap_bss']['bss_hi']:.4f}]"
    )

    # GBM: vary random_state across seeds
    from sklearn.linear_model import LogisticRegression as LR

    gbm_overall = []
    gbm_cat12 = []
    print("GBM across seeds:")
    for seed in args.seeds:
        gbm = GradientBoostingClassifier(random_state=seed)
        gbm.fit(x_train, y_train)
        val_scores = gbm.predict_proba(x_val)[:, 1].reshape(-1, 1)
        cal = LR(max_iter=1000)
        cal.fit(val_scores, y_val)
        test_scores = gbm.predict_proba(x_test)[:, 1].reshape(-1, 1)
        probs = cal.predict_proba(test_scores)[:, 1]
        ev = _eval_probs(y_test, probs, storms, test)
        results["per_seed"][f"gbm_seed{seed}"] = ev
        gbm_overall.append(ev["overall"]["brier_skill_vs_climatology"])
        gbm_cat12.append(ev["cat12"]["brier_skill_vs_climatology"])
        print(
            f"  seed {seed}: overall bss={gbm_overall[-1]:.4f} "
            f"[{ev['overall']['bootstrap_bss']['bss_lo']:.4f}, "
            f"{ev['overall']['bootstrap_bss']['bss_hi']:.4f}] "
            f"cat12 bss={gbm_cat12[-1]:.4f} "
            f"[{ev['cat12']['bootstrap_bss']['bss_lo']:.4f}, "
            f"{ev['cat12']['bootstrap_bss']['bss_hi']:.4f}]"
        )

    gbm_overall = np.asarray(gbm_overall, dtype=float)
    gbm_cat12 = np.asarray(gbm_cat12, dtype=float)
    results["gbm"] = {
        "seeds": args.seeds,
        "overall_bss_per_seed": gbm_overall.tolist(),
        "overall_bss_mean": float(gbm_overall.mean()),
        "overall_bss_std": float(gbm_overall.std(ddof=0)),
        "cat12_bss_per_seed": gbm_cat12.tolist(),
        "cat12_bss_mean": float(gbm_cat12.mean()),
        "cat12_bss_std": float(gbm_cat12.std(ddof=0)),
        # keep seed-42 storm-clustered CI as illustrative
        "illustrative_seed": args.seeds[0],
        "illustrative": results["per_seed"][f"gbm_seed{args.seeds[0]}"],
    }
    print(
        f"{'gbm':<10} mean overall bss={results['gbm']['overall_bss_mean']:.4f}"
        f"±{results['gbm']['overall_bss_std']:.4f} "
        f"mean cat12 bss={results['gbm']['cat12_bss_mean']:.4f}"
        f"±{results['gbm']['cat12_bss_std']:.4f}"
    )

    # Train-set basin x intensity climatology on test
    train_framed = assign_regimes(train.assign(label=y_train))
    test_framed = assign_regimes(test.assign(label=y_test))
    rates = (
        train_framed.groupby(["basin", "regime_intensity"], dropna=False)["label"]
        .mean()
        .to_dict()
    )
    clim_probs = np.array(
        [
            rates.get((b, r), float(y_train.mean()))
            for b, r in zip(
                test_framed["basin"].to_numpy(),
                test_framed["regime_intensity"].to_numpy(),
            )
        ],
        dtype=float,
    )
    clim_summary = summarize_predictions(y_test, clim_probs)
    clim_boot = bootstrap_bss(y_test, clim_probs, clusters=test["storm_id"].to_numpy())
    results["train_basin_intensity_climatology"] = {
        "overall": {**clim_summary, "bootstrap_bss": clim_boot}
    }
    print(
        f"{'clim_bxi':<10} brier={clim_summary['brier']:.4f} "
        f"bss={clim_summary['brier_skill_vs_climatology']:.4f} "
        f"[{clim_boot['bss_lo']:.4f}, {clim_boot['bss_hi']:.4f}] "
        f"pod={clim_summary['pod']:.3f}"
    )

    out = paths["metrics"] / f"tabular_baselines_{tag}.json"
    save_json(results, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
