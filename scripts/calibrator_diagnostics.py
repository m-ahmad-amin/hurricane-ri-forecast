from __future__ import annotations

"""Mean predicted probs, event-rate-matched POD, and CSI-optimal (val) POD."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from src.train.metrics import binary_counts
from src.utils import load_config, project_paths, save_json


def _fill(x, med):
    out = x.copy()
    inds = np.where(np.isnan(out))
    out[inds] = np.take(med, inds[1])
    return out


def _csi(y, p, thr):
    c = binary_counts(y, p, threshold=thr)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    den = tp + fp + fn
    return (tp / den) if den else float("nan")


def _best_csi_threshold(y_val, p_val, n_grid: int = 201) -> float:
    """Maximize CSI on validation over thresholds in (0,1)."""
    y_val = np.asarray(y_val, dtype=float)
    p_val = np.asarray(p_val, dtype=float)
    # include unique probs plus a fine grid
    cands = np.unique(
        np.concatenate(
            [
                np.linspace(1e-4, 0.999, n_grid),
                np.quantile(p_val, np.linspace(0.01, 0.99, 99)),
            ]
        )
    )
    best_t, best_c = 0.5, -1.0
    for t in cands:
        c = _csi(y_val, p_val, float(t))
        if np.isfinite(c) and c > best_c:
            best_c, best_t = c, float(t)
    return best_t


def _pod_at_rate(y, p, target_rate: float) -> dict:
    """Threshold so forecast positive rate ~= target_rate; report POD and realized FAR."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(p)
    k = max(int(round(target_rate * n)), 1)
    # alert on top-k scores
    order = np.argsort(-p)
    hat = np.zeros(n, dtype=int)
    hat[order[:k]] = 1
    thr = float(p[order[k - 1]]) if k <= n else 0.0
    c = binary_counts(y, hat.astype(float), threshold=0.5)
    # binary_counts with hat as "prob" and thr 0.5 works if hat in {0,1}
    # clearer: compute directly
    tp = int(((hat == 1) & (y == 1)).sum())
    fp = int(((hat == 1) & (y == 0)).sum())
    fn = int(((hat == 0) & (y == 1)).sum())
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    return {
        "threshold_topk": thr,
        "alert_rate": float(hat.mean()),
        "pod_rate_matched": pod,
        "far_rate_matched": far,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _pod_at_fixed_far(y, p, target_far: float = 0.5) -> dict:
    """Sweep thresholds; pick the one with FAR closest to target_far (among alerts>0)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    thrs = np.unique(p)
    best = None
    best_gap = 1e9
    for t in thrs:
        c = binary_counts(y, p, threshold=float(t))
        if c["tp"] + c["fp"] == 0:
            continue
        far = c["far"]
        if not np.isfinite(far):
            continue
        gap = abs(far - target_far)
        if gap < best_gap:
            best_gap = gap
            best = {
                "threshold": float(t),
                "pod_at_far": c["pod"],
                "far": far,
                "far_target": target_far,
                "tp": c["tp"],
                "fp": c["fp"],
                "fn": c["fn"],
            }
    return best or {
        "threshold": float("nan"),
        "pod_at_far": float("nan"),
        "far": float("nan"),
        "far_target": target_far,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--far-target", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    tag = args.tag or Path(args.config).stem
    default_seed = int(cfg["train"].get("seed", 42))

    table = pd.read_parquet(paths["processed"] / f"samples_{tag}.parquet")
    feat_path = paths["processed"] / f"features_{tag}.json"
    if feat_path.exists():
        with open(feat_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in TABULAR_FEATURES if c in table.columns]

    rates = {
        s: float(table.loc[table["split"] == s, "ri_label"].mean())
        for s in ("train", "val", "test")
    }
    print("RI rates:", rates)

    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    test = table[table["split"] == "test"].reset_index(drop=True)
    y_train = train["ri_label"].to_numpy(dtype=int)
    y_val = val["ri_label"].to_numpy(dtype=int)
    y_test = test["ri_label"].to_numpy(dtype=float)
    x_train = train[feature_cols].to_numpy(dtype=np.float64)
    x_val = val[feature_cols].to_numpy(dtype=np.float64)
    x_test = test[feature_cols].to_numpy(dtype=np.float64)
    med = np.nanmedian(x_train, axis=0)
    x_train, x_val, x_test = _fill(x_train, med), _fill(x_val, med), _fill(x_test, med)

    rows = []

    # Neural from CSVs (test only for rate-matched / mean / FAR;
    # CSI-opt needs val — approximate with temperature/isotonic using saved
    # primary columns only for rate-matched on test)
    for seed in args.seeds:
        for model in ("tabular", "imagery", "fusion"):
            candidates = [
                paths["metrics"] / f"predictions_{model}_{tag}_seed{seed}.csv",
            ]
            if seed == default_seed:
                candidates.append(paths["metrics"] / f"predictions_{model}_{tag}.csv")
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                continue
            pred = pd.read_csv(path)
            merged = test[["storm_id", "time"]].merge(pred, on=["storm_id", "time"], how="inner")
            y = merged["label"].to_numpy(dtype=float) if "label" in merged.columns else y_test[: len(merged)]
            if "label" not in merged.columns and "ri_label" in merged.columns:
                y = merged["ri_label"].to_numpy(dtype=float)

            for cal, col in (
                ("raw", "prob_raw"),
                ("temperature", "prob_temperature"),
                ("isotonic", "prob_isotonic"),
            ):
                p = merged[col].to_numpy(dtype=float)
                rate_m = _pod_at_rate(y, p, rates["test"])
                far_m = _pod_at_fixed_far(y, p, args.far_target)
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "calibrator": cal,
                        "mean_pred": float(np.mean(p)),
                        "obs_rate": float(np.mean(y)),
                        "pod_0.5": binary_counts(y, p, 0.5)["pod"],
                        **rate_m,
                        **{f"farfix_{k}": v for k, v in far_m.items()},
                    }
                )
                print(
                    f"{model:8} s{seed} {cal:11} mean_p={np.mean(p):.4f} "
                    f"obs={np.mean(y):.4f} POD@0.5={binary_counts(y,p,0.5)['pod']:.3f} "
                    f"POD@rate={rate_m['pod_rate_matched']:.3f} "
                    f"POD@FAR~{args.far_target}={far_m.get('pod_at_far', float('nan')):.3f}"
                )

        # GBM with full val calibrators + CSI-opt threshold from val
        gbm = GradientBoostingClassifier(random_state=seed)
        gbm.fit(x_train, y_train)
        val_raw = gbm.predict_proba(x_val)[:, 1]
        test_raw = gbm.predict_proba(x_test)[:, 1]
        T = fit_temperature(y_val.astype(float), val_raw)
        platt = fit_platt(y_val.astype(float), val_raw)
        iso = fit_isotonic(y_val.astype(float), val_raw)
        cals = {
            "raw": (val_raw, test_raw),
            "temperature": (
                temperature_scale(probs_to_logits(val_raw), T),
                temperature_scale(probs_to_logits(test_raw), T),
            ),
            "platt": (apply_platt(platt, val_raw), apply_platt(platt, test_raw)),
            "isotonic": (apply_isotonic(iso, val_raw), apply_isotonic(iso, test_raw)),
        }
        for cal, (pv, pt) in cals.items():
            thr_csi = _best_csi_threshold(y_val.astype(float), pv)
            c_test = binary_counts(y_test, pt, thr_csi)
            rate_m = _pod_at_rate(y_test, pt, rates["test"])
            far_m = _pod_at_fixed_far(y_test, pt, args.far_target)
            rows.append(
                {
                    "model": "gbm",
                    "seed": seed,
                    "calibrator": cal,
                    "mean_pred": float(np.mean(pt)),
                    "obs_rate": float(np.mean(y_test)),
                    "pod_0.5": binary_counts(y_test, pt, 0.5)["pod"],
                    "csi_val_threshold": thr_csi,
                    "pod_at_csi_thr": c_test["pod"],
                    "far_at_csi_thr": c_test["far"],
                    **rate_m,
                    **{f"farfix_{k}": v for k, v in far_m.items()},
                }
            )
            print(
                f"{'gbm':8} s{seed} {cal:11} mean_p={np.mean(pt):.4f} "
                f"POD@0.5={binary_counts(y_test,pt,0.5)['pod']:.3f} "
                f"POD@CSI-thr={c_test['pod']:.3f} (t={thr_csi:.3f}) "
                f"POD@rate={rate_m['pod_rate_matched']:.3f}"
            )

    # Tabular Platt/isotonic already in CSVs as isotonic; add Platt mean from
    # calibrator_grid if present — also recompute tabular CSI via checkpoint? skip;
    # for tabular use isotonic/temp from CSV and note CSI-opt only for GBM + we
    # can fit calibrators on... we don't have val neural probs in CSV.
    # Approximate neural CSI-opt: use test-only rate-matched as primary detection metric.

    # Seed-mean summary for paper
    summary = {}
    for model in ("tabular", "imagery", "fusion", "gbm"):
        summary[model] = {}
        for cal in ("raw", "temperature", "platt", "isotonic"):
            sub = [r for r in rows if r["model"] == model and r["calibrator"] == cal]
            if not sub:
                continue
            summary[model][cal] = {
                "mean_pred": float(np.mean([r["mean_pred"] for r in sub])),
                "obs_rate": float(np.mean([r["obs_rate"] for r in sub])),
                "pod_0.5": float(np.mean([r["pod_0.5"] for r in sub])),
                "pod_rate_matched": float(np.mean([r["pod_rate_matched"] for r in sub])),
                "pod_at_far": float(
                    np.nanmean([r.get("farfix_pod_at_far", float("nan")) for r in sub])
                ),
                "pod_at_csi_thr": float(
                    np.nanmean([r.get("pod_at_csi_thr", float("nan")) for r in sub])
                )
                if any("pod_at_csi_thr" in r for r in sub)
                else None,
            }

    out = {
        "ri_rates": rates,
        "far_target": args.far_target,
        "note": (
            "POD@rate_matched: alert on top-k scores with k/n = test RI rate. "
            "POD@FAR: threshold with FAR nearest target. "
            "POD@CSI-thr: threshold maximizing CSI on validation (GBM only here)."
        ),
        "rows": rows,
        "seed_mean": summary,
    }
    out_path = paths["metrics"] / f"calibrator_diagnostics_{tag}.json"
    save_json(out, out_path)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
