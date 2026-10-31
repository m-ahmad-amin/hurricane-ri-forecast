from __future__ import annotations

"""Compute remaining publishable metrics from saved prediction CSVs (no GPU)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.prepare import TABULAR_FEATURES
from src.eval.calibration import apply_isotonic, apply_platt, fit_isotonic, fit_platt
from src.eval.regimes import assign_regimes
from src.train.metrics import (
    binary_counts,
    bootstrap_bss,
    bootstrap_delta_bss,
    brier_score,
    brier_skill_score,
)
from src.utils import load_config, project_paths, save_json


def _fill(x, med):
    out = x.copy()
    inds = np.where(np.isnan(out))
    out[inds] = np.take(med, inds[1])
    return out


def _csi(y, p, thr):
    c = binary_counts(y, p, threshold=thr)
    den = c["tp"] + c["fp"] + c["fn"]
    return (c["tp"] / den) if den else float("nan"), c


def _best_csi_thr(y_val, p_val, n_grid=201):
    cands = np.unique(
        np.concatenate(
            [
                np.linspace(1e-4, 0.999, n_grid),
                np.quantile(np.asarray(p_val, float), np.linspace(0.01, 0.99, 99)),
            ]
        )
    )
    best_t, best_c = 0.5, -1.0
    for t in cands:
        csi, _ = _csi(y_val, p_val, float(t))
        if np.isfinite(csi) and csi > best_c:
            best_c, best_t = csi, float(t)
    return best_t


def _boot_det(y, p, thr, storms, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    storms = np.asarray(storms)
    uniq = np.unique(storms)
    pods, fars, csis = [], [], []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(storms == c) for c in draw])
        csi, c = _csi(y[idx], p[idx], thr)
        pods.append(c["pod"])
        fars.append(c["far"])
        csis.append(csi)
    csi0, c0 = _csi(y, p, thr)

    def summ(vals, point):
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        return {
            "point": float(point),
            "lo": float(np.quantile(vals, 0.025)) if vals.size else float("nan"),
            "hi": float(np.quantile(vals, 0.975)) if vals.size else float("nan"),
        }

    return {"threshold": thr, "csi": summ(csis, csi0), "far": summ(fars, c0["far"]), "pod": summ(pods, c0["pod"])}


def murphy_decomp(y, p, n_bins=10):
    """Murphy (1973) reliability-resolution-uncertainty for Brier score."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    clim = float(np.mean(y))
    unc = clim * (1.0 - clim)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rel = 0.0
    res = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        nk = int(mask.sum())
        if nk == 0:
            continue
        o_k = float(np.mean(y[mask]))
        f_k = float(np.mean(p[mask]))
        rel += (nk / n) * (f_k - o_k) ** 2
        res += (nk / n) * (o_k - clim) ** 2
    bs = brier_score(y, p)
    return {
        "brier": bs,
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "bss": float(1.0 - bs / unc) if unc > 1e-12 else float("nan"),
        "check": float(unc - res + rel),  # should ≈ brier
    }


def _load(paths, model, name_tag, seed):
    p = paths["metrics"] / f"predictions_{model}_{name_tag}_seed{seed}.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def main():
    cfg = load_config(ROOT / "configs/default.yaml")
    paths = project_paths(cfg, ROOT)
    tag = "default"
    seeds = [42, 43, 44]
    n_boot = 500

    table = pd.read_parquet(paths["processed"] / f"samples_{tag}.parquet")
    feat_path = paths["processed"] / f"features_{tag}.json"
    feature_cols = json.loads(feat_path.read_text(encoding="utf-8")) if feat_path.exists() else TABULAR_FEATURES
    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    test = table[table["split"] == "test"].reset_index(drop=True)
    y_train = train["ri_label"].to_numpy(int)
    y_val = val["ri_label"].to_numpy(float)
    y_test = test["ri_label"].to_numpy(float)
    storms = test["storm_id"].to_numpy()
    framed = assign_regimes(test.assign(label=y_test))
    cat = framed["regime_intensity"].to_numpy() == "Cat1-2_64-95kt"

    # Seed-average unweighted neural probs
    series = {}
    for key, model, col in (
        ("mlp_raw", "tabular", "prob_raw"),
        ("mlp_iso", "tabular", "prob_isotonic"),
        ("img_raw", "imagery", "prob_raw"),
        ("img_iso", "imagery", "prob_isotonic"),
        ("img_pri", "imagery", "prob_primary"),
        ("fus_raw", "fusion", "prob_raw"),
        ("fus_iso", "fusion", "prob_isotonic"),
        ("fus_pri", "fusion", "prob_primary"),
    ):
        mats = []
        for seed in seeds:
            df = _load(paths, model, "default_noweight", seed)
            if df is None:
                raise FileNotFoundError(model, seed)
            m = test[["storm_id", "time"]].merge(df, on=["storm_id", "time"], how="inner")
            assert len(m) == len(test)
            mats.append(m[col].to_numpy(float))
        series[key] = np.mean(np.stack(mats, 0), 0)

    # Single-seed neural raw/iso (for ensemble-vs-single reporting)
    per_seed = {}
    for seed in seeds:
        per_seed[seed] = {}
        for key, model, col in (
            ("mlp_raw", "tabular", "prob_raw"),
            ("mlp_iso", "tabular", "prob_isotonic"),
            ("fus_raw", "fusion", "prob_raw"),
            ("fus_iso", "fusion", "prob_isotonic"),
            ("img_iso", "imagery", "prob_isotonic"),
        ):
            df = _load(paths, model, "default_noweight", seed)
            m = test[["storm_id", "time"]].merge(df, on=["storm_id", "time"], how="inner")
            per_seed[seed][key] = m[col].to_numpy(float)

    # GBM: deterministic default + 3 subsample ensembles
    med = np.nanmedian(train[feature_cols].to_numpy(np.float64), 0)
    x_train = _fill(train[feature_cols].to_numpy(np.float64), med)
    x_val = _fill(val[feature_cols].to_numpy(np.float64), med)
    x_test = _fill(test[feature_cols].to_numpy(np.float64), med)

    gbm_iso_seeds = []
    gbm_platt_seeds = []
    for seed in seeds:
        gbm = GradientBoostingClassifier(
            random_state=seed,
            subsample=0.8,
            max_features="sqrt",
        )
        gbm.fit(x_train, y_train)
        vr = gbm.predict_proba(x_val)[:, 1]
        tr = gbm.predict_proba(x_test)[:, 1]
        gbm_iso_seeds.append(apply_isotonic(fit_isotonic(y_val, vr), tr))
        gbm_platt_seeds.append(apply_platt(fit_platt(y_val, vr), tr))
    gbm_iso_ens = np.mean(np.stack(gbm_iso_seeds, 0), 0)
    gbm_platt_ens = np.mean(np.stack(gbm_platt_seeds, 0), 0)

    # Deterministic GBM (seed 42, no subsample) for comparison
    gbm0 = GradientBoostingClassifier(random_state=42)
    gbm0.fit(x_train, y_train)
    vr0 = gbm0.predict_proba(x_val)[:, 1]
    tr0 = gbm0.predict_proba(x_test)[:, 1]
    gbm_iso_det = apply_isotonic(fit_isotonic(y_val, vr0), tr0)
    gbm_platt_det = apply_platt(fit_platt(y_val, vr0), tr0)

    def pair(name, a, b, mask, scope):
        y = y_test[mask]
        st = storms[mask]
        point = float(brier_skill_score(y, a[mask]) - brier_skill_score(y, b[mask]))
        boot = bootstrap_delta_bss(y, a[mask], b[mask], n_boot=n_boot, clusters=st)
        rec = {
            "contrast": name,
            "scope": scope,
            "point": point,
            "bss_a": float(brier_skill_score(y, a[mask])),
            "bss_b": float(brier_skill_score(y, b[mask])),
            **boot,
        }
        print(
            f"{name} {scope}: {point:+.3f} [{boot['delta_lo']:.3f},{boot['delta_hi']:.3f}] "
            f"(A={rec['bss_a']:+.3f} B={rec['bss_b']:+.3f})"
        )
        return rec

    paired = []
    # MATCHED overall fusion iso - gbm iso (deterministic and ensemble)
    paired.append(pair("fus_iso_minus_gbm_iso_det", series["fus_iso"], gbm_iso_det, np.ones(len(y_test), bool), "overall"))
    paired.append(pair("fus_iso_minus_gbm_iso_ens", series["fus_iso"], gbm_iso_ens, np.ones(len(y_test), bool), "overall"))
    paired.append(pair("fus_iso_minus_mlp_iso", series["fus_iso"], series["mlp_iso"], np.ones(len(y_test), bool), "overall"))
    paired.append(pair("img_iso_minus_mlp_iso", series["img_iso"], series["mlp_iso"], np.ones(len(y_test), bool), "overall"))
    paired.append(pair("img_iso_minus_mlp_iso", series["img_iso"], series["mlp_iso"], cat, "Cat1-2"))
    paired.append(pair("fus_iso_minus_gbm_iso_det", series["fus_iso"], gbm_iso_det, cat, "Cat1-2"))

    # Single-seed matched fusion iso - gbm iso det (same gbm for all; neural varies)
    single_seed_pairs = []
    for seed in seeds:
        y = y_test
        point = float(
            brier_skill_score(y, per_seed[seed]["fus_iso"]) - brier_skill_score(y, gbm_iso_det)
        )
        boot = bootstrap_delta_bss(y, per_seed[seed]["fus_iso"], gbm_iso_det, n_boot=n_boot, clusters=storms)
        rec = {"seed": seed, "contrast": "fus_iso_minus_gbm_iso_det", "scope": "overall", "point": point, **boot}
        single_seed_pairs.append(rec)
        print(
            f"seed{seed} fus_iso-gbm_iso_det overall: {point:+.3f} "
            f"[{boot['delta_lo']:.3f},{boot['delta_hi']:.3f}]"
        )

    # Ladder: env vs env+IR using saved ladder checkpoints? recompute GBM+MLP from feature_ladder json + paired
    ladder = json.loads((paths["metrics"] / "feature_ladder_default.json").read_text(encoding="utf-8"))
    ENV = [
        "sst_c",
        "shear_low_0_800",
        "shear_high_0_800",
        "potential_intensity_kt",
        "pw_mean",
        "distance_to_land_km",
        "storm_speed",
        "intensity_kt",
    ]
    IR = [c for c in feature_cols if c.startswith("ir_")]
    # Rebuild seed-avg MLP probs for env and env+ir by loading ladder checkpoints if present
    # Fallback: use per-seed BSS from JSON and compute paired GBM/MLP on the fly with sklearn only for GBM;
    # for MLP ladder use saved checkpoints
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import RIDataset, fit_feature_scaler
    from src.models.imagery import build_model
    from src.utils import get_device

    device = get_device()

    def mlp_seed_avg(rung_name, feats):
        mats = []
        mean, std = fit_feature_scaler(table, feats)
        for seed in seeds:
            ckpt = paths["checkpoints"] / f"tabular_{tag}_ladder_{rung_name}_seed{seed}.pt"
            if not ckpt.exists():
                return None
            model = build_model("tabular", n_features=len(feats))
            try:
                state = torch.load(ckpt, map_location=device, weights_only=False)
            except TypeError:
                state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
            model.to(device)
            model.eval()
            ds = RIDataset(test, None, feats, mean, std)
            loader = DataLoader(ds, batch_size=256, shuffle=False)
            ps = []
            with torch.no_grad():
                for b in loader:
                    logit = model(b["tabular"].to(device), b["image"].to(device))
                    ps.append(torch.sigmoid(logit).cpu().numpy())
            mats.append(np.concatenate(ps))
        return np.mean(np.stack(mats, 0), 0)

    def gbm_seed_avg(feats):
        mats = []
        medf = np.nanmedian(train[feats].to_numpy(np.float64), 0)
        xt = _fill(train[feats].to_numpy(np.float64), medf)
        xv = _fill(val[feats].to_numpy(np.float64), medf)
        xe = _fill(test[feats].to_numpy(np.float64), medf)
        for seed in seeds:
            gbm = GradientBoostingClassifier(random_state=seed)
            gbm.fit(xt, y_train)
            vr = gbm.predict_proba(xv)[:, 1]
            tr = gbm.predict_proba(xe)[:, 1]
            mats.append(apply_platt(fit_platt(y_val, vr), tr))
        return np.mean(np.stack(mats, 0), 0)

    ladder_out = {"from_json": ladder["rungs"]}
    p_env_mlp = mlp_seed_avg("env_only", ENV)
    p_full_mlp = mlp_seed_avg("env_plus_ir_scalars", ENV + IR)
    p_env_gbm = gbm_seed_avg(ENV)
    p_full_gbm = gbm_seed_avg(ENV + IR)
    ladder_pairs = []
    if p_env_mlp is not None and p_full_mlp is not None:
        ladder_pairs.append(pair("mlp_full_minus_env", p_full_mlp, p_env_mlp, np.ones(len(y_test), bool), "overall"))
        ladder_pairs.append(pair("gbm_full_minus_env", p_full_gbm, p_env_gbm, np.ones(len(y_test), bool), "overall"))
        # +image = fusion iso vs full tabular mlp iso (seed avg already have fus and mlp)
        ladder_pairs.append(pair("fus_iso_minus_mlp_iso_as_image_rung", series["fus_iso"], series["mlp_iso"], np.ones(len(y_test), bool), "overall"))
        for name, p in (
            ("env_mlp", p_env_mlp),
            ("full_mlp", p_full_mlp),
            ("env_gbm", p_env_gbm),
            ("full_gbm", p_full_gbm),
            ("fus_iso", series["fus_iso"]),
            ("mlp_iso", series["mlp_iso"]),
        ):
            boot = bootstrap_bss(y_test, p, n_boot=n_boot, clusters=storms)
            print(f"ladder {name} BSS={brier_skill_score(y_test,p):+.3f} [{boot['bss_lo']:.3f},{boot['bss_hi']:.3f}]")
            ladder_out.setdefault("clustered", {})[name] = {
                "bss": float(brier_skill_score(y_test, p)),
                **boot,
            }

    # Murphy decomp
    murphy = {
        "mlp_u_raw": murphy_decomp(y_test, series["mlp_raw"]),
        "mlp_u_iso": murphy_decomp(y_test, series["mlp_iso"]),
        "gbm_iso_det": murphy_decomp(y_test, gbm_iso_det),
        "gbm_platt_det": murphy_decomp(y_test, gbm_platt_det),
        "fus_iso": murphy_decomp(y_test, series["fus_iso"]),
        "img_iso": murphy_decomp(y_test, series["img_iso"]),
    }
    print("MURPHY")
    for k, v in murphy.items():
        print(k, {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()})

    # AUC
    auc = {}
    for name, p in (
        ("mlp_raw", series["mlp_raw"]),
        ("img_raw", series["img_raw"]),
        ("fus_raw", series["fus_raw"]),
        ("gbm_platt_det", gbm_platt_det),
    ):
        auc[name] = {
            "roc_auc": float(roc_auc_score(y_test, p)),
            "pr_auc": float(average_precision_score(y_test, p)),
        }
        print(name, auc[name])

    # Detection for imagery/fusion using rate-matched? Use CSI thr from val unavailable for img without forward.
    # CSI-maximizing threshold; same rate-matched approach as seed_pooled analysis
    # Better: maximize CSI on validation by reloading noweight checkpoints for imagery/fusion — heavy.
    # Use seed-averaged test probs with threshold from maximizing CSI on validation via saved... 
    # Practical: pick thr maximizing CSI on the *training* labels? No.
    # Use climatology-quantile threshold (alert rate = base rate) AND report clearly.
    rate = float(np.mean(y_test))
    detection = []
    for name, p in (
        ("mlp_raw", series["mlp_raw"]),
        ("mlp_iso", series["mlp_iso"]),
        ("fus_raw", series["fus_raw"]),
        ("fus_iso", series["fus_iso"]),
        ("img_raw", series["img_raw"]),
        ("img_iso", series["img_iso"]),
        ("gbm_iso_det", gbm_iso_det),
        ("gbm_platt_det", gbm_platt_det),
    ):
        k = max(int(round(rate * len(p))), 1)
        thr = float(np.sort(p)[-k])
        det = _boot_det(y_test, p, thr, storms, n_boot=n_boot)
        det["model"] = name
        det["rule"] = "alert_rate_equals_test_ri_rate"
        detection.append(det)
        print(
            f"det {name} thr={thr:.3f} CSI={det['csi']['point']:.3f} "
            f"[{det['csi']['lo']:.3f},{det['csi']['hi']:.3f}] "
            f"FAR={det['far']['point']:.3f} POD={det['pod']['point']:.3f}"
        )

    # Also val-CSI for MLP/GBM using val reload tabular only
    # (already have in seed_pooled; keep rate-matched for all models consistently here)

    out = {
        "paired": paired,
        "single_seed_fus_minus_gbm": single_seed_pairs,
        "ladder_pairs": ladder_pairs,
        "ladder_clustered": ladder_out.get("clustered", {}),
        "murphy": murphy,
        "auc": auc,
        "detection_rate_matched": detection,
        "gbm_iso_det_bss": float(brier_skill_score(y_test, gbm_iso_det)),
        "gbm_iso_ens_bss": float(brier_skill_score(y_test, gbm_iso_ens)),
        "notes": [
            "Matched pairwise claims use isotonic vs isotonic (or raw vs raw).",
            "GBM ensemble uses subsample=0.8, max_features=sqrt, seeds 42/43/44, then averages probs.",
            "Detection table here uses alert-rate = test RI rate for all models (fair across pathways).",
        ],
    }
    out_path = paths["metrics"] / "finalize_paper_metrics.json"
    save_json(out, out_path)
    print("saved", out_path)


if __name__ == "__main__":
    main()
