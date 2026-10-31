from __future__ import annotations

"""Revision metrics: ensemble-first GBM, matched Murphy, clustered AUC, fair single-seed."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def murphy_decomp(y, p, n_bins=10):
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
    recon = unc - res + rel
    return {
        "brier": float(bs),
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "bss": float(1.0 - bs / unc) if unc > 1e-12 else float("nan"),
        "recon_brier": float(recon),
        "recon_minus_brier": float(recon - bs),
    }


def bootstrap_murphy(y, p, storms, n_boot=500, seed=42, n_bins=10):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    storms = np.asarray(storms)
    uniq = np.unique(storms)
    rels, ress, bss = [], [], []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(storms == c) for c in draw])
        d = murphy_decomp(y[idx], p[idx], n_bins=n_bins)
        rels.append(d["reliability"])
        ress.append(d["resolution"])
        bss.append(d["bss"])
    point = murphy_decomp(y, p, n_bins=n_bins)

    def q(vals):
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

    rlo, rhi = q(rels)
    slo, shi = q(ress)
    blo, bhi = q(bss)
    return {
        **point,
        "reliability_lo": rlo,
        "reliability_hi": rhi,
        "resolution_lo": slo,
        "resolution_hi": shi,
        "bss_lo": blo,
        "bss_hi": bhi,
    }


def bootstrap_auc(y, p, storms, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    storms = np.asarray(storms)
    uniq = np.unique(storms)
    rocs, prs = [], []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(storms == c) for c in draw])
        yt, pt = y[idx], p[idx]
        if len(np.unique(yt)) < 2:
            continue
        rocs.append(roc_auc_score(yt, pt))
        prs.append(average_precision_score(yt, pt))
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_lo": float(np.quantile(rocs, 0.025)),
        "roc_hi": float(np.quantile(rocs, 0.975)),
        "pr_lo": float(np.quantile(prs, 0.025)),
        "pr_hi": float(np.quantile(prs, 0.975)),
    }


def _csi(y, p, thr):
    c = binary_counts(y, p, threshold=thr)
    den = c["tp"] + c["fp"] + c["fn"]
    return (c["tp"] / den) if den else float("nan"), c


def boot_det(y, p, thr, storms, n_boot=500, seed=42):
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

    return {
        "threshold": float(thr),
        "csi": summ(csis, csi0),
        "far": summ(fars, c0["far"]),
        "pod": summ(pods, c0["pod"]),
    }


def rate_thr(p, rate):
    k = max(int(round(rate * len(p))), 1)
    return float(np.sort(p)[-k])


def _load(paths, model, seed):
    p = paths["metrics"] / f"predictions_{model}_default_noweight_seed{seed}.csv"
    return pd.read_csv(p)


def main():
    cfg = load_config(ROOT / "configs/default.yaml")
    paths = project_paths(cfg, ROOT)
    seeds = [42, 43, 44]
    n_boot = 500

    table = pd.read_parquet(paths["processed"] / "samples_default.parquet")
    feat_path = paths["processed"] / "features_default.json"
    feature_cols = json.loads(feat_path.read_text(encoding="utf-8"))
    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    test = table[table["split"] == "test"].reset_index(drop=True)
    y_train = train["ri_label"].to_numpy(int)
    y_val = val["ri_label"].to_numpy(float)
    y_test = test["ri_label"].to_numpy(float)
    storms = test["storm_id"].to_numpy()
    rate = float(np.mean(y_test))

    # Basin / year rates
    basin_rates = {}
    if "basin" in table.columns:
        for split in ["train", "val", "test", "all"]:
            sub = table if split == "all" else table[table["split"] == split]
            basin_rates[split] = (
                sub.groupby("basin")["ri_label"].agg(["mean", "sum", "count"]).reset_index().to_dict(orient="records")
            )
    year_col = "year" if "year" in table.columns else None
    if year_col is None and "time" in table.columns:
        table = table.copy()
        table["year"] = pd.to_datetime(table["time"]).dt.year
        test = test.copy()
        test["year"] = pd.to_datetime(test["time"]).dt.year
        year_col = "year"
    year_test = {}
    if year_col:
        g = test.groupby(year_col)["ri_label"].agg(["mean", "count"])
        year_test = {str(int(i)): {"ri_rate": float(r["mean"]), "n": int(r["count"])} for i, r in g.iterrows()}

    # Neural seed-avg + per-seed
    series = {}
    per_seed = {s: {} for s in seeds}
    for key, model, col in (
        ("mlp_raw", "tabular", "prob_raw"),
        ("mlp_iso", "tabular", "prob_isotonic"),
        ("img_raw", "imagery", "prob_raw"),
        ("img_iso", "imagery", "prob_isotonic"),
        ("fus_raw", "fusion", "prob_raw"),
        ("fus_iso", "fusion", "prob_isotonic"),
    ):
        mats = []
        for seed in seeds:
            df = _load(paths, model, seed)
            m = test[["storm_id", "time"]].merge(df, on=["storm_id", "time"], how="inner")
            assert len(m) == len(test)
            arr = m[col].to_numpy(float)
            per_seed[seed][key] = arr
            mats.append(arr)
        series[key] = np.mean(np.stack(mats, 0), 0)

    # GBM deterministic (sklearn defaults) and ens (subsample+sqrt)
    med = np.nanmedian(train[feature_cols].to_numpy(np.float64), 0)
    x_train = _fill(train[feature_cols].to_numpy(np.float64), med)
    x_val = _fill(val[feature_cols].to_numpy(np.float64), med)
    x_test = _fill(test[feature_cols].to_numpy(np.float64), med)

    gbm0 = GradientBoostingClassifier(random_state=42)
    gbm0.fit(x_train, y_train)
    vr0 = gbm0.predict_proba(x_val)[:, 1]
    tr0 = gbm0.predict_proba(x_test)[:, 1]
    gbm_det = {
        "raw": tr0,
        "platt": apply_platt(fit_platt(y_val, vr0), tr0),
        "iso": apply_isotonic(fit_isotonic(y_val, vr0), tr0),
    }

    # Per-seed GBM with SAME hyperparameters as ensemble members (for fair single-seed)
    gbm_seed = {}
    ens_raw, ens_iso, ens_platt = [], [], []
    for seed in seeds:
        gbm = GradientBoostingClassifier(
            random_state=seed,
            subsample=0.8,
            max_features="sqrt",
        )
        gbm.fit(x_train, y_train)
        vr = gbm.predict_proba(x_val)[:, 1]
        tr = gbm.predict_proba(x_test)[:, 1]
        iso = apply_isotonic(fit_isotonic(y_val, vr), tr)
        platt = apply_platt(fit_platt(y_val, vr), tr)
        gbm_seed[seed] = {"raw": tr, "iso": iso, "platt": platt}
        ens_raw.append(tr)
        ens_iso.append(iso)
        ens_platt.append(platt)
    gbm_ens = {
        "raw": np.mean(np.stack(ens_raw, 0), 0),
        "iso": np.mean(np.stack(ens_iso, 0), 0),
        "platt": np.mean(np.stack(ens_platt, 0), 0),
    }

    # Also: default-HP GBM averaged across 3 seeds (seed-only ensemble, no subsample/sqrt)
    default_seed_iso = []
    for seed in seeds:
        g = GradientBoostingClassifier(random_state=seed)
        g.fit(x_train, y_train)
        vr = g.predict_proba(x_val)[:, 1]
        tr = g.predict_proba(x_test)[:, 1]
        default_seed_iso.append(apply_isotonic(fit_isotonic(y_val, vr), tr))
    gbm_seed_only_ens_iso = np.mean(np.stack(default_seed_iso, 0), 0)

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
        print(f"{name} {scope}: {point:+.3f} [{boot['delta_lo']:.3f},{boot['delta_hi']:.3f}]")
        return rec

    overall = np.ones(len(y_test), bool)
    framed = assign_regimes(test.assign(label=y_test))
    cat = framed["regime_intensity"].to_numpy() == "Cat1-2_64-95kt"

    paired = []
    # Lead with ensemble-fair
    paired.append(pair("fus_iso_minus_gbm_ens_iso", series["fus_iso"], gbm_ens["iso"], overall, "overall"))
    paired.append(pair("mlp_iso_minus_gbm_ens_iso", series["mlp_iso"], gbm_ens["iso"], overall, "overall"))
    paired.append(pair("fus_iso_minus_mlp_iso", series["fus_iso"], series["mlp_iso"], overall, "overall"))
    paired.append(pair("img_iso_minus_mlp_iso", series["img_iso"], series["mlp_iso"], overall, "overall"))
    paired.append(pair("fus_raw_minus_gbm_ens_raw", series["fus_raw"], gbm_ens["raw"], overall, "overall"))
    paired.append(pair("mlp_raw_minus_gbm_ens_raw", series["mlp_raw"], gbm_ens["raw"], overall, "overall"))
    # Deterministic as ablation
    paired.append(pair("fus_iso_minus_gbm_det_iso", series["fus_iso"], gbm_det["iso"], overall, "overall"))
    paired.append(pair("fus_iso_minus_gbm_det_iso", series["fus_iso"], gbm_det["iso"], cat, "Cat1-2"))
    paired.append(pair("mlp_iso_minus_gbm_det_iso", series["mlp_iso"], gbm_det["iso"], overall, "overall"))
    # seed-only ens (no HP change)
    paired.append(pair("fus_iso_minus_gbm_seedonly_iso", series["fus_iso"], gbm_seed_only_ens_iso, overall, "overall"))
    paired.append(pair("gbm_ens_iso_minus_gbm_det_iso", gbm_ens["iso"], gbm_det["iso"], overall, "overall"))
    paired.append(pair("gbm_ens_iso_minus_gbm_seedonly_iso", gbm_ens["iso"], gbm_seed_only_ens_iso, overall, "overall"))

    # Fair single-seed: neural seed s vs GBM member seed s (same subsample/sqrt HP)
    single_fair = []
    for seed in seeds:
        for name, a, b in (
            ("fus_iso_minus_gbm_member_iso", per_seed[seed]["fus_iso"], gbm_seed[seed]["iso"]),
            ("mlp_iso_minus_gbm_member_iso", per_seed[seed]["mlp_iso"], gbm_seed[seed]["iso"]),
        ):
            point = float(brier_skill_score(y_test, a) - brier_skill_score(y_test, b))
            boot = bootstrap_delta_bss(y_test, a, b, n_boot=n_boot, clusters=storms)
            rec = {"seed": seed, "contrast": name, "scope": "overall", "point": point, **boot,
                   "bss_a": float(brier_skill_score(y_test, a)), "bss_b": float(brier_skill_score(y_test, b))}
            single_fair.append(rec)
            print(f"seed{seed} {name}: {point:+.3f} [{boot['delta_lo']:.3f},{boot['delta_hi']:.3f}]")

    # Absolute BSS + bar p
    abs_rows = {}
    for name, p in (
        ("mlp_raw", series["mlp_raw"]),
        ("mlp_iso", series["mlp_iso"]),
        ("fus_raw", series["fus_raw"]),
        ("fus_iso", series["fus_iso"]),
        ("img_raw", series["img_raw"]),
        ("img_iso", series["img_iso"]),
        ("gbm_det_raw", gbm_det["raw"]),
        ("gbm_det_iso", gbm_det["iso"]),
        ("gbm_det_platt", gbm_det["platt"]),
        ("gbm_ens_raw", gbm_ens["raw"]),
        ("gbm_ens_iso", gbm_ens["iso"]),
        ("gbm_ens_platt", gbm_ens["platt"]),
        ("gbm_seedonly_iso", gbm_seed_only_ens_iso),
    ):
        boot = bootstrap_bss(y_test, p, n_boot=n_boot, clusters=storms)
        abs_rows[name] = {
            "bss": float(brier_skill_score(y_test, p)),
            "mean_pred": float(np.mean(p)),
            "brier": float(brier_score(y_test, p)),
            **boot,
        }
        print(name, f"BSS={abs_rows[name]['bss']:+.3f} [{boot['bss_lo']:.3f},{boot['bss_hi']:.3f}] pbar={abs_rows[name]['mean_pred']:.3f}")

    # AUC seed-avg vs mean per-seed
    auc = {}
    for name, pkey in (
        ("mlp_raw", "mlp_raw"),
        ("fus_raw", "fus_raw"),
        ("img_raw", "img_raw"),
    ):
        avg = bootstrap_auc(y_test, series[pkey], storms, n_boot=n_boot)
        per = [float(roc_auc_score(y_test, per_seed[s][pkey])) for s in seeds]
        pr_per = [float(average_precision_score(y_test, per_seed[s][pkey])) for s in seeds]
        auc[name] = {
            **avg,
            "mean_per_seed_roc": float(np.mean(per)),
            "per_seed_roc": per,
            "mean_per_seed_pr": float(np.mean(pr_per)),
        }
        print(name, "seed-avg AUC", avg["roc_auc"], "mean-per-seed", np.mean(per), "CI", avg["roc_lo"], avg["roc_hi"])

    for name, p in (
        ("gbm_det_platt", gbm_det["platt"]),
        ("gbm_det_raw", gbm_det["raw"]),
        ("gbm_ens_raw", gbm_ens["raw"]),
        ("gbm_ens_platt", gbm_ens["platt"]),
    ):
        auc[name] = bootstrap_auc(y_test, p, storms, n_boot=n_boot)
        print(name, auc[name]["roc_auc"], auc[name]["roc_lo"], auc[name]["roc_hi"])

    # Murphy matched: raw vs raw, iso vs iso, ens
    murphy = {}
    for name, p in (
        ("mlp_raw", series["mlp_raw"]),
        ("mlp_iso", series["mlp_iso"]),
        ("fus_raw", series["fus_raw"]),
        ("fus_iso", series["fus_iso"]),
        ("gbm_det_raw", gbm_det["raw"]),
        ("gbm_det_iso", gbm_det["iso"]),
        ("gbm_ens_raw", gbm_ens["raw"]),
        ("gbm_ens_iso", gbm_ens["iso"]),
    ):
        murphy[name] = bootstrap_murphy(y_test, p, storms, n_boot=n_boot)
        m = murphy[name]
        print(
            f"MURPHY {name}: BS={m['brier']:.4f} REL={m['reliability']:.4f} "
            f"[{m['reliability_lo']:.4f},{m['reliability_hi']:.4f}] "
            f"RES={m['resolution']:.4f} [{m['resolution_lo']:.4f},{m['resolution_hi']:.4f}] "
            f"UNC={m['uncertainty']:.4f} recon-BS={m['recon_minus_brier']:+.4f}"
        )

    # Detection one row per model (raw ranking for neural/ens; raw for GBM ens)
    detection = []
    for name, p in (
        ("mlp", series["mlp_raw"]),
        ("fusion", series["fus_raw"]),
        ("imagery", series["img_raw"]),
        ("gbm_ens", gbm_ens["raw"]),
        ("gbm_det", gbm_det["raw"]),
    ):
        thr = rate_thr(p, rate)
        det = boot_det(y_test, p, thr, storms, n_boot=n_boot)
        det["model"] = name
        detection.append(det)
        print(f"det {name} CSI={det['csi']['point']:.3f} [{det['csi']['lo']:.3f},{det['csi']['hi']:.3f}]")

    # Ladder: matched isotonic for MLP rungs via reloading checkpoints? Use raw for all ladder MLP
    # from prior finalize clustered, but recompute GBM+MLP with Platt for GBM and raw for MLP consistently
    # Better: use ladder checkpoints for MLP raw seed-avg, GBM with defaults Platt, and fus_iso separately noted
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import RIDataset, fit_feature_scaler
    from src.models.imagery import build_model
    from src.utils import get_device

    device = get_device()
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

    def mlp_seed_avg(rung_name, feats):
        mats = []
        mean, std = fit_feature_scaler(table, feats)
        for seed in seeds:
            ckpt = paths["checkpoints"] / f"tabular_default_ladder_{rung_name}_seed{seed}.pt"
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

    def gbm_platt_avg(feats):
        mats = []
        medf = np.nanmedian(train[feats].to_numpy(np.float64), 0)
        xt = _fill(train[feats].to_numpy(np.float64), medf)
        xv = _fill(val[feats].to_numpy(np.float64), medf)
        xe = _fill(test[feats].to_numpy(np.float64), medf)
        for seed in seeds:
            gbm = GradientBoostingClassifier(random_state=seed)  # defaults: seed-only
            gbm.fit(xt, y_train)
            vr = gbm.predict_proba(xv)[:, 1]
            tr = gbm.predict_proba(xe)[:, 1]
            mats.append(apply_platt(fit_platt(y_val, vr), tr))
        return np.mean(np.stack(mats, 0), 0)

    p_env = mlp_seed_avg("env_only", ENV)
    p_full = mlp_seed_avg("env_plus_ir_scalars", ENV + IR)
    # Also isotonic on ladder MLP using val predictions - need val forward
    def mlp_iso_from_raw(rung_name, feats, p_test_raw):
        mean, std = fit_feature_scaler(table, feats)
        val_ps = []
        for seed in seeds:
            ckpt = paths["checkpoints"] / f"tabular_default_ladder_{rung_name}_seed{seed}.pt"
            model = build_model("tabular", n_features=len(feats))
            try:
                state = torch.load(ckpt, map_location=device, weights_only=False)
            except TypeError:
                state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
            model.to(device)
            model.eval()
            ds = RIDataset(val, None, feats, mean, std)
            loader = DataLoader(ds, batch_size=256, shuffle=False)
            ps = []
            with torch.no_grad():
                for b in loader:
                    logit = model(b["tabular"].to(device), b["image"].to(device))
                    ps.append(torch.sigmoid(logit).cpu().numpy())
            val_ps.append(np.concatenate(ps))
        p_val = np.mean(np.stack(val_ps, 0), 0)
        return apply_isotonic(fit_isotonic(y_val, p_val), p_test_raw)

    ladder = {}
    ladder_pairs = []
    if p_env is not None and p_full is not None:
        p_env_iso = mlp_iso_from_raw("env_only", ENV, p_env)
        p_full_iso = mlp_iso_from_raw("env_plus_ir_scalars", ENV + IR, p_full)
        g_env = gbm_platt_avg(ENV)
        g_full = gbm_platt_avg(ENV + IR)
        # Also GBM iso for matched
        def gbm_iso_avg(feats):
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
                mats.append(apply_isotonic(fit_isotonic(y_val, vr), tr))
            return np.mean(np.stack(mats, 0), 0)

        g_env_iso = gbm_iso_avg(ENV)
        g_full_iso = gbm_iso_avg(ENV + IR)

        for name, p in (
            ("env_mlp_raw", p_env),
            ("full_mlp_raw", p_full),
            ("env_mlp_iso", p_env_iso),
            ("full_mlp_iso", p_full_iso),
            ("env_gbm_iso", g_env_iso),
            ("full_gbm_iso", g_full_iso),
            ("env_gbm_platt", g_env),
            ("full_gbm_platt", g_full),
            ("fus_iso", series["fus_iso"]),
            ("mlp_main_iso", series["mlp_iso"]),
        ):
            boot = bootstrap_bss(y_test, p, n_boot=n_boot, clusters=storms)
            ladder[name] = {"bss": float(brier_skill_score(y_test, p)), "mean_pred": float(np.mean(p)), **boot}
            print(f"ladder {name} {ladder[name]['bss']:+.3f} [{boot['bss_lo']:.3f},{boot['bss_hi']:.3f}]")

        ladder_pairs.append(pair("mlp_iso_full_minus_env", p_full_iso, p_env_iso, overall, "overall"))
        ladder_pairs.append(pair("mlp_raw_full_minus_env", p_full, p_env, overall, "overall"))
        ladder_pairs.append(pair("gbm_iso_full_minus_env", g_full_iso, g_env_iso, overall, "overall"))
        ladder_pairs.append(pair("fus_iso_minus_full_mlp_iso", series["fus_iso"], p_full_iso, overall, "overall"))
        ladder_pairs.append(pair("fus_iso_minus_main_mlp_iso", series["fus_iso"], series["mlp_iso"], overall, "overall"))

    # Sample vs clustered for MLP raw BSS (illustrate)
    sample_boot = bootstrap_bss(y_test, series["mlp_raw"], n_boot=n_boot, clusters=None)
    cluster_boot = bootstrap_bss(y_test, series["mlp_raw"], n_boot=n_boot, clusters=storms)

    out = {
        "features": feature_cols,
        "basin_rates": basin_rates,
        "year_test": year_test,
        "ri_rate_test": rate,
        "abs": abs_rows,
        "paired": paired,
        "single_seed_fair": single_fair,
        "auc": auc,
        "murphy": murphy,
        "detection": detection,
        "ladder": ladder,
        "ladder_pairs": ladder_pairs,
        "sample_vs_clustered_mlp_raw": {"sample": sample_boot, "clustered": cluster_boot},
        "notes": [
            "GBM ensemble uses subsample=0.8 and max_features=sqrt plus three seeds; not seed-count alone.",
            "gbm_seedonly_iso averages three default-HP GBMs (random_state only).",
            "Fair single-seed pairs each neural seed with the GBM ensemble member of the same seed.",
            "AUC seed-avg is on averaged probabilities; mean_per_seed_roc averages per-seed AUCs.",
            "Detection uses raw scores at alert-rate=test RI rate (rank-based; one row per model).",
            "Unweighted controls and GBM ensemble were computed after initial weighted results (post hoc).",
        ],
    }
    out_path = paths["metrics"] / "revision_metrics.json"
    save_json(out, out_path)
    print("saved", out_path)


if __name__ == "__main__":
    main()
