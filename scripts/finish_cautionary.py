from __future__ import annotations

"""Finish cautionary metrics without reloading imagery for val thresholds."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.calibration import apply_isotonic, apply_platt, fit_isotonic, fit_platt
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


def _load(paths, model, tag, seed):
    p = paths["metrics"] / f"predictions_{model}_{tag}_seed{seed}.csv"
    return pd.read_csv(p)


def _avg(paths, test, model, tag, col, seeds):
    mats = []
    for seed in seeds:
        df = _load(paths, model, tag, seed)
        m = test[["storm_id", "time"]].merge(df, on=["storm_id", "time"], how="inner")
        mats.append(m[col].to_numpy(float))
    return np.mean(np.stack(mats, 0), 0)


def _csi(y, p, thr):
    c = binary_counts(y, p, threshold=thr)
    den = c["tp"] + c["fp"] + c["fn"]
    return (c["tp"] / den) if den else float("nan"), c


def best_thr(y, p):
    cands = np.unique(
        np.concatenate(
            [np.linspace(1e-4, 0.999, 201), np.quantile(np.asarray(p, float), np.linspace(0.01, 0.99, 99))]
        )
    )
    best_t, best_c = 0.5, -1.0
    for t in cands:
        csi, _ = _csi(y, p, float(t))
        if np.isfinite(csi) and csi > best_c:
            best_c, best_t = float(csi), float(t)
    return best_t


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


def abs_row(y, p, storms):
    boot = bootstrap_bss(y, p, n_boot=500, clusters=storms)
    return {
        "bss": float(brier_skill_score(y, p)),
        "mean_pred": float(np.mean(p)),
        "brier": float(brier_score(y, p)),
        **boot,
    }


def main():
    cfg = load_config(ROOT / "configs/default.yaml")
    paths = project_paths(cfg, ROOT)
    seeds = [42, 43, 44]
    table = pd.read_parquet(paths["processed"] / "samples_default.parquet")
    feats = json.loads((paths["processed"] / "features_default.json").read_text())
    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    test = table[table["split"] == "test"].reset_index(drop=True)
    y_train = train["ri_label"].to_numpy(int)
    y_val = val["ri_label"].to_numpy(float)
    y_test = test["ri_label"].to_numpy(float)
    storms = test["storm_id"].to_numpy()
    rate = float(np.mean(y_test))

    series = {
        "mlp_u_raw": _avg(paths, test, "tabular", "default_noweight", "prob_raw", seeds),
        "mlp_u_iso": _avg(paths, test, "tabular", "default_noweight", "prob_isotonic", seeds),
        "img_u_raw": _avg(paths, test, "imagery", "default_noweight", "prob_raw", seeds),
        "img_u_iso": _avg(paths, test, "imagery", "default_noweight", "prob_isotonic", seeds),
        "fus_u_raw": _avg(paths, test, "fusion", "default_noweight", "prob_raw", seeds),
        "fus_u_iso": _avg(paths, test, "fusion", "default_noweight", "prob_isotonic", seeds),
        "img_w_raw": _avg(paths, test, "imagery", "default", "prob_raw", seeds),
        "img_w_iso": _avg(paths, test, "imagery", "default", "prob_isotonic", seeds),
        "fus_w_raw": _avg(paths, test, "fusion", "default", "prob_raw", seeds),
        "fus_w_iso": _avg(paths, test, "fusion", "default", "prob_isotonic", seeds),
        "mlp_w_raw": _avg(paths, test, "tabular", "default", "prob_raw", seeds),
        "mlp_w_iso": _avg(paths, test, "tabular", "default", "prob_isotonic", seeds),
    }

    # IR-only from checkpoint if present
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import RIDataset, fit_feature_scaler
    from src.models.imagery import build_model
    from src.utils import get_device

    device = get_device()
    IR = [c for c in feats if c.startswith("ir_")]
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

    def forward(feats_, ckpt, split):
        mean, std = fit_feature_scaler(table, feats_)
        model = build_model("tabular", n_features=len(feats_))
        try:
            state = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        model.to(device)
        model.eval()
        sub = table[table["split"] == split].reset_index(drop=True)
        ds = RIDataset(sub, None, feats_, mean, std)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        ps = []
        with torch.no_grad():
            for b in loader:
                logit = model(b["tabular"].to(device), b["image"].to(device))
                ps.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(ps)

    ir_ckpts = [paths["checkpoints"] / f"tabular_default_ladder_ir_scalars_only_seed{s}.pt" for s in seeds]
    if all(c.exists() for c in ir_ckpts):
        ir_test = np.mean([forward(IR, c, "test") for c in ir_ckpts], 0)
        ir_val = np.mean([forward(IR, c, "val") for c in ir_ckpts], 0)
        ir_iso = apply_isotonic(fit_isotonic(y_val, ir_val), ir_test)
    else:
        raise FileNotFoundError("IR-only checkpoints missing; re-run cautionary_metrics training step")

    # GBM: val-selected defaults vs post-hoc regularized 0.8/sqrt
    med = np.nanmedian(train[feats].to_numpy(np.float64), 0)
    xt = _fill(train[feats].to_numpy(np.float64), med)
    xv = _fill(val[feats].to_numpy(np.float64), med)
    xe = _fill(test[feats].to_numpy(np.float64), med)

    def gbm_avg(kwargs):
        raws, isos, platts, val_raws, per = [], [], [], [], {}
        for seed in seeds:
            g = GradientBoostingClassifier(random_state=seed, **kwargs)
            g.fit(xt, y_train)
            vr = g.predict_proba(xv)[:, 1]
            tr = g.predict_proba(xe)[:, 1]
            iso = apply_isotonic(fit_isotonic(y_val, vr), tr)
            platt = apply_platt(fit_platt(y_val, vr), tr)
            raws.append(tr)
            isos.append(iso)
            platts.append(platt)
            val_raws.append(vr)
            per[seed] = {"raw": tr, "iso": iso, "platt": platt, "val_raw": vr}
        return {
            "raw": np.mean(np.stack(raws, 0), 0),
            "iso": np.mean(np.stack(isos, 0), 0),
            "platt": np.mean(np.stack(platts, 0), 0),
            "val_raw": np.mean(np.stack(val_raws, 0), 0),
            "per_seed": per,
            "val_brier_mean": float(np.mean([brier_score(y_val, v) for v in val_raws])),
        }

    gbm_default = gbm_avg({})  # val-selected
    gbm_reg = gbm_avg({"subsample": 0.8, "max_features": "sqrt"})  # post-hoc regularized
    g0 = GradientBoostingClassifier(random_state=42)
    g0.fit(xt, y_train)
    vr0 = g0.predict_proba(xv)[:, 1]
    tr0 = g0.predict_proba(xe)[:, 1]
    gbm_det = {
        "raw": tr0,
        "iso": apply_isotonic(fit_isotonic(y_val, vr0), tr0),
        "platt": apply_platt(fit_platt(y_val, vr0), tr0),
        "val_raw": vr0,
    }

    # calibrator pick for default GBM on held-out val half
    rng = np.random.default_rng(0)
    idx = np.arange(len(y_val))
    rng.shuffle(idx)
    fit_i, sc_i = idx[: len(idx) // 2], idx[len(idx) // 2 :]
    vr = gbm_default["val_raw"]
    cal = {
        "raw": float(brier_score(y_val[sc_i], vr[sc_i])),
        "isotonic": float(brier_score(y_val[sc_i], apply_isotonic(fit_isotonic(y_val[fit_i], vr[fit_i]), vr[sc_i]))),
        "platt": float(brier_score(y_val[sc_i], apply_platt(fit_platt(y_val[fit_i], vr[fit_i]), vr[sc_i]))),
    }
    primary_cal = min(cal, key=cal.get)

    # ladder GBM default on env / full
    def gbm_feats(feature_list, kwargs):
        medf = np.nanmedian(train[feature_list].to_numpy(np.float64), 0)
        xtr = _fill(train[feature_list].to_numpy(np.float64), medf)
        xva = _fill(val[feature_list].to_numpy(np.float64), medf)
        xte = _fill(test[feature_list].to_numpy(np.float64), medf)
        mats = []
        for seed in seeds:
            g = GradientBoostingClassifier(random_state=seed, **kwargs)
            g.fit(xtr, y_train)
            vr = g.predict_proba(xva)[:, 1]
            tr = g.predict_proba(xte)[:, 1]
            mats.append(apply_isotonic(fit_isotonic(y_val, vr), tr))
        return np.mean(np.stack(mats, 0), 0)

    env_gbm = gbm_feats(ENV, {})
    full_gbm = gbm_feats(ENV + IR, {})
    env_gbm_reg = gbm_feats(ENV, {"subsample": 0.8, "max_features": "sqrt"})
    full_gbm_reg = gbm_feats(ENV + IR, {"subsample": 0.8, "max_features": "sqrt"})

    abs_rows = {}
    for name, p in {
        **series,
        "ir_only_iso": ir_iso,
        "ir_only_raw": ir_test,
        "gbm_default_avg_iso": gbm_default["iso"],
        "gbm_default_avg_raw": gbm_default["raw"],
        "gbm_default_avg_platt": gbm_default["platt"],
        "gbm_reg_iso": gbm_reg["iso"],
        "gbm_reg_raw": gbm_reg["raw"],
        "gbm_reg_platt": gbm_reg["platt"],
        "gbm_det_iso": gbm_det["iso"],
        "gbm_det_raw": gbm_det["raw"],
        "gbm_det_platt": gbm_det["platt"],
        "env_gbm_default_iso": env_gbm,
        "full_gbm_default_iso": full_gbm,
        "env_gbm_reg_iso": env_gbm_reg,
        "full_gbm_reg_iso": full_gbm_reg,
    }.items():
        abs_rows[name] = abs_row(y_test, p, storms)
        print(name, f"{abs_rows[name]['bss']:+.3f}", f"[{abs_rows[name]['bss_lo']:.3f},{abs_rows[name]['bss_hi']:.3f}]")

    def pair(name, a, b, mask=None, scope="overall"):
        if mask is None:
            mask = np.ones(len(y_test), bool)
        y, st = y_test[mask], storms[mask]
        aa, bb = a[mask], b[mask]
        point = float(brier_skill_score(y, aa) - brier_skill_score(y, bb))
        boot = bootstrap_delta_bss(y, aa, bb, n_boot=500, clusters=st)
        rec = {"contrast": name, "scope": scope, "point": point, **boot}
        print(name, scope, f"{point:+.3f} [{boot['delta_lo']:.3f},{boot['delta_hi']:.3f}]")
        return rec

    from src.eval.regimes import assign_regimes

    cat = assign_regimes(test.assign(label=y_test))["regime_intensity"].to_numpy() == "Cat1-2_64-95kt"
    paired = [
        pair("fus_u_iso_minus_mlp_u_iso", series["fus_u_iso"], series["mlp_u_iso"]),
        pair("fus_u_iso_minus_gbm_default_iso", series["fus_u_iso"], gbm_default["iso"]),
        pair("mlp_u_iso_minus_gbm_default_iso", series["mlp_u_iso"], gbm_default["iso"]),
        pair("fus_u_iso_minus_gbm_reg_iso", series["fus_u_iso"], gbm_reg["iso"]),
        pair("mlp_u_iso_minus_gbm_reg_iso", series["mlp_u_iso"], gbm_reg["iso"]),
        pair("img_u_iso_minus_mlp_u_iso", series["img_u_iso"], series["mlp_u_iso"]),
        pair("img_u_iso_minus_ir_only_iso", series["img_u_iso"], ir_iso),
        pair("ir_only_iso_minus_img_u_iso", ir_iso, series["img_u_iso"]),
        pair("gbm_reg_iso_minus_gbm_default_iso", gbm_reg["iso"], gbm_default["iso"]),
        pair("img_u_iso_minus_mlp_u_iso", series["img_u_iso"], series["mlp_u_iso"], cat, "Cat1-2"),
    ]

    # single-seed vs default member AND vs regularized member
    single = []
    for seed in seeds:
        mf = test[["storm_id", "time"]].merge(_load(paths, "fusion", "default_noweight", seed), on=["storm_id", "time"])
        mm = test[["storm_id", "time"]].merge(_load(paths, "tabular", "default_noweight", seed), on=["storm_id", "time"])
        fus = mf["prob_isotonic"].to_numpy(float)
        mlp = mm["prob_isotonic"].to_numpy(float)
        for label, gbm_s in (
            ("default", gbm_default["per_seed"][seed]["iso"]),
            ("regularized", gbm_reg["per_seed"][seed]["iso"]),
        ):
            for cname, a in ((f"fus_iso_minus_gbm_{label}_member", fus), (f"mlp_iso_minus_gbm_{label}_member", mlp)):
                point = float(brier_skill_score(y_test, a) - brier_skill_score(y_test, gbm_s))
                boot = bootstrap_delta_bss(y_test, a, gbm_s, n_boot=500, clusters=storms)
                single.append({"seed": seed, "contrast": cname, "point": point, **boot})
                print(f"seed{seed} {cname}: {point:+.3f} [{boot['delta_lo']:.3f},{boot['delta_hi']:.3f}]")

    # detection: rate-matched + val CSI for tabular/GBM only (no image reload)
    # For fusion/imagery val CSI: use test-rate as diagnostic only; for mlp/gbm use val
    # Approximate fusion/imagery val threshold from temperature? Better: use seed-avg raw from
    # prediction files - they don't include val. Use rate-matched for all + val for mlp/gbm.
    detection = {"rate_matched": [], "val_csi": []}
    for name, p in (
        ("mlp", series["mlp_u_raw"]),
        ("fusion", series["fus_u_raw"]),
        ("imagery", series["img_u_raw"]),
        ("gbm_default", gbm_default["raw"]),
        ("gbm_reg", gbm_reg["raw"]),
        ("gbm_det", gbm_det["raw"]),
        ("ir_only", ir_test),
    ):
        thr = float(np.sort(p)[-max(int(round(rate * len(p))), 1)])
        det = boot_det(y_test, p, thr, storms)
        det.update({"model": name, "rule": "alert_rate_equals_test_ri_rate"})
        detection["rate_matched"].append(det)

    # val CSI for mlp via checkpoint forward
    mlp_val = np.mean(
        [forward(feats, paths["checkpoints"] / f"tabular_default_noweight_seed{s}.pt", "val") for s in seeds], 0
    )
    for name, p_test, p_val in (
        ("mlp", series["mlp_u_raw"], mlp_val),
        ("gbm_default", gbm_default["raw"], gbm_default["val_raw"]),
        ("gbm_reg", gbm_reg["raw"], gbm_reg["val_raw"]),
        ("gbm_det", gbm_det["raw"], gbm_det["val_raw"]),
        ("ir_only", ir_test, ir_val),
    ):
        thr = best_thr(y_val, p_val)
        det = boot_det(y_test, p_test, thr, storms)
        det.update({"model": name, "rule": "max_csi_on_validation"})
        detection["val_csi"].append(det)
        print("val det", name, thr, det["csi"]["point"])

    sample_vs = {
        "mlp_u_iso_sample": bootstrap_bss(y_test, series["mlp_u_iso"], n_boot=500, clusters=None),
        "mlp_u_iso_clustered": bootstrap_bss(y_test, series["mlp_u_iso"], n_boot=500, clusters=storms),
        "fus_u_iso_sample": bootstrap_bss(y_test, series["fus_u_iso"], n_boot=500, clusters=None),
        "fus_u_iso_clustered": bootstrap_bss(y_test, series["fus_u_iso"], n_boot=500, clusters=storms),
    }
    by_year = {}
    for season, _ in test.groupby("season"):
        mask = test["season"].to_numpy() == season
        by_year[str(int(season))] = {
            "n": int(mask.sum()),
            "n_ri": int(y_test[mask].sum()),
            "ri_rate": float(np.mean(y_test[mask])),
            "mlp_u_iso": float(brier_skill_score(y_test[mask], series["mlp_u_iso"][mask])),
            "fus_u_iso": float(brier_skill_score(y_test[mask], series["fus_u_iso"][mask])),
            "gbm_default_iso": float(brier_skill_score(y_test[mask], gbm_default["iso"][mask])),
            "gbm_reg_iso": float(brier_skill_score(y_test[mask], gbm_reg["iso"][mask])),
            "img_u_iso": float(brier_skill_score(y_test[mask], series["img_u_iso"][mask])),
            "ir_only_iso": float(brier_skill_score(y_test[mask], ir_iso[mask])),
        }

    out = {
        "n_test": int(len(y_test)),
        "n_storms": int(pd.Series(storms).nunique()),
        "n_ri": int(y_test.sum()),
        "ri_rate": rate,
        "gbm_grid_note": "Defaults (subsample=1, max_features=None) win raw validation Brier (0.04009) over subsample=0.8/max_features=sqrt (0.04039).",
        "primary_gbm": "three-seed average of sklearn-default GBM (validation-selected)",
        "regularized_gbm": "post-hoc subsample=0.8, max_features=sqrt; better on test BSS but worse on validation Brier",
        "calibrator_val_scores": cal,
        "primary_calibrator": primary_cal,
        "abs": abs_rows,
        "paired": paired,
        "single_seed": single,
        "detection": detection,
        "sample_vs_clustered": sample_vs,
        "by_year": by_year,
    }
    out_path = paths["metrics"] / "cautionary_metrics.json"
    save_json(out, out_path)
    print("saved", out_path)


if __name__ == "__main__":
    main()
