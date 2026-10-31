from __future__ import annotations

"""Cautionary-revision metrics: regularized GBM via val grid, IR-scalars MLP, weighted artifact rows."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import RIDataset, fit_feature_scaler
from src.data.prepare import TABULAR_FEATURES
from src.eval.calibration import apply_isotonic, apply_platt, fit_isotonic, fit_platt
from src.models.imagery import build_model
from src.train.loop import train_model
from src.train.metrics import (
    binary_counts,
    bootstrap_bss,
    bootstrap_delta_bss,
    brier_score,
    brier_skill_score,
)
from src.utils import get_device, load_config, project_paths, save_json, set_seed


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
IR = [c for c in TABULAR_FEATURES if c.startswith("ir_")]


def _fill(x, med):
    out = x.copy()
    inds = np.where(np.isnan(out))
    out[inds] = np.take(med, inds[1])
    return out


def _load_pred(paths, model, name_tag, seed):
    for suffix in (f"predictions_{model}_{name_tag}_seed{seed}.csv",
                   f"predictions_{model}_{name_tag}_seed{seed}_with_platt.csv"):
        p = paths["metrics"] / suffix
        if p.exists():
            return pd.read_csv(p)
    return None


def _seed_avg(paths, test, model, name_tag, col, seeds):
    mats = []
    for seed in seeds:
        df = _load_pred(paths, model, name_tag, seed)
        if df is None:
            raise FileNotFoundError(model, name_tag, seed)
        m = test[["storm_id", "time"]].merge(df, on=["storm_id", "time"], how="inner")
        assert len(m) == len(test), (model, name_tag, seed, len(m), len(test))
        mats.append(m[col].to_numpy(float))
    return np.mean(np.stack(mats, 0), 0)


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


def rate_thr(p, rate):
    k = max(int(round(rate * len(p))), 1)
    return float(np.sort(p)[-k])


def abs_row(y, p, storms, n_boot=500):
    boot = bootstrap_bss(y, p, n_boot=n_boot, clusters=storms)
    return {
        "bss": float(brier_skill_score(y, p)),
        "mean_pred": float(np.mean(p)),
        "brier": float(brier_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        **boot,
    }


def pair(name, a, b, y, storms, scope="overall"):
    point = float(brier_skill_score(y, a) - brier_skill_score(y, b))
    boot = bootstrap_delta_bss(y, a, b, n_boot=500, clusters=storms)
    rec = {"contrast": name, "scope": scope, "point": point,
           "bss_a": float(brier_skill_score(y, a)), "bss_b": float(brier_skill_score(y, b)), **boot}
    print(f"{name} {scope}: {point:+.3f} [{boot['delta_lo']:.3f},{boot['delta_hi']:.3f}]")
    return rec


def mlp_forward(table, feats, ckpt, split, device):
    mean, std = fit_feature_scaler(table, feats)
    model = build_model("tabular", n_features=len(feats))
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    sub = table[table["split"] == split].reset_index(drop=True)
    ds = RIDataset(sub, None, feats, mean, std)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    ps = []
    with torch.no_grad():
        for b in loader:
            logit = model(b["tabular"].to(device), b["image"].to(device))
            ps.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(ps)


def train_ir_only(table, cfg, paths, seeds, device):
    """Train IR-scalars-only unweighted MLP if checkpoints missing."""
    feats = IR
    outs = []
    for seed in seeds:
        ckpt = paths["checkpoints"] / f"tabular_default_ladder_ir_scalars_only_seed{seed}.pt"
        if not ckpt.exists():
            print(f"training IR-scalars-only seed {seed}")
            set_seed(seed)
            mean, std = fit_feature_scaler(table, feats)
            loaders = {}
            for split in ("train", "val", "test"):
                mask = table["split"].to_numpy() == split
                ds = RIDataset(table.loc[mask], None, feats, mean, std)
                loaders[split] = DataLoader(
                    ds, batch_size=cfg["train"]["batch_size"], shuffle=(split == "train")
                )
            model = build_model("tabular", n_features=len(feats))
            train_model(
                model,
                loaders,
                device=device,
                epochs=cfg["train"]["epochs"],
                lr=cfg["train"]["lr"],
                weight_decay=cfg["train"]["weight_decay"],
                checkpoint_path=ckpt,
                patience=cfg["train"].get("patience", 8),
                min_epochs=cfg["train"].get("min_epochs", 5),
                use_pos_weight=False,
            )
        outs.append(ckpt)
    return outs


def main():
    cfg = load_config(ROOT / "configs/default.yaml")
    paths = project_paths(cfg, ROOT)
    seeds = [42, 43, 44]
    device = get_device()
    table = pd.read_parquet(paths["processed"] / "samples_default.parquet")
    feature_cols = json.loads((paths["processed"] / "features_default.json").read_text())
    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    test = table[table["split"] == "test"].reset_index(drop=True)
    y_train = train["ri_label"].to_numpy(int)
    y_val = val["ri_label"].to_numpy(float)
    y_test = test["ri_label"].to_numpy(float)
    storms = test["storm_id"].to_numpy()
    n_storms = int(pd.Series(storms).nunique())
    n_ri = int(y_test.sum())
    rate = float(np.mean(y_test))

    # --- neural seed-avg ---
    series = {
        "mlp_u_raw": _seed_avg(paths, test, "tabular", "default_noweight", "prob_raw", seeds),
        "mlp_u_iso": _seed_avg(paths, test, "tabular", "default_noweight", "prob_isotonic", seeds),
        "img_u_raw": _seed_avg(paths, test, "imagery", "default_noweight", "prob_raw", seeds),
        "img_u_iso": _seed_avg(paths, test, "imagery", "default_noweight", "prob_isotonic", seeds),
        "fus_u_raw": _seed_avg(paths, test, "fusion", "default_noweight", "prob_raw", seeds),
        "fus_u_iso": _seed_avg(paths, test, "fusion", "default_noweight", "prob_isotonic", seeds),
        "img_w_raw": _seed_avg(paths, test, "imagery", "default", "prob_raw", seeds),
        "img_w_iso": _seed_avg(paths, test, "imagery", "default", "prob_isotonic", seeds),
        "fus_w_raw": _seed_avg(paths, test, "fusion", "default", "prob_raw", seeds),
        "fus_w_iso": _seed_avg(paths, test, "fusion", "default", "prob_isotonic", seeds),
        "mlp_w_raw": _seed_avg(paths, test, "tabular", "default", "prob_raw", seeds),
        "mlp_w_iso": _seed_avg(paths, test, "tabular", "default", "prob_isotonic", seeds),
    }
    # val seed-avg for detection thresholds
    val_frame = val.reset_index(drop=True)
    # merge needs time - val may have time
    series_val = {}
    for key, model, tag, col in (
        ("mlp_u_raw", "tabular", "default_noweight", "prob_raw"),
        ("fus_u_raw", "fusion", "default_noweight", "prob_raw"),
        ("img_u_raw", "imagery", "default_noweight", "prob_raw"),
    ):
        mats = []
        for seed in seeds:
            df = _load_pred(paths, model, tag, seed)
            # predictions are test-only usually - check
            if "split" in df.columns:
                dfv = df  # unlikely
            # val probs may not be in prediction CSVs - rebuild from checkpoints if needed
            mats.append(None)
        series_val[key] = None

    # --- GBM validation grid (choose by mean val Brier across seeds, raw probs) ---
    med = np.nanmedian(train[feature_cols].to_numpy(np.float64), 0)
    x_train = _fill(train[feature_cols].to_numpy(np.float64), med)
    x_val = _fill(val[feature_cols].to_numpy(np.float64), med)
    x_test = _fill(test[feature_cols].to_numpy(np.float64), med)

    grid = []
    for subsample in (1.0, 0.8, 0.6):
        for max_features in (None, "sqrt"):
            for max_depth in (None, 3):
                val_briers = []
                test_iso_mats = []
                test_raw_mats = []
                test_platt_mats = []
                val_raw_mats = []
                per_seed = {}
                for seed in seeds:
                    kwargs = {"random_state": seed, "subsample": subsample}
                    if max_features is not None:
                        kwargs["max_features"] = max_features
                    if max_depth is not None:
                        kwargs["max_depth"] = max_depth
                    gbm = GradientBoostingClassifier(**kwargs)
                    gbm.fit(x_train, y_train)
                    vr = gbm.predict_proba(x_val)[:, 1]
                    tr = gbm.predict_proba(x_test)[:, 1]
                    val_briers.append(brier_score(y_val, vr))
                    val_raw_mats.append(vr)
                    test_raw_mats.append(tr)
                    test_iso_mats.append(apply_isotonic(fit_isotonic(y_val, vr), tr))
                    test_platt_mats.append(apply_platt(fit_platt(y_val, vr), tr))
                    per_seed[seed] = {"raw": tr, "iso": test_iso_mats[-1], "platt": test_platt_mats[-1], "val_raw": vr}
                rec = {
                    "subsample": subsample,
                    "max_features": max_features if max_features else "None",
                    "max_depth": max_depth if max_depth else "None",
                    "val_brier_mean": float(np.mean(val_briers)),
                    "val_brier_per_seed": [float(x) for x in val_briers],
                }
                grid.append(rec)
                print(
                    f"grid subsample={subsample} max_features={rec['max_features']} "
                    f"max_depth={rec['max_depth']}: valBrier={rec['val_brier_mean']:.5f}"
                )
                # stash probs under key
                rec["_probs"] = {
                    "raw": np.mean(np.stack(test_raw_mats, 0), 0),
                    "iso": np.mean(np.stack(test_iso_mats, 0), 0),
                    "platt": np.mean(np.stack(test_platt_mats, 0), 0),
                    "val_raw": np.mean(np.stack(val_raw_mats, 0), 0),
                    "per_seed": per_seed,
                }

    grid_sorted = sorted(grid, key=lambda r: r["val_brier_mean"])
    best = grid_sorted[0]
    print("BEST by val Brier:", {k: best[k] for k in best if not k.startswith("_")})

    # calibrator choice for best HP by validation Brier
    # Need val calibrated probs: fit calibrator on val using... can't use same val for fit and score.
    # Protocol: for calibrator selection, use raw val Brier vs leave-as-raw; for iso/platt
    # we report test after fitting on full val (standard). Choose primary calibrator by
    # comparing raw val Brier to... Actually common: fit calibrator on val, apply to test only;
    # select among {raw, iso, platt} by which has best *validation* score when calibrator
    # Fit calibrator on held-out validation logits:
    # Select HP by raw val Brier. Then among raw/iso/platt on TEST we report all three,
    # and designate primary as the one with lowest *validation* Brier where iso/platt are
    # fit with a 50/50 val split.
    rng = np.random.default_rng(0)
    val_idx = np.arange(len(y_val))
    rng.shuffle(val_idx)
    half = len(val_idx) // 2
    fit_idx, score_idx = val_idx[:half], val_idx[half:]
    # rebuild best HP single seed-avg on val for calibrator pick using seed 42 member only for speed
    # Use seed-averaged val raw from best
    vr = best["_probs"]["val_raw"]
    # For seed-avg val raw we already have. Split for calibrator:
    cal_scores = {}
    cal_scores["raw"] = float(brier_score(y_val[score_idx], vr[score_idx]))
    iso_m = fit_isotonic(y_val[fit_idx], vr[fit_idx])
    platt_m = fit_platt(y_val[fit_idx], vr[fit_idx])
    cal_scores["isotonic"] = float(brier_score(y_val[score_idx], apply_isotonic(iso_m, vr[score_idx])))
    cal_scores["platt"] = float(brier_score(y_val[score_idx], apply_platt(platt_m, vr[score_idx])))
    primary_cal = min(cal_scores, key=cal_scores.get)
    print("calibrator val scores (held-out half):", cal_scores, "primary=", primary_cal)

    gbm_reg = best["_probs"]  # regularized (val-selected)
    # default 3-seed average (subsample=1, max_features=None, max_depth=None)
    default_cfg = next(
        r for r in grid
        if r["subsample"] == 1.0 and r["max_features"] == "None" and r["max_depth"] == "None"
    )
    gbm_default_avg = default_cfg["_probs"]
    # deterministic single default
    gbm0 = GradientBoostingClassifier(random_state=42)
    gbm0.fit(x_train, y_train)
    vr0 = gbm0.predict_proba(x_val)[:, 1]
    tr0 = gbm0.predict_proba(x_test)[:, 1]
    gbm_det = {
        "raw": tr0,
        "iso": apply_isotonic(fit_isotonic(y_val, vr0), tr0),
        "platt": apply_platt(fit_platt(y_val, vr0), tr0),
    }

    # Also keep the previously used 0.8/sqrt regardless of winner for disclosure
    prev = next(
        r for r in grid
        if r["subsample"] == 0.8 and r["max_features"] == "sqrt" and r["max_depth"] == "None"
    )

    # --- IR-scalars-only MLP ---
    ir_ckpts = train_ir_only(table, cfg, paths, seeds, device)
    ir_test = np.mean([mlp_forward(table, IR, c, "test", device) for c in ir_ckpts], 0)
    ir_val = np.mean([mlp_forward(table, IR, c, "val", device) for c in ir_ckpts], 0)
    ir_iso = apply_isotonic(fit_isotonic(y_val, ir_val), ir_test)

    # ladder MLP env / full from checkpoints
    def ladder_mlp(rung, feats):
        mats_t, mats_v = [], []
        for seed in seeds:
            ckpt = paths["checkpoints"] / f"tabular_default_ladder_{rung}_seed{seed}.pt"
            mats_t.append(mlp_forward(table, feats, ckpt, "test", device))
            mats_v.append(mlp_forward(table, feats, ckpt, "val", device))
        pt = np.mean(np.stack(mats_t, 0), 0)
        pv = np.mean(np.stack(mats_v, 0), 0)
        return pt, apply_isotonic(fit_isotonic(y_val, pv), pt)

    env_raw, env_iso = ladder_mlp("env_only", ENV)
    full_raw, full_iso = ladder_mlp("env_plus_ir_scalars", ENV + IR)

    # regularized GBM on feature subsets (same HP as best overall)
    def gbm_reg_feats(feats):
        medf = np.nanmedian(train[feats].to_numpy(np.float64), 0)
        xt = _fill(train[feats].to_numpy(np.float64), medf)
        xv = _fill(val[feats].to_numpy(np.float64), medf)
        xe = _fill(test[feats].to_numpy(np.float64), medf)
        kwargs_base = {"subsample": best["subsample"]}
        if best["max_features"] != "None":
            kwargs_base["max_features"] = best["max_features"]
        if best["max_depth"] != "None":
            kwargs_base["max_depth"] = int(best["max_depth"])
        mats_iso, mats_raw = [], []
        for seed in seeds:
            gbm = GradientBoostingClassifier(random_state=seed, **kwargs_base)
            gbm.fit(xt, y_train)
            vr = gbm.predict_proba(xv)[:, 1]
            tr = gbm.predict_proba(xe)[:, 1]
            mats_raw.append(tr)
            mats_iso.append(apply_isotonic(fit_isotonic(y_val, vr), tr))
        return np.mean(np.stack(mats_raw, 0), 0), np.mean(np.stack(mats_iso, 0), 0)

    env_g_raw, env_g_iso = gbm_reg_feats(ENV)
    full_g_raw, full_g_iso = gbm_reg_feats(ENV + IR)

    # --- absolute rows ---
    abs_rows = {}
    for name, p in {
        **{k: series[k] for k in series},
        "ir_only_raw": ir_test,
        "ir_only_iso": ir_iso,
        "env_mlp_iso": env_iso,
        "full_mlp_iso": full_iso,
        "env_mlp_raw": env_raw,
        "full_mlp_raw": full_raw,
        "gbm_reg_raw": gbm_reg["raw"],
        "gbm_reg_iso": gbm_reg["iso"],
        "gbm_reg_platt": gbm_reg["platt"],
        "gbm_default_avg_raw": gbm_default_avg["raw"],
        "gbm_default_avg_iso": gbm_default_avg["iso"],
        "gbm_det_iso": gbm_det["iso"],
        "gbm_det_platt": gbm_det["platt"],
        "env_gbm_reg_iso": env_g_iso,
        "full_gbm_reg_iso": full_g_iso,
        "prev_08_sqrt_iso": prev["_probs"]["iso"],
    }.items():
        abs_rows[name] = abs_row(y_test, p, storms)
        print(f"ABS {name}: BSS={abs_rows[name]['bss']:+.3f} [{abs_rows[name]['bss_lo']:.3f},{abs_rows[name]['bss_hi']:.3f}] "
              f"pbar={abs_rows[name]['mean_pred']:.3f} AUC={abs_rows[name]['roc_auc']:.3f}")

    # --- pairs ---
    paired = []
    paired.append(pair("fus_u_iso_minus_mlp_u_iso", series["fus_u_iso"], series["mlp_u_iso"], y_test, storms))
    paired.append(pair("fus_u_iso_minus_gbm_reg_iso", series["fus_u_iso"], gbm_reg["iso"], y_test, storms))
    paired.append(pair("mlp_u_iso_minus_gbm_reg_iso", series["mlp_u_iso"], gbm_reg["iso"], y_test, storms))
    paired.append(pair("img_u_iso_minus_mlp_u_iso", series["img_u_iso"], series["mlp_u_iso"], y_test, storms))
    paired.append(pair("img_u_iso_minus_ir_only_iso", series["img_u_iso"], ir_iso, y_test, storms))
    paired.append(pair("ir_only_iso_minus_img_u_iso", ir_iso, series["img_u_iso"], y_test, storms))
    paired.append(pair("mlp_iso_full_minus_env", full_iso, env_iso, y_test, storms))
    paired.append(pair("fus_u_iso_minus_full_mlp_iso", series["fus_u_iso"], full_iso, y_test, storms))
    paired.append(pair("gbm_reg_iso_minus_gbm_default_avg_iso", gbm_reg["iso"], gbm_default_avg["iso"], y_test, storms))
    paired.append(pair("gbm_reg_iso_minus_gbm_det_iso", gbm_reg["iso"], gbm_det["iso"], y_test, storms))

    # Cat1-2 borderline
    from src.eval.regimes import assign_regimes
    framed = assign_regimes(test.assign(label=y_test))
    cat = framed["regime_intensity"].to_numpy() == "Cat1-2_64-95kt"
    paired.append(pair("img_u_iso_minus_mlp_u_iso", series["img_u_iso"][cat], series["mlp_u_iso"][cat], y_test[cat], storms[cat], "Cat1-2"))

    # single-seed fair: neural vs same-seed regularized GBM member
    single = []
    for seed in seeds:
        df_f = _load_pred(paths, "fusion", "default_noweight", seed)
        df_m = _load_pred(paths, "tabular", "default_noweight", seed)
        mf = test[["storm_id", "time"]].merge(df_f, on=["storm_id", "time"], how="inner")
        mm = test[["storm_id", "time"]].merge(df_m, on=["storm_id", "time"], how="inner")
        fus_iso_s = mf["prob_isotonic"].to_numpy(float)
        mlp_iso_s = mm["prob_isotonic"].to_numpy(float)
        gbm_iso_s = best["_probs"]["per_seed"][seed]["iso"]
        for name, a in (("fus_iso_minus_gbm_reg_member", fus_iso_s), ("mlp_iso_minus_gbm_reg_member", mlp_iso_s)):
            point = float(brier_skill_score(y_test, a) - brier_skill_score(y_test, gbm_iso_s))
            boot = bootstrap_delta_bss(y_test, a, gbm_iso_s, n_boot=500, clusters=storms)
            rec = {"seed": seed, "contrast": name, "point": point, **boot}
            single.append(rec)
            print(f"seed{seed} {name}: {point:+.3f} [{boot['delta_lo']:.3f},{boot['delta_hi']:.3f}]")

    # detection: rate-matched diagnostic + val CSI threshold
    # Need val raw probs for mlp/fus/img/gbm_reg
    def val_raw_neural(model, tag):
        # predictions CSVs are typically test-only; reload checkpoints
        mats = []
        for seed in seeds:
            if model == "tabular":
                ckpt = paths["checkpoints"] / f"tabular_{tag}_seed{seed}.pt"
                # noweight tag in filename
                if tag == "default_noweight":
                    ckpt = paths["checkpoints"] / f"tabular_default_noweight_seed{seed}.pt"
                elif tag == "default":
                    ckpt = paths["checkpoints"] / f"tabular_default_seed{seed}.pt"
                mats.append(mlp_forward(table, feature_cols, ckpt, "val", device))
            else:
                # imagery/fusion: use saved if available else skip
                ckpt = paths["checkpoints"] / f"{model}_{tag}_seed{seed}.pt"
                if not ckpt.exists() and tag == "default_noweight":
                    ckpt = paths["checkpoints"] / f"{model}_default_noweight_seed{seed}.pt"
                if not ckpt.exists():
                    return None
                # need image path
                from src.data.dataset import RIDataset as RDS
                mean, std = fit_feature_scaler(table, feature_cols)
                images = np.load(paths["processed"] / "images_default.npy", mmap_mode="r")
                # map val indices
                # RIDataset with images: need index alignment - use table index
                # Simpler: read from eval json mean - skip if heavy
                try:
                    model_nn = build_model(model, n_features=len(feature_cols))
                    try:
                        state = torch.load(ckpt, map_location=device, weights_only=False)
                    except TypeError:
                        state = torch.load(ckpt, map_location=device)
                    model_nn.load_state_dict(state["model"])
                    model_nn.to(device)
                    model_nn.eval()
                    # build val subset indices from full table order used in image npy
                    full = table.reset_index(drop=True)
                    assert len(full) == len(images)
                    val_mask = full["split"].to_numpy() == "val"
                    mean_t, std_t = mean, std
                    ds = RDS(full.loc[val_mask], images[val_mask], feature_cols, mean_t, std_t)
                    loader = DataLoader(ds, batch_size=64, shuffle=False)
                    ps = []
                    with torch.no_grad():
                        for b in loader:
                            logit = model_nn(b["tabular"].to(device), b["image"].to(device))
                            ps.append(torch.sigmoid(logit).cpu().numpy())
                    mats.append(np.concatenate(ps))
                except Exception as e:
                    print("val forward failed", model, seed, e)
                    return None
        return np.mean(np.stack(mats, 0), 0)

    detection = {"rate_matched": [], "val_csi": []}
    det_models = [
        ("mlp", series["mlp_u_raw"], "tabular", "default_noweight"),
        ("fusion", series["fus_u_raw"], "fusion", "default_noweight"),
        ("imagery", series["img_u_raw"], "imagery", "default_noweight"),
        ("gbm_reg", gbm_reg["raw"], None, None),
        ("gbm_det", gbm_det["raw"], None, None),
    ]
    for name, p_test, model, tag in det_models:
        thr_rate = rate_thr(p_test, rate)
        det = boot_det(y_test, p_test, thr_rate, storms)
        det["model"] = name
        det["rule"] = "alert_rate_equals_test_ri_rate"
        detection["rate_matched"].append(det)
        print(f"det rate {name}: CSI={det['csi']['point']:.3f}")

        if model is None:
            # GBM: use val raw for threshold
            if name == "gbm_reg":
                p_val = gbm_reg["val_raw"]
            else:
                p_val = vr0
            thr_v = _best_csi_thr(y_val, p_val)
        else:
            p_val = val_raw_neural(model, tag)
            if p_val is None:
                print("skip val csi for", name)
                continue
            thr_v = _best_csi_thr(y_val, p_val)
        det2 = boot_det(y_test, p_test, thr_v, storms)
        det2["model"] = name
        det2["rule"] = "max_csi_on_validation"
        detection["val_csi"].append(det2)
        print(f"det val {name}: thr={thr_v:.3f} CSI={det2['csi']['point']:.3f}")

    # sample vs clustered
    sample_vs = {
        "mlp_u_raw_sample": bootstrap_bss(y_test, series["mlp_u_raw"], n_boot=500, clusters=None),
        "mlp_u_raw_clustered": bootstrap_bss(y_test, series["mlp_u_raw"], n_boot=500, clusters=storms),
        "fus_u_iso_sample": bootstrap_bss(y_test, series["fus_u_iso"], n_boot=500, clusters=None),
        "fus_u_iso_clustered": bootstrap_bss(y_test, series["fus_u_iso"], n_boot=500, clusters=storms),
    }

    # by year
    by_year = {}
    for season, sub in test.groupby("season"):
        idx = sub.index.to_numpy()
        # test was reset so index 0..n-1 matches
        mask = test["season"].to_numpy() == season
        by_year[str(int(season))] = {
            "n": int(mask.sum()),
            "n_ri": int(y_test[mask].sum()),
            "ri_rate": float(np.mean(y_test[mask])),
            "mlp_u_iso_bss": float(brier_skill_score(y_test[mask], series["mlp_u_iso"][mask])),
            "fus_u_iso_bss": float(brier_skill_score(y_test[mask], series["fus_u_iso"][mask])),
            "gbm_reg_iso_bss": float(brier_skill_score(y_test[mask], gbm_reg["iso"][mask])),
            "img_u_iso_bss": float(brier_skill_score(y_test[mask], series["img_u_iso"][mask])),
        }

    # strip non-json probs from grid
    grid_public = []
    for r in grid_sorted:
        grid_public.append({k: v for k, v in r.items() if not k.startswith("_")})

    out = {
        "n_test": len(y_test),
        "n_storms": n_storms,
        "n_ri": n_ri,
        "ri_rate": rate,
        "gbm_grid": grid_public,
        "gbm_best": {k: best[k] for k in best if not k.startswith("_")},
        "calibrator_val_scores": cal_scores,
        "primary_calibrator": primary_cal,
        "abs": abs_rows,
        "paired": paired,
        "single_seed": single,
        "detection": detection,
        "sample_vs_clustered": sample_vs,
        "by_year": by_year,
        "notes": [
            "Regularized GBM HP selected by mean raw validation Brier across seeds 42/43/44.",
            "Primary calibrator selected by held-out half of validation Brier after fitting on the other half.",
            "IR-scalars-only MLP is the informative image-content control versus imagery CNN.",
            "Rate-matched detection is diagnostic; val-CSI threshold is the deployable operating point.",
        ],
    }
    # remove huge nested from save - already public grid
    out_path = paths["metrics"] / "cautionary_metrics.json"
    save_json(out, out_path)
    print("saved", out_path)


if __name__ == "__main__":
    main()
