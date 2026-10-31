from __future__ import annotations

"""Seed-averaged probabilities + one storm-clustered bootstrap per contrast.

Also builds validation-CSI-threshold detection metrics with clustered intervals.
"""

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
    bootstrap_bss,
    bootstrap_delta_bss,
    brier_skill_score,
)
from src.utils import get_device, load_config, project_paths, save_json


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


def _predict(model, loader, device):
    model.eval()
    ps = []
    with torch.no_grad():
        for b in loader:
            logit = model(b["tabular"].to(device), b["image"].to(device))
            ps.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(ps)


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


def _bootstrap_detection(y, p, thr, clusters, n_boot=500, seed=42):
    """Cluster bootstrap of POD/FAR/CSI at fixed threshold."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    pods, fars, csis = [], [], []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(clusters == c) for c in draw])
        csi, c = _csi(y[idx], p[idx], thr)
        pods.append(c["pod"])
        fars.append(c["far"])
        csis.append(csi)
    def _summ(vals, point):
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"point": point, "lo": float("nan"), "hi": float("nan")}
        return {
            "point": float(point),
            "lo": float(np.quantile(vals, 0.025)),
            "hi": float(np.quantile(vals, 0.975)),
        }
    csi0, c0 = _csi(y, p, thr)
    return {
        "threshold": thr,
        "pod": _summ(pods, c0["pod"]),
        "far": _summ(fars, c0["far"]),
        "csi": _summ(csis, csi0),
    }


def _load_pred(paths, model, name_tag, seed):
    p = paths["metrics"] / f"predictions_{model}_{name_tag}_seed{seed}.csv"
    if not p.exists() and name_tag.endswith("_noweight") is False:
        # weighted defaults
        p = paths["metrics"] / f"predictions_{model}_default_seed{seed}.csv"
        if not p.exists() and seed == 42:
            p = paths["metrics"] / f"predictions_{model}_default.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _align(test, pred):
    m = test[["storm_id", "time"]].merge(pred, on=["storm_id", "time"], how="inner")
    if len(m) != len(test):
        raise RuntimeError(f"merge size {len(m)} != {len(test)}")
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    tag = "default"
    table = pd.read_parquet(paths["processed"] / f"samples_{tag}.parquet")
    feat_path = paths["processed"] / f"features_{tag}.json"
    feature_cols = (
        json.load(open(feat_path, encoding="utf-8"))
        if feat_path.exists()
        else [c for c in TABULAR_FEATURES if c in table.columns]
    )

    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    test = table[table["split"] == "test"].reset_index(drop=True)
    y_train = train["ri_label"].to_numpy(int)
    y_val = val["ri_label"].to_numpy(float)
    y_test = test["ri_label"].to_numpy(float)
    storms = test["storm_id"].to_numpy()
    framed = assign_regimes(test.assign(label=y_test))
    cat = framed["regime_intensity"].to_numpy() == "Cat1-2_64-95kt"

    med = np.nanmedian(train[feature_cols].to_numpy(np.float64), 0)
    x_train = _fill(train[feature_cols].to_numpy(np.float64), med)
    x_val = _fill(val[feature_cols].to_numpy(np.float64), med)
    x_test = _fill(test[feature_cols].to_numpy(np.float64), med)

    # Collect per-seed calibrated probs, then average
    series = {
        "mlp_w_raw": [],
        "mlp_w_temp": [],
        "mlp_w_iso": [],
        "mlp_u_raw": [],
        "mlp_u_temp": [],
        "mlp_u_iso": [],
        "img_u_raw": [],
        "img_u_temp": [],
        "img_u_iso": [],
        "img_u_primary": [],
        "fus_u_raw": [],
        "fus_u_temp": [],
        "fus_u_iso": [],
        "fus_u_primary": [],
        "img_raw": [],
        "img_temp": [],
        "img_iso": [],
        "fus_iso": [],
        "gbm_raw": [],
        "gbm_temp": [],
        "gbm_platt": [],
        "gbm_iso": [],
    }
    # also need val probs for CSI threshold / calibrators on seed-avg: fit cal on
    # seed-averaged val scores. For neural we only have test CSVs; approximate
    # CSI threshold using seed-42 val from re-fit GBM and for neural use
    # isotonic test with threshold chosen on... we need val. For MLP unweighted
    # we can reload checkpoints; for speed: choose CSI thr on test is wrong.
    # Use per-seed val CSI thr averaged, OR fit calibrators on pooled approach:
    # For detection table: use seed-averaged TEST probs, choose threshold by
    # maximizing CSI on seed-averaged... we don't have val neural probs in CSV.
    #
    # Practical approach: for each model, take seed-averaged test probs for a
    # calibrator; choose threshold on a storm-held-out 20% of TEST for reporting
    # sensitivity — NO that's still test leakage.
    #
    # Better: for GBM fit on train, calibrate on val, seed-average GBM probs.
    # For neural: load noweight/weighted prediction CSVs (test only); for CSI
    # threshold, use the mean of per-seed validation CSI thresholds computed
    # by reloading checkpoints for tabular only; for imagery use temperature
    # primary and threshold grid on... 
    #
    # Seed-averaged probabilities, then one clustered bootstrap:
    # - Seed-average test probabilities per calibrator
    # - For CSI threshold: maximize CSI on VALIDATION using seed-averaged val
    #   probs for GBM and tabular (reload); for imagery, fit isotonic on each
    #   seed's... we don't have imagery val.
    #
    # Imagery/fusion: report detection using threshold maximizing CSI on the
    # *training-season validation is unavailable without images*. Use the
    # seed-mean of thresholds that maximize CSI on each seed's TEST fold? Bad.
    #
    # Report for imagery: threshold = climatology rate quantile (alert rate =
    # base rate) as a labeled operating point, plus note CSI@val-thr only for
    # models with val scores (MLP, GBM).

    mlp_u_val_raw = []  # collect for threshold: use seed-42 checkpoint style
    # We'll compute CSI thr from GBM val and MLP by re-predicting tabular.

    for seed in args.seeds:
        # weighted MLP / imagery / fusion from default CSVs
        for key, model, col, name_tag in (
            ("mlp_w_raw", "tabular", "prob_raw", "default"),
            ("mlp_w_temp", "tabular", "prob_temperature", "default"),
            ("mlp_w_iso", "tabular", "prob_isotonic", "default"),
            ("mlp_u_raw", "tabular", "prob_raw", "default_noweight"),
            ("mlp_u_temp", "tabular", "prob_temperature", "default_noweight"),
            ("mlp_u_iso", "tabular", "prob_isotonic", "default_noweight"),
            # Prefer unweighted imagery/fusion when available (publishable control)
            ("img_u_raw", "imagery", "prob_raw", "default_noweight"),
            ("img_u_temp", "imagery", "prob_temperature", "default_noweight"),
            ("img_u_iso", "imagery", "prob_isotonic", "default_noweight"),
            ("img_u_primary", "imagery", "prob_primary", "default_noweight"),
            ("fus_u_raw", "fusion", "prob_raw", "default_noweight"),
            ("fus_u_temp", "fusion", "prob_temperature", "default_noweight"),
            ("fus_u_iso", "fusion", "prob_isotonic", "default_noweight"),
            ("fus_u_primary", "fusion", "prob_primary", "default_noweight"),
            # Weighted originals (ablation)
            ("img_raw", "imagery", "prob_raw", "default"),
            ("img_temp", "imagery", "prob_temperature", "default"),
            ("img_iso", "imagery", "prob_isotonic", "default"),
            ("fus_iso", "fusion", "prob_isotonic", "default"),
        ):
            pred = _load_pred(paths, model, name_tag, seed)
            if pred is None:
                continue
            m = _align(test, pred)
            series[key].append(m[col].to_numpy(float))

        gbm = GradientBoostingClassifier(random_state=seed)
        gbm.fit(x_train, y_train)
        vr = gbm.predict_proba(x_val)[:, 1]
        tr = gbm.predict_proba(x_test)[:, 1]
        T = fit_temperature(y_val, vr)
        pl = fit_platt(y_val, vr)
        iso = fit_isotonic(y_val, vr)
        series["gbm_raw"].append(tr)
        series["gbm_temp"].append(temperature_scale(probs_to_logits(tr), T))
        series["gbm_platt"].append(apply_platt(pl, tr))
        series["gbm_iso"].append(apply_isotonic(iso, tr))
        if seed == args.seeds[0]:
            gbm_val = {
                "raw": vr,
                "temperature": temperature_scale(probs_to_logits(vr), T),
                "platt": apply_platt(pl, vr),
                "isotonic": apply_isotonic(iso, vr),
            }

    avg = {}
    for k, lst in series.items():
        if not lst:
            continue
        avg[k] = np.mean(np.stack(lst, axis=0), axis=0)

    def _bss(p, mask=None):
        if mask is None:
            mask = np.ones(len(y_test), bool)
        return float(brier_skill_score(y_test[mask], p[mask]))

    def _pair(name, pa, pb, mask, scope):
        y = y_test[mask]
        st = storms[mask]
        point = float(brier_skill_score(y, pa[mask]) - brier_skill_score(y, pb[mask]))
        boot = bootstrap_delta_bss(y, pa[mask], pb[mask], n_boot=args.n_boot, clusters=st)
        rec = {
            "contrast": name,
            "scope": scope,
            "point_delta_bss": point,
            "bss_a": float(brier_skill_score(y, pa[mask])),
            "bss_b": float(brier_skill_score(y, pb[mask])),
            "n_storms": int(len(np.unique(st))),
            **boot,
        }
        print(
            f"{name} {scope}: point={point:+.3f} "
            f"[{boot['delta_lo']:.3f},{boot['delta_hi']:.3f}] "
            f"(A={rec['bss_a']:+.3f} B={rec['bss_b']:+.3f})"
        )
        return rec

    paired = []
    # Prefer unweighted imagery/fusion for primary publishable contrasts
    img_iso = avg.get("img_u_iso", avg.get("img_iso"))
    fus_pri = avg.get("fus_u_primary", avg.get("fus_u_raw", avg.get("fus_iso")))
    fus_iso = avg.get("fus_u_iso", avg.get("fus_iso"))

    paired.append(_pair("img_u_iso_minus_gbm_iso", img_iso, avg["gbm_iso"], cat, "Cat1-2"))
    paired.append(_pair("img_u_iso_minus_mlp_u_iso", img_iso, avg["mlp_u_iso"], cat, "Cat1-2"))
    if "img_iso" in avg:
        paired.append(_pair("img_w_iso_minus_gbm_iso", avg["img_iso"], avg["gbm_iso"], cat, "Cat1-2"))
    paired.append(_pair("fus_u_primary_minus_mlp_u_raw", fus_pri, avg["mlp_u_raw"], np.ones(len(y_test), bool), "overall"))
    paired.append(_pair("fus_u_primary_minus_gbm_platt", fus_pri, avg["gbm_platt"], np.ones(len(y_test), bool), "overall"))
    paired.append(_pair("fus_u_iso_minus_gbm_iso", fus_iso, avg["gbm_iso"], cat, "Cat1-2"))
    paired.append(_pair("mlp_u_iso_minus_gbm_iso", avg["mlp_u_iso"], avg["gbm_iso"], np.ones(len(y_test), bool), "overall"))
    paired.append(_pair("mlp_u_raw_minus_gbm_raw", avg["mlp_u_raw"], avg["gbm_raw"], np.ones(len(y_test), bool), "overall"))
    paired.append(_pair("mlp_w_iso_minus_gbm_iso", avg["mlp_w_iso"], avg["gbm_iso"], np.ones(len(y_test), bool), "overall"))
    paired.append(_pair("mlp_u_iso_minus_gbm_iso", avg["mlp_u_iso"], avg["gbm_iso"], cat, "Cat1-2"))

    # Single-model BSS with clustered CI on seed-avg
    singles = []
    for name, p in avg.items():
        for scope, mask in (("overall", np.ones(len(y_test), bool)), ("Cat1-2", cat)):
            boot = bootstrap_bss(y_test[mask], p[mask], n_boot=args.n_boot, clusters=storms[mask])
            rec = {
                "model": name,
                "scope": scope,
                "bss_point": _bss(p, mask),
                "mean_pred": float(np.mean(p[mask])),
                **boot,
            }
            singles.append(rec)
            if scope == "overall":
                print(
                    f"{name:20s} overall BSS={rec['bss_point']:+.3f} "
                    f"[{boot['bss_lo']:.3f},{boot['bss_hi']:.3f}] pbar={rec['mean_pred']:.3f}"
                )

    # Detection table: CSI thr from GBM val; for MLP unweighted/weighted use
    # isotonic/raw and choose thr maximizing CSI on a proxy — reload not done.
    # Use GBM val for GBM; for neural seed-avg test probs, choose thr by
    # maximizing CSI on validation is impossible without val probs.
    # Compromise: report threshold = RI rate quantile (forecast rate = clim)
    # labeled as rate-matched, AND for GBM report true val-CSI-thr.
    detection = []
    rate = float(np.mean(y_test))
    for name, p in (
        ("mlp_u_raw", avg["mlp_u_raw"]),
        ("mlp_u_iso", avg["mlp_u_iso"]),
        ("mlp_w_raw", avg["mlp_w_raw"]),
        ("mlp_w_iso", avg["mlp_w_iso"]),
        ("img_iso", avg["img_iso"]),
        ("gbm_platt", avg["gbm_platt"]),
        ("gbm_iso", avg["gbm_iso"]),
        ("gbm_raw", avg["gbm_raw"]),
    ):
        # rate-matched threshold: kth score
        k = max(int(round(rate * len(p))), 1)
        thr_rate = float(np.sort(p)[-k])
        det_rate = _bootstrap_detection(y_test, p, thr_rate, storms, n_boot=args.n_boot)
        det_rate["rule"] = "alert_rate_equals_test_ri_rate"
        det_rate["model"] = name
        detection.append(det_rate)

        if name.startswith("gbm_"):
            cal = name[len("gbm_"):]
            # map short names
            cal_key = {"iso": "isotonic", "temp": "temperature"}.get(cal, cal)
            thr = _best_csi_thr(y_val, gbm_val[cal_key])
            det = _bootstrap_detection(y_test, p, thr, storms, n_boot=args.n_boot)
            det["rule"] = "max_csi_on_validation"
            det["model"] = name
            detection.append(det)
            print(
                f"det {name} val-CSI-thr={thr:.3f} "
                f"CSI={det['csi']['point']:.3f} [{det['csi']['lo']:.3f},{det['csi']['hi']:.3f}] "
                f"FAR={det['far']['point']:.3f} POD={det['pod']['point']:.3f}"
            )

    # Tabular MLP val-CSI thresholds via checkpoints (seed-averaged val probs)
    device = get_device()
    mean_s, std_s = fit_feature_scaler(table, feature_cols)
    val_loader = DataLoader(_TabDS(val, feature_cols, mean_s, std_s), batch_size=256, shuffle=False)
    for label, name_tag in (("mlp_u", "default_noweight"), ("mlp_w", "default")):
        val_raws = []
        for seed in args.seeds:
            ckpt = paths["checkpoints"] / f"tabular_{name_tag}_seed{seed}.pt"
            if not ckpt.exists() and name_tag == "default" and seed == 42:
                ckpt = paths["checkpoints"] / "tabular_default.pt"
            if not ckpt.exists():
                continue
            model = build_model("tabular", n_features=len(feature_cols))
            try:
                state = torch.load(ckpt, map_location=device, weights_only=False)
            except TypeError:
                state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
            model.to(device)
            val_raws.append(_predict(model, val_loader, device))
        if not val_raws:
            continue
        p_val = np.mean(np.stack(val_raws, 0), 0)
        iso = fit_isotonic(y_val, p_val)
        p_val_iso = apply_isotonic(iso, p_val)
        for cal, p_v, p_t_key in (
            ("raw", p_val, f"{label}_raw"),
            ("isotonic", p_val_iso, f"{label}_iso"),
        ):
            if p_t_key not in avg:
                continue
            thr = _best_csi_thr(y_val, p_v)
            det = _bootstrap_detection(y_test, avg[p_t_key], thr, storms, n_boot=args.n_boot)
            det["rule"] = "max_csi_on_validation"
            det["model"] = p_t_key
            detection.append(det)
            print(
                f"det {p_t_key} val-CSI-thr={thr:.3f} "
                f"CSI={det['csi']['point']:.3f} [{det['csi']['lo']:.3f},{det['csi']['hi']:.3f}] "
                f"FAR={det['far']['point']:.3f} POD={det['pod']['point']:.3f}"
            )

    # Mean pred audit
    mean_pred = {
        k: {
            "test": float(np.mean(v)),
            "obs_test": float(np.mean(y_test)),
        }
        for k, v in avg.items()
    }

    out = {
        "note": (
            "Probabilities are averages across seeds 42/43/44 before scoring. "
            "One storm-clustered bootstrap per contrast (seeds are not independent "
            "replicates of the test set). Many contrasts are reported without "
            "multiplicity correction."
        ),
        "seeds": args.seeds,
        "ri_rates": {
            "train": float(np.mean(y_train)),
            "val": float(np.mean(y_val)),
            "test": float(np.mean(y_test)),
        },
        "mean_pred": mean_pred,
        "singles": singles,
        "paired_delta_bss": paired,
        "detection": detection,
        "loss_statement": (
            "All neural pathways (tabular, imagery, fusion) in the original runs "
            "used identical BCEWithLogitsLoss with pos_weight=n_neg/n_pos. "
            "Unweighted tabular MLP is a separate control (suffix noweight). "
            "Different test mean predicted probabilities under the same loss "
            "reflect pathway behavior, not different objectives."
        ),
    }
    out_path = paths["metrics"] / "seed_pooled_default.json"
    save_json(out, out_path)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
