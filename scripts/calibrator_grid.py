from __future__ import annotations

"""Matched calibrator grid + multi-seed imagery-minus-GBM + sample vs clustered CI.

Neural raw / temperature / isotonic come from saved prediction CSVs (and eval
JSON for consistency). Platt for neural models requires validation scores:
tabular uses a checkpoint forward pass with dummy images; imagery/fusion use
real images if --images-path is provided (or IMAGE_NPY env), else Platt is
skipped for those two models.
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
    bootstrap_bss,
    bootstrap_delta_bss,
    brier_skill_score,
    summarize_predictions,
)
from src.utils import ensure_dirs, get_device, load_config, project_paths, save_json


class _TableDataset(Dataset):
    def __init__(self, table, images, feature_cols, mean, std):
        self.table = table.reset_index(drop=True)
        self.images = images
        self.feature_cols = feature_cols
        feats = self.table[feature_cols].to_numpy(dtype=np.float32)
        feats = np.where(np.isfinite(feats), feats, mean)
        self.features = ((feats - mean) / std).astype(np.float32)
        self.labels = self.table["ri_label"].to_numpy(dtype=np.float32)
        self.use_real_images = images is not None

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        if self.use_real_images:
            img = torch.from_numpy(np.asarray(self.images[idx], dtype=np.float32))
        else:
            img = torch.zeros(1, 128, 128, dtype=torch.float32)
        return {
            "tabular": torch.from_numpy(self.features[idx]),
            "image": img,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def _predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            tab = batch["tabular"].to(device)
            img = batch["image"].to(device)
            logit = model(tab, img)
            probs.append(torch.sigmoid(logit).cpu().numpy())
            labels.append(batch["label"].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def _ckpt_path(paths, model: str, tag: str, seed: int, default_seed: int) -> Path:
    tagged = paths["checkpoints"] / f"{model}_{tag}_seed{seed}.pt"
    if tagged.exists():
        return tagged
    if seed == default_seed:
        untagged = paths["checkpoints"] / f"{model}_{tag}.pt"
        if untagged.exists():
            return untagged
    raise FileNotFoundError(f"checkpoint not found for {model} seed={seed}")


def _score_pack(y: np.ndarray, p: np.ndarray) -> dict:
    s = summarize_predictions(y, p)
    return {
        "brier": s["brier"],
        "bss": s["brier_skill_vs_climatology"],
        "pod": s["pod"],
        "roc_auc": s["roc_auc"],
        "pr_auc": s["pr_auc"],
        "tp": s["tp"],
        "fp": s["fp"],
        "fn": s["fn"],
        "tn": s["tn"],
    }


def _calibrate_all(val_y, val_p, test_p) -> dict[str, np.ndarray]:
    temp = fit_temperature(val_y, val_p)
    platt = fit_platt(val_y, val_p)
    iso = fit_isotonic(val_y, val_p)
    logits = probs_to_logits(test_p)
    return {
        "raw": np.asarray(test_p, dtype=float),
        "temperature": temperature_scale(logits, temp),
        "platt": apply_platt(platt, test_p),
        "isotonic": apply_isotonic(iso, test_p),
    }


def _fill_nan(x, med):
    out = x.copy()
    inds = np.where(np.isnan(out))
    out[inds] = np.take(med, inds[1])
    return out


def _load_pred_csv(paths, model, tag, seed, default_seed):
    candidates = [
        paths["metrics"] / f"predictions_{model}_{tag}_seed{seed}.csv",
    ]
    if seed == default_seed:
        candidates.append(paths["metrics"] / f"predictions_{model}_{tag}.csv")
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched calibrator grid and paired GBM contrasts")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument(
        "--images-path",
        default=os.environ.get("IMAGE_NPY"),
        help="Optional path to hydrated images_*.npy for imagery/fusion Platt",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    ensure_dirs(paths)
    tag = args.tag or Path(args.config).stem
    default_seed = int(cfg["train"].get("seed", 42))
    device = get_device()
    batch_size = int(cfg["train"]["batch_size"])

    print("Loading samples parquet (no Drive image memmap required for CSV path)...")
    table = pd.read_parquet(paths["processed"] / f"samples_{tag}.parquet")
    feat_path = paths["processed"] / f"features_{tag}.json"
    if feat_path.exists():
        with open(feat_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in TABULAR_FEATURES if c in table.columns]

    images = None
    if args.images_path:
        img_path = Path(args.images_path)
        print(f"Loading images from {img_path} ...")
        images = np.load(img_path, mmap_mode="r")
        print(f"  shape={images.shape}")

    mean, std = fit_feature_scaler(table, feature_cols)
    test_df = table[table["split"] == "test"].copy().reset_index(drop=True)
    val_df = table[table["split"] == "val"].copy().reset_index(drop=True)
    train_df = table[table["split"] == "train"].copy()
    # row indices into full table / images array
    full_index = {i: n for n, i in enumerate(table.index)}
    # samples parquet usually has RangeIndex matching image rows
    test_rows = test_df.index.to_numpy() if hasattr(test_df, "index") else np.arange(len(test_df))
    # Prefer original positional index stored before reset
    test_pos = table.index[table["split"] == "test"].to_numpy()
    val_pos = table.index[table["split"] == "val"].to_numpy()
    # After reading parquet, index is 0..n-1 matching image rows if prepared that way
    test_pos = np.flatnonzero(table["split"].to_numpy() == "test")
    val_pos = np.flatnonzero(table["split"].to_numpy() == "val")

    y_test = test_df["ri_label"].to_numpy(dtype=float)
    storms_test = test_df["storm_id"].to_numpy()
    framed_test = assign_regimes(test_df.assign(label=y_test))
    cat_mask = framed_test["regime_intensity"].to_numpy() == "Cat1-2_64-95kt"

    x_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df["ri_label"].to_numpy(dtype=int)
    x_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    y_val = val_df["ri_label"].to_numpy(dtype=int)
    x_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    med = np.nanmedian(x_train, axis=0)
    x_train, x_val, x_test = _fill_nan(x_train, med), _fill_nan(x_val, med), _fill_nan(x_test, med)

    grid_rows = []
    paired_rows = []
    sample_vs_cluster = []
    gbm_cat12_probs: dict[int, np.ndarray] = {}
    imagery_cat12: dict[int, dict] = {}

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")

        # GBM full calibrator grid
        gbm = GradientBoostingClassifier(random_state=seed)
        gbm.fit(x_train, y_train)
        val_raw = gbm.predict_proba(x_val)[:, 1]
        test_raw = gbm.predict_proba(x_test)[:, 1]
        cal = _calibrate_all(y_val.astype(float), val_raw, test_raw)
        gbm_cat12_probs[seed] = cal["platt"][cat_mask]
        for name in ("raw", "temperature", "platt", "isotonic"):
            p = cal[name]
            pack = _score_pack(y_test, p)
            cat_pack = _score_pack(y_test[cat_mask], p[cat_mask])
            grid_rows.append(
                {"model": "gbm", "seed": seed, "calibrator": name, "overall": pack, "cat12": cat_pack}
            )
            print(
                f"  gbm/{name:<11} BSS={pack['bss']:+.3f} POD={pack['pod']:.3f} "
                f"AUC={pack['roc_auc']:.3f} PR={pack['pr_auc']:.3f} "
                f"| cat12 BSS={cat_pack['bss']:+.3f}"
            )

        for model_name in ("tabular", "imagery", "fusion"):
            pred = _load_pred_csv(paths, model_name, tag, seed, default_seed)
            if pred is None:
                print(f"  missing predictions for {model_name} seed={seed}")
                continue
            # align to test_df order via storm_id+time
            merged = test_df[["storm_id", "time", "ri_label"]].merge(
                pred,
                on=["storm_id", "time"],
                how="inner",
                suffixes=("", "_pred"),
            )
            if len(merged) != len(test_df):
                print(f"  WARN {model_name} seed={seed}: merge {len(merged)}/{len(test_df)}")
            y = merged["ri_label"].to_numpy(dtype=float)
            storms = merged["storm_id"].to_numpy()
            framed = assign_regimes(merged.assign(label=y, intensity_kt=merged.get("intensity_kt", test_df["intensity_kt"].to_numpy()[: len(merged)])))
            # intensity may already be in pred csv
            if "intensity_kt" in merged.columns:
                framed = assign_regimes(merged.assign(label=y))
            else:
                tmp = merged.copy()
                tmp["intensity_kt"] = test_df["intensity_kt"].to_numpy()
                framed = assign_regimes(tmp.assign(label=y))
            cmask = framed["regime_intensity"].to_numpy() == "Cat1-2_64-95kt"

            for name, col in (
                ("raw", "prob_raw"),
                ("temperature", "prob_temperature"),
                ("isotonic", "prob_isotonic"),
            ):
                p = merged[col].to_numpy(dtype=float)
                pack = _score_pack(y, p)
                cat_pack = _score_pack(y[cmask], p[cmask])
                grid_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "calibrator": name,
                        "overall": pack,
                        "cat12": cat_pack,
                    }
                )
                print(
                    f"  {model_name}/{name:<11} BSS={pack['bss']:+.3f} POD={pack['pod']:.3f} "
                    f"AUC={pack['roc_auc']:.3f} PR={pack['pr_auc']:.3f} "
                    f"| cat12 BSS={cat_pack['bss']:+.3f}"
                )

            # Platt: need validation scores
            need_images = model_name != "tabular"
            if need_images and images is None:
                print(f"  {model_name}/platt       SKIPPED (no hydrated images)")
            else:
                ckpt = _ckpt_path(paths, model_name, tag, seed, default_seed)
                model = build_model(model_name, n_features=len(feature_cols))
                try:
                    state = torch.load(ckpt, map_location=device, weights_only=False)
                except TypeError:
                    state = torch.load(ckpt, map_location=device)
                model.load_state_dict(state["model"])
                model.to(device)

                if need_images:
                    val_imgs = images[val_pos]
                    test_imgs = images[test_pos]
                else:
                    val_imgs = None
                    test_imgs = None

                val_ds = _TableDataset(val_df, val_imgs, feature_cols, mean, std)
                test_ds = _TableDataset(test_df, test_imgs, feature_cols, mean, std)
                # When using real images sliced arrays, Dataset must index 0..len-1 into sliced array
                if need_images:
                    # rebuild with positional image arrays matching val_df/test_df order
                    class _SliceDS(Dataset):
                        def __init__(self, tab, imgs, feats, mu, sg):
                            self.tab = tab.reset_index(drop=True)
                            self.imgs = imgs
                            f = self.tab[feats].to_numpy(dtype=np.float32)
                            f = np.where(np.isfinite(f), f, mu)
                            self.x = ((f - mu) / sg).astype(np.float32)
                            self.y = self.tab["ri_label"].to_numpy(dtype=np.float32)

                        def __len__(self):
                            return len(self.tab)

                        def __getitem__(self, i):
                            return {
                                "tabular": torch.from_numpy(self.x[i]),
                                "image": torch.from_numpy(np.asarray(self.imgs[i], dtype=np.float32)),
                                "label": torch.tensor(self.y[i], dtype=torch.float32),
                                "index": torch.tensor(i, dtype=torch.long),
                            }

                    val_ds = _SliceDS(val_df, val_imgs, feature_cols, mean, std)
                    test_ds = _SliceDS(test_df, test_imgs, feature_cols, mean, std)

                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
                val_p, val_y_hat = _predict(model, val_loader, device)
                test_p, _ = _predict(model, test_loader, device)
                platt_p = apply_platt(fit_platt(val_y_hat, val_p), test_p)
                # align platt to merged order (same as test_df order if merge complete)
                if len(merged) == len(test_df):
                    p = platt_p
                    y_p = y_test
                    cmask_p = cat_mask
                else:
                    p = platt_p
                    y_p = y
                    cmask_p = cmask
                pack = _score_pack(y_p, p)
                cat_pack = _score_pack(y_p[cmask_p], p[cmask_p])
                grid_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "calibrator": "platt",
                        "overall": pack,
                        "cat12": cat_pack,
                    }
                )
                print(
                    f"  {model_name}/{'platt':<11} BSS={pack['bss']:+.3f} POD={pack['pod']:.3f} "
                    f"AUC={pack['roc_auc']:.3f} PR={pack['pr_auc']:.3f} "
                    f"| cat12 BSS={cat_pack['bss']:+.3f}"
                )

            # primary probs for paired / sample-vs-cluster
            if model_name == "imagery":
                # primary = better of raw/temperature by... we use selected from CSV
                primary_col = "prob_primary" if "prob_primary" in merged.columns else "prob_temperature"
                p_prim = merged[primary_col].to_numpy(dtype=float)
                imagery_cat12[seed] = {
                    "y": y[cmask],
                    "p": p_prim[cmask],
                    "storms": storms[cmask],
                    "bss": float(brier_skill_score(y[cmask], p_prim[cmask])),
                }
                sample_boot = bootstrap_bss(y[cmask], p_prim[cmask], n_boot=args.n_boot, clusters=None)
                cluster_boot = bootstrap_bss(
                    y[cmask], p_prim[cmask], n_boot=args.n_boot, clusters=storms[cmask]
                )
                sample_vs_cluster.append(
                    {
                        "seed": seed,
                        "model": "imagery",
                        "regime": "Cat1-2",
                        "bss_point": imagery_cat12[seed]["bss"],
                        "sample_bootstrap": sample_boot,
                        "storm_clustered": cluster_boot,
                    }
                )
                print(
                    f"  imagery Cat1-2 sample CI "
                    f"[{sample_boot['bss_lo']:.3f}, {sample_boot['bss_hi']:.3f}] vs clustered "
                    f"[{cluster_boot['bss_lo']:.3f}, {cluster_boot['bss_hi']:.3f}]"
                )

        if seed in imagery_cat12 and seed in gbm_cat12_probs:
            y = imagery_cat12[seed]["y"]
            p_img = imagery_cat12[seed]["p"]
            p_gbm = gbm_cat12_probs[seed]
            # lengths must match; both are Cat1-2 of same test set
            n = min(len(p_img), len(p_gbm))
            y, p_img, p_gbm = y[:n], p_img[:n], p_gbm[:n]
            st = imagery_cat12[seed]["storms"][:n]
            point = float(brier_skill_score(y, p_img) - brier_skill_score(y, p_gbm))
            delta = bootstrap_delta_bss(y, p_img, p_gbm, n_boot=args.n_boot, clusters=st)
            rec = {
                "seed": seed,
                "contrast": "imagery_minus_gbm",
                "regime": "Cat1-2",
                "point_delta_bss": point,
                "imagery_bss": float(brier_skill_score(y, p_img)),
                "gbm_bss": float(brier_skill_score(y, p_gbm)),
                "n": int(len(y)),
                "n_storms": int(len(np.unique(st))),
                "n_ri": int(y.sum()),
                **delta,
            }
            paired_rows.append(rec)
            print(
                f"  paired imagery-GBM Cat1-2: point={point:+.3f} "
                f"boot mean={delta['delta_mean']:+.3f} "
                f"[{delta['delta_lo']:.3f}, {delta['delta_hi']:.3f}]"
            )

    summary = {}
    for model in ["tabular", "imagery", "fusion", "gbm"]:
        summary[model] = {}
        for cal_name in ("raw", "temperature", "platt", "isotonic"):
            rows = [r for r in grid_rows if r["model"] == model and r["calibrator"] == cal_name]
            if not rows:
                continue
            bss = np.array([r["overall"]["bss"] for r in rows], dtype=float)
            pod = np.array([r["overall"]["pod"] for r in rows], dtype=float)
            auc = np.array([r["overall"]["roc_auc"] for r in rows], dtype=float)
            pra = np.array([r["overall"]["pr_auc"] for r in rows], dtype=float)
            summary[model][cal_name] = {
                "n_seeds": len(rows),
                "bss_mean": float(bss.mean()),
                "bss_std": float(bss.std(ddof=1)) if len(bss) > 1 else 0.0,
                "bss_per_seed": bss.tolist(),
                "pod_mean": float(pod.mean()),
                "pod_per_seed": pod.tolist(),
                "roc_auc_mean": float(auc.mean()),
                "roc_auc_std": float(auc.std(ddof=1)) if len(auc) > 1 else 0.0,
                "roc_auc_per_seed": auc.tolist(),
                "pr_auc_mean": float(pra.mean()),
                "pr_auc_per_seed": pra.tolist(),
            }

    out = {
        "tag": tag,
        "n_boot": args.n_boot,
        "images_used_for_platt": bool(images is not None),
        "calibrators": ["raw", "temperature", "platt", "isotonic"],
        "note": (
            "Calibrators fit on validation only when scores are available. "
            "Neural raw/temperature/isotonic from saved prediction CSVs. "
            "GBM is untuned sklearn GradientBoostingClassifier defaults; "
            "paper primary GBM uses Platt."
        ),
        "rows": grid_rows,
        "seed_mean_summary": summary,
        "cat12_imagery_minus_gbm": paired_rows,
        "imagery_cat12_sample_vs_clustered": sample_vs_cluster,
    }
    out_path = paths["metrics"] / f"calibrator_grid_{tag}.json"
    save_json(out, out_path)
    print(f"\nsaved {out_path}")

    print("\n--- seed-mean overall BSS by calibrator ---")
    for model, cals in summary.items():
        for cal_name, s in cals.items():
            print(
                f"  {model:<8} {cal_name:<11} "
                f"BSS={s['bss_mean']:+.3f}±{s['bss_std']:.3f} "
                f"AUC={s['roc_auc_mean']:.3f} POD={s['pod_mean']:.3f}"
            )


if __name__ == "__main__":
    main()
