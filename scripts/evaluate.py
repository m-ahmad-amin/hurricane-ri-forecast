from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_processed, make_loaders
from src.data.prepare import TABULAR_FEATURES
from src.eval.calibration import (
    apply_isotonic,
    fit_isotonic,
    fit_temperature,
    probs_to_logits,
    temperature_scale,
)
from src.eval.cases import pick_cases, save_case_figure, save_reliability_figure
from src.eval.regimes import regime_breakdown
from src.eval.ships_rii import compare_to_ships_rii
from src.models.imagery import build_model
from src.train.loop import predict_loader
from src.train.metrics import bootstrap_bss, summarize_predictions
from src.utils import ensure_dirs, get_device, load_config, project_paths, save_json


def _subset_summary(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> dict:
    if len(y_true) == 0:
        return {"name": name, "n": 0}
    out = summarize_predictions(y_true, y_prob)
    out["bootstrap_bss"] = bootstrap_bss(y_true, y_prob)
    out["name"] = name
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RI models with calibration and basin splits")
    parser.add_argument("--config", default="configs/al062018.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--model", default="fusion")
    parser.add_argument("--ships-csv", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="load {model}_{tag}_seed{N}.pt; falls back to untagged ckpt for default seed",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="checkpoint/metrics suffix matching train.py --suffix (e.g. noweight)",
    )
    parser.add_argument(
        "--tabular-only",
        action="store_true",
        help="skip images_*.npy; zero tensors (tabular only)",
    )
    parser.add_argument(
        "--images-path",
        default=os.environ.get("IMAGE_NPY"),
        help="optional path to images_*.npy (local Colab copy recommended)",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    ensure_dirs(paths)

    tag = args.tag or Path(args.config).stem
    suffix = args.suffix.strip("_")
    name_tag = f"{tag}_{suffix}" if suffix else tag
    seed = args.seed
    if args.tabular_only:
        table = pd.read_parquet(paths["processed"] / f"samples_{tag}.parquet")
        feat_path = paths["processed"] / f"features_{tag}.json"
        if feat_path.exists():
            with open(feat_path, "r", encoding="utf-8") as f:
                feature_cols = json.load(f)
        else:
            feature_cols = [c for c in TABULAR_FEATURES if c in table.columns]
        images = None
    else:
        table, images, feature_cols, _ = load_processed(paths["processed"], tag)
        if args.images_path:
            images = np.load(args.images_path, mmap_mode="r")
    loaders, _, _ = make_loaders(
        table,
        images,
        feature_cols=feature_cols,
        batch_size=cfg["train"]["batch_size"],
        num_workers=0,
    )
    if "test" not in loaders:
        raise RuntimeError("no test split available")
    if "val" not in loaders:
        raise RuntimeError("validation split required for calibration")

    device = get_device()
    model = build_model(args.model, n_features=len(feature_cols))
    if seed is not None or suffix:
        ckpt_path = paths["checkpoints"] / f"{args.model}_{name_tag}_seed{seed if seed is not None else cfg['train'].get('seed', 42)}.pt"
        if not ckpt_path.exists() and seed is not None:
            # fall back to untagged name_tag without seed only if needed
            alt = paths["checkpoints"] / f"{args.model}_{name_tag}.pt"
            if alt.exists():
                ckpt_path = alt
    else:
        ckpt_path = paths["checkpoints"] / f"{args.model}_{tag}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    val_probs, val_labels, _ = predict_loader(model, loaders["val"], device)
    test_probs, test_labels, _ = predict_loader(model, loaders["test"], device)

    temperature = fit_temperature(val_labels, val_probs)
    iso_model = fit_isotonic(val_labels, val_probs)

    val_logits = probs_to_logits(val_probs)
    val_temp = temperature_scale(val_logits, temperature)
    val_iso = apply_isotonic(iso_model, val_probs)

    test_logits = probs_to_logits(test_probs)
    test_temp = temperature_scale(test_logits, temperature)
    test_iso = apply_isotonic(iso_model, test_probs)

    test_table = table[table["split"] == "test"].reset_index(drop=True)
    if images is None:
        test_images = np.zeros((len(test_table), 1, 128, 128), dtype=np.float32)
    else:
        test_images = images[table["split"].to_numpy() == "test"]

    raw = summarize_predictions(test_labels, test_probs)
    raw["bootstrap_bss"] = bootstrap_bss(test_labels, test_probs)
    temp = summarize_predictions(test_labels, test_temp)
    temp["bootstrap_bss"] = bootstrap_bss(test_labels, test_temp)
    iso = summarize_predictions(test_labels, test_iso)
    iso["bootstrap_bss"] = bootstrap_bss(test_labels, test_iso)

    # Operational primary: temperature (preserves ranking / POD).
    # Isotonic is an audit only; it can win Brier by collapsing below 0.5.
    # Choose raw vs temperature on validation Brier (never on test).
    val_raw = summarize_predictions(val_labels, val_probs)
    val_temp_m = summarize_predictions(val_labels, val_temp)
    val_iso_m = summarize_predictions(val_labels, val_iso)
    primary_name = "temperature" if val_temp_m["brier"] <= val_raw["brier"] else "raw"
    primary_probs = test_temp if primary_name == "temperature" else test_probs
    primary_summary = temp if primary_name == "temperature" else raw

    candidates = {
        "raw": (test_probs, raw),
        "temperature": (test_temp, temp),
        "isotonic": (test_iso, iso),
    }

    basin_breakdown = {}
    for basin in sorted(test_table["basin"].dropna().unique().tolist()):
        mask = test_table["basin"].to_numpy() == basin
        basin_breakdown[basin] = {
            "raw": _subset_summary(test_labels[mask], test_probs[mask], f"{basin}_raw"),
            "temperature": _subset_summary(test_labels[mask], test_temp[mask], f"{basin}_temperature"),
            "isotonic": _subset_summary(test_labels[mask], test_iso[mask], f"{basin}_isotonic"),
            "primary": _subset_summary(
                test_labels[mask], primary_probs[mask], f"{basin}_{primary_name}"
            ),
        }

    regimes = regime_breakdown(test_table, test_labels, primary_probs, _subset_summary)

    cases = pick_cases(test_table, primary_probs)
    save_case_figure(
        test_images,
        cases,
        paths["figures"] / f"cases_{args.model}_{name_tag}_{primary_name}.png",
    )
    save_reliability_figure(
        primary_summary["reliability"],
        paths["figures"] / f"reliability_{args.model}_{name_tag}_{primary_name}.png",
        title=f"{args.model} {primary_name} reliability ({name_tag})",
    )
    save_reliability_figure(
        raw["reliability"],
        paths["figures"] / f"reliability_{args.model}_{name_tag}_raw.png",
        title=f"{args.model} raw reliability ({name_tag})",
    )
    save_reliability_figure(
        iso["reliability"],
        paths["figures"] / f"reliability_{args.model}_{name_tag}_isotonic.png",
        title=f"{args.model} isotonic reliability ({name_tag})",
    )

    seed_suffix = f"_seed{seed}" if seed is not None else ""

    summary = {
        "model": args.model,
        "tag": tag,
        "name_tag": name_tag,
        "seed": seed,
        "temperature": temperature,
        "selected_calibration": primary_name,
        "calibration_policy": (
            "primary=raw_or_temperature by val Brier; "
            "isotonic reported as calibration audit only"
        ),
        "val_calibration": {
            "raw_brier": val_raw["brier"],
            "temperature_brier": val_temp_m["brier"],
            "isotonic_brier": val_iso_m["brier"],
            "isotonic_pod": val_iso_m["pod"],
        },
        "raw": raw,
        "temperature_scaled": temp,
        "isotonic": iso,
        "basin_breakdown": basin_breakdown,
        "regime_breakdown": regimes,
        "cases": cases,
    }

    if args.ships_csv:
        ships = pd.read_csv(args.ships_csv)
        merged = test_table.merge(ships, on=["storm_id", "time"], how="left")
        summary["ships_rii_comparison"] = compare_to_ships_rii(
            test_labels,
            primary_probs,
            merged["ships_rii_prob"].to_numpy(dtype=float),
        )

    out_pred = test_table.copy()
    out_pred["prob_raw"] = test_probs
    out_pred["prob_temperature"] = test_temp
    out_pred["prob_isotonic"] = test_iso
    out_pred["prob_primary"] = primary_probs
    out_pred["label"] = test_labels
    out_pred.to_csv(
        paths["metrics"] / f"predictions_{args.model}_{name_tag}{seed_suffix}.csv",
        index=False,
    )

    save_json(summary, paths["metrics"] / f"eval_{args.model}_{name_tag}{seed_suffix}.json")

    print(f"\n=== {args.model} {name_tag}{seed_suffix} ===")
    print(f"temperature T={temperature:.3f}")
    print(f"primary calibration: {primary_name} (val Brier; isotonic = audit only)")
    for name, (_, metrics) in candidates.items():
        boot = metrics["bootstrap_bss"]
        mark = "  <- primary" if name == primary_name else ""
        print(
            f"{name:12s} brier={metrics['brier']:.4f} "
            f"bss={metrics['brier_skill_vs_climatology']:.4f} "
            f"[{boot['bss_lo']:.4f}, {boot['bss_hi']:.4f}] "
            f"pod={metrics['pod']:.3f} succ={metrics['success_ratio']:.3f}{mark}"
        )
    if iso["pod"] < 0.05:
        print(
            "note: isotonic POD collapsed; better Brier here means safer probs, "
            "not better RI detection at 0.5"
        )
    print("\nBasin breakdown (primary calibration):")
    for basin, block in basin_breakdown.items():
        m = block["primary"]
        if m.get("n", 0) == 0:
            continue
        print(
            f"{basin}: n={m['n']} ri_rate={m['ri_rate']:.3f} "
            f"brier={m['brier']:.4f} bss={m['brier_skill_vs_climatology']:.4f} "
            f"pod={m['pod']:.3f} succ={m['success_ratio']:.3f}"
        )
    print("\nRegime breakdown (primary calibration):")
    for axis, buckets in regimes.items():
        print(f"  [{axis}]")
        for name, m in buckets.items():
            if m.get("n", 0) == 0:
                continue
            print(
                f"    {name}: n={m['n']} ri={m['ri_rate']:.3f} "
                f"brier={m['brier']:.4f} bss={m['brier_skill_vs_climatology']:.4f} "
                f"pod={m['pod']:.3f} succ={m['success_ratio']:.3f}"
            )


if __name__ == "__main__":
    main()
