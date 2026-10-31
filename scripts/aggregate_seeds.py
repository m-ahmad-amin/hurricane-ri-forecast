from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config, project_paths, save_json


def _pick(data: dict, key: str) -> dict:
    if key in data and isinstance(data[key], dict) and "brier" in data[key]:
        return data[key]
    if key == "raw" and "brier" in data:
        return data
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed eval metrics")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--models", nargs="+", default=["tabular", "imagery", "fusion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    tag = args.tag or Path(args.config).stem
    default_seed = int(cfg["train"].get("seed", 42))

    records = []
    for model in args.models:
        for seed in args.seeds:
            path = paths["metrics"] / f"eval_{model}_{tag}_seed{seed}.json"
            if not path.exists() and seed == default_seed:
                path = paths["metrics"] / f"eval_{model}_{tag}.json"
            if not path.exists():
                print(f"missing {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            selected = data.get("selected_calibration", "temperature")
            block_key = {
                "raw": "raw",
                "temperature": "temperature_scaled",
                "isotonic": "isotonic",
            }.get(selected, "temperature_scaled")
            block = _pick(data, block_key) or _pick(data, "temperature_scaled") or _pick(data, "raw")
            cat12 = None
            regimes = data.get("regime_breakdown") or {}
            intensity = regimes.get("regime_intensity") or {}
            if "Cat1-2_64-95kt" in intensity:
                cat12 = intensity["Cat1-2_64-95kt"]
            records.append(
                {
                    "model": model,
                    "seed": seed,
                    "calibration": selected,
                    "brier": block.get("brier"),
                    "bss": block.get("brier_skill_vs_climatology"),
                    "pod": block.get("pod"),
                    "success_ratio": block.get("success_ratio"),
                    "cat12_bss": (cat12 or {}).get("brier_skill_vs_climatology"),
                    "cat12_brier": (cat12 or {}).get("brier"),
                    "cat12_pod": (cat12 or {}).get("pod"),
                }
            )

    if not records:
        raise RuntimeError("no seed eval json files found")

    print("\nPer-seed primary metrics")
    print(
        f"{'model':<10} {'seed':>5} {'brier':>8} {'bss':>8} {'pod':>8} "
        f"{'cat12_bss':>10} {'cat12_pod':>10}"
    )
    for r in records:
        print(
            f"{r['model']:<10} {r['seed']:5d} "
            f"{r['brier']:8.4f} {r['bss']:8.4f} {r['pod']:8.4f} "
            f"{(r['cat12_bss'] if r['cat12_bss'] is not None else float('nan')):10.4f} "
            f"{(r['cat12_pod'] if r['cat12_pod'] is not None else float('nan')):10.4f}"
        )

    summary = []
    print("\nMean ± std across seeds")
    print(f"{'model':<10} {'brier':>18} {'bss':>18} {'cat12_bss':>18}")
    for model in args.models:
        subset = [r for r in records if r["model"] == model]
        if not subset:
            continue

        def _ms(key: str) -> tuple[float, float]:
            vals = np.asarray([r[key] for r in subset if r[key] is not None], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.std(vals, ddof=0))

        b_m, b_s = _ms("brier")
        s_m, s_s = _ms("bss")
        c_m, c_s = _ms("cat12_bss")
        summary.append(
            {
                "model": model,
                "n_seeds": len(subset),
                "brier_mean": b_m,
                "brier_std": b_s,
                "bss_mean": s_m,
                "bss_std": s_s,
                "cat12_bss_mean": c_m,
                "cat12_bss_std": c_s,
            }
        )
        print(
            f"{model:<10} "
            f"{b_m:7.4f}±{b_s:<7.4f} "
            f"{s_m:7.4f}±{s_s:<7.4f} "
            f"{c_m:7.4f}±{c_s:<7.4f}"
        )

    print("\nPer-seed fusion BSS minus tabular BSS")
    deltas = []
    for seed in args.seeds:
        tab = next((r for r in records if r["model"] == "tabular" and r["seed"] == seed), None)
        fus = next((r for r in records if r["model"] == "fusion" and r["seed"] == seed), None)
        if tab and fus and tab["bss"] is not None and fus["bss"] is not None:
            d = fus["bss"] - tab["bss"]
            deltas.append(d)
            print(f"  seed {seed}: {d:.4f}")
    if deltas:
        arr = np.asarray(deltas, dtype=float)
        print(f"  mean±std: {arr.mean():.4f}±{arr.std(ddof=0):.4f}")

    out = paths["metrics"] / f"seed_aggregate_{tag}.json"
    save_json({"records": records, "summary": summary, "fusion_minus_tabular_bss": deltas}, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
