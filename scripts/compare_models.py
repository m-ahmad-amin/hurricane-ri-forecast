from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config, project_paths, save_json


def _pick_block(data: dict, key: str) -> dict:
    # support old and new eval json shapes
    if key in data and isinstance(data[key], dict) and "brier" in data[key]:
        return data[key]
    if key == "raw" and "brier" in data:
        return data
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained model eval metrics")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--models", nargs="+", default=["tabular", "imagery", "fusion"])
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    tag = args.tag or Path(args.config).stem

    rows = []
    basin_rows = []
    regime_rows = []
    for model in args.models:
        path = paths["metrics"] / f"eval_{model}_{tag}.json"
        if not path.exists():
            print(f"missing {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        selected = data.get("selected_calibration", "raw")
        for cal_name, block_key in [
            ("raw", "raw"),
            ("temperature", "temperature_scaled"),
            ("isotonic", "isotonic"),
        ]:
            block = _pick_block(data, block_key)
            if not block:
                continue
            boot = block.get("bootstrap_bss", {})
            rows.append(
                {
                    "model": model,
                    "calibration": cal_name,
                    "selected": cal_name == selected,
                    "n": block.get("n"),
                    "ri_rate": block.get("ri_rate"),
                    "brier": block.get("brier"),
                    "bss": block.get("brier_skill_vs_climatology"),
                    "bss_lo": boot.get("bss_lo"),
                    "bss_hi": boot.get("bss_hi"),
                    "pod": block.get("pod"),
                    "success_ratio": block.get("success_ratio"),
                    "far": block.get("far"),
                    "temperature": data.get("temperature"),
                }
            )

        for basin, block in (data.get("basin_breakdown") or {}).items():
            primary = block.get("primary") or block.get("best") or {}
            if not primary or primary.get("n", 0) == 0:
                continue
            boot = primary.get("bootstrap_bss", {})
            basin_rows.append(
                {
                    "model": model,
                    "basin": basin,
                    "calibration": selected,
                    "n": primary.get("n"),
                    "ri_rate": primary.get("ri_rate"),
                    "brier": primary.get("brier"),
                    "bss": primary.get("brier_skill_vs_climatology"),
                    "bss_lo": boot.get("bss_lo"),
                    "bss_hi": boot.get("bss_hi"),
                    "pod": primary.get("pod"),
                    "success_ratio": primary.get("success_ratio"),
                }
            )

        for axis, buckets in (data.get("regime_breakdown") or {}).items():
            for name, block in buckets.items():
                if not block or block.get("n", 0) == 0:
                    continue
                regime_rows.append(
                    {
                        "model": model,
                        "axis": axis,
                        "regime": name,
                        "calibration": selected,
                        "n": block.get("n"),
                        "ri_rate": block.get("ri_rate"),
                        "brier": block.get("brier"),
                        "bss": block.get("brier_skill_vs_climatology"),
                        "pod": block.get("pod"),
                        "success_ratio": block.get("success_ratio"),
                    }
                )

    if not rows:
        raise RuntimeError("no eval json files found; run evaluate.py first")

    print("\nCalibration comparison (held-out test)")
    print("(* = primary = raw/temperature by val Brier; isotonic is audit-only)")
    print(
        f"{'model':<10} {'calib':<12} {'brier':>8} {'bss':>8} {'bss_lo':>8} {'bss_hi':>8} "
        f"{'pod':>8} {'succ':>8} {'pri':>4}"
    )
    for row in rows:
        print(
            f"{row['model']:<10} "
            f"{row['calibration']:<12} "
            f"{row['brier']:8.4f} "
            f"{row['bss']:8.4f} "
            f"{(row['bss_lo'] if row['bss_lo'] is not None else float('nan')):8.4f} "
            f"{(row['bss_hi'] if row['bss_hi'] is not None else float('nan')):8.4f} "
            f"{row['pod']:8.4f} "
            f"{row['success_ratio']:8.4f} "
            f"{('*' if row['selected'] else ''):>4}"
        )

    primary_rows = [r for r in rows if r["selected"]]
    by_name = {r["model"]: r for r in primary_rows}
    if "tabular" in by_name and "fusion" in by_name:
        gain = by_name["fusion"]["bss"] - by_name["tabular"]["bss"]
        print(f"\nprimary fusion BSS minus tabular BSS = {gain:.4f}")
    best = min(primary_rows, key=lambda r: r["brier"]) if primary_rows else None
    if best:
        print(f"best primary Brier: {best['model']} / {best['calibration']} ({best['brier']:.4f})")
    skilled = [r for r in primary_rows if r["bss"] is not None and r["bss"] > 0]
    if skilled:
        print("primary models with positive BSS vs climatology:", ", ".join(r["model"] for r in skilled))
    else:
        print("no primary model beat climatology on BSS yet")

    iso_rows = [r for r in rows if r["calibration"] == "isotonic"]
    if iso_rows:
        print("\nIsotonic audit (probability quality; may kill POD@0.5):")
        for row in iso_rows:
            print(
                f"  {row['model']:<10} brier={row['brier']:.4f} "
                f"bss={row['bss']:.4f} pod={row['pod']:.4f}"
            )

    if basin_rows:
        print("\nBasin breakdown (primary calibration)")
        print(
            f"{'model':<10} {'basin':<6} {'n':>6} {'ri':>7} {'brier':>8} {'bss':>8} {'pod':>8} {'succ':>8}"
        )
        for row in basin_rows:
            print(
                f"{row['model']:<10} "
                f"{row['basin']:<6} "
                f"{row['n']:6d} "
                f"{row['ri_rate']:7.3f} "
                f"{row['brier']:8.4f} "
                f"{row['bss']:8.4f} "
                f"{row['pod']:8.4f} "
                f"{row['success_ratio']:8.4f}"
            )

    if regime_rows:
        print("\nRegime breakdown (primary calibration)")
        print(
            f"{'model':<10} {'axis':<18} {'regime':<18} {'n':>6} {'ri':>7} "
            f"{'brier':>8} {'bss':>8} {'pod':>8} {'succ':>8}"
        )
        for row in sorted(regime_rows, key=lambda r: (r["axis"], r["regime"], r["model"])):
            print(
                f"{row['model']:<10} "
                f"{row['axis']:<18} "
                f"{row['regime']:<18} "
                f"{row['n']:6d} "
                f"{row['ri_rate']:7.3f} "
                f"{row['brier']:8.4f} "
                f"{row['bss']:8.4f} "
                f"{row['pod']:8.4f} "
                f"{row['success_ratio']:8.4f}"
            )

    out = paths["metrics"] / f"ablation_{tag}.json"
    save_json({"rows": rows, "basin_rows": basin_rows, "regime_rows": regime_rows}, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
