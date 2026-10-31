from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.metrics import bootstrap_bss, summarize_predictions
from src.utils import load_config, project_paths, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Primary BSS by held-out test year")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--models", nargs="+", default=["tabular", "imagery", "fusion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--prob-col", default="prob_primary")
    parser.add_argument("--n-boot", type=int, default=300)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    tag = args.tag or Path(args.config).stem

    rows = []
    print("\nHeld-out skill by test year (storm-clustered BSS CI)")
    print(
        f"{'model':<10} {'seed':>5} {'year':>6} {'n':>6} {'ri':>7} "
        f"{'brier':>8} {'bss':>8} {'bss_lo':>8} {'bss_hi':>8} {'pod':>8}"
    )
    for seed in args.seeds:
        for model in args.models:
            path = paths["metrics"] / f"predictions_{model}_{tag}_seed{seed}.csv"
            if not path.exists() and seed == int(cfg["train"].get("seed", 42)):
                path = paths["metrics"] / f"predictions_{model}_{tag}.csv"
            if not path.exists():
                print(f"missing {path}")
                continue
            df = pd.read_csv(path)
            for year in sorted(df["season"].dropna().unique()):
                m = df["season"].to_numpy() == year
                y = df.loc[m, "label"].to_numpy(dtype=float)
                p = df.loc[m, args.prob_col].to_numpy(dtype=float)
                storms = df.loc[m, "storm_id"].to_numpy()
                s = summarize_predictions(y, p)
                boot = bootstrap_bss(y, p, n_boot=args.n_boot, clusters=storms)
                row = {
                    "model": model,
                    "seed": seed,
                    "year": int(year),
                    "n": s["n"],
                    "ri_rate": s["ri_rate"],
                    "brier": s["brier"],
                    "bss": s["brier_skill_vs_climatology"],
                    "pod": s["pod"],
                    **boot,
                }
                rows.append(row)
                print(
                    f"{model:<10} {seed:5d} {int(year):6d} {row['n']:6d} {row['ri_rate']:7.3f} "
                    f"{row['brier']:8.4f} {row['bss']:8.4f} "
                    f"{row['bss_lo']:8.4f} {row['bss_hi']:8.4f} {row['pod']:8.4f}"
                )

    out = paths["metrics"] / f"year_breakdown_{tag}.json"
    save_json({"rows": rows}, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
