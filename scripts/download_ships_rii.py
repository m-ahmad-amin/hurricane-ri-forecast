from __future__ import annotations

"""
Build a matched SHIPS-RII probability CSV for side-by-side comparison.

SHIPS-RII probabilities are not in TC PRIMED. Common sources:
  - NHC / CIRA SHIPS text diagnostics archives
  - Research dumps that already expose RI probabilities

This script:
  1. Writes a template CSV with the required schema, or
  2. Normalizes a user-supplied SHIPS file and reports match rate vs predictions.

Required output columns:
  storm_id,time,ships_rii_prob

time must match TC PRIMED overpass epoch seconds used in predictions_*.csv
(or be convertible via --time-unit s|ms|ns).

Example:
  python scripts/download_ships_rii.py --write-template
  python scripts/download_ships_rii.py --input path/to/ships.csv \\
      --from-predictions outputs/metrics/predictions_fusion_default.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SHIPS-RII probability CSV")
    parser.add_argument(
        "--out",
        default="data/raw/ships_rii/probabilities.csv",
        help="Output CSV path",
    )
    parser.add_argument("--write-template", action="store_true")
    parser.add_argument("--input", default=None, help="User-supplied SHIPS CSV to normalize")
    parser.add_argument(
        "--from-predictions",
        default=None,
        help="Prediction CSV used to report match rate on storm_id+time",
    )
    parser.add_argument("--storm-col", default="storm_id")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--prob-col", default="ships_rii_prob")
    parser.add_argument("--time-unit", choices=["s", "ms", "ns"], default="s")
    args = parser.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.write_template:
        template = pd.DataFrame(
            {
                "storm_id": ["AL052023"],
                "time": [0],
                "ships_rii_prob": [0.12],
            }
        )
        template.to_csv(out, index=False)
        print(f"wrote template {out}")
        print("Wrote template CSV; fill ships_rii_prob with values matched to overpass times.")
        return

    if not args.input:
        raise SystemExit("pass --write-template or --input <ships.csv>")

    raw = pd.read_csv(args.input)
    rename = {}
    if args.storm_col != "storm_id" and args.storm_col in raw.columns:
        rename[args.storm_col] = "storm_id"
    if args.time_col != "time" and args.time_col in raw.columns:
        rename[args.time_col] = "time"
    if args.prob_col != "ships_rii_prob" and args.prob_col in raw.columns:
        rename[args.prob_col] = "ships_rii_prob"
    ships = raw.rename(columns=rename)
    need = {"storm_id", "time", "ships_rii_prob"}
    missing = need - set(ships.columns)
    if missing:
        raise SystemExit(f"input missing columns after rename: {sorted(missing)}")

    ships = ships[["storm_id", "time", "ships_rii_prob"]].copy()
    ships["storm_id"] = ships["storm_id"].astype(str).str.upper()
    t = ships["time"].to_numpy(dtype=np.float64)
    if args.time_unit == "ms":
        t = t / 1000.0
    elif args.time_unit == "ns":
        t = t / 1e9
    ships["time"] = t
    ships["ships_rii_prob"] = ships["ships_rii_prob"].astype(float)
    ships = ships.dropna(subset=["storm_id", "time", "ships_rii_prob"])
    ships = ships.drop_duplicates(subset=["storm_id", "time"], keep="last")
    ships.to_csv(out, index=False)
    print(f"wrote {out} n={len(ships)}")

    if args.from_predictions:
        pred = pd.read_csv(ROOT / args.from_predictions)
        pred["storm_id"] = pred["storm_id"].astype(str).str.upper()
        merged = pred.merge(ships, on=["storm_id", "time"], how="left")
        matched = int(np.isfinite(merged["ships_rii_prob"]).sum())
        print(
            f"match rate vs {args.from_predictions}: "
            f"{matched}/{len(pred)} ({100.0 * matched / max(len(pred), 1):.1f}%)"
        )


if __name__ == "__main__":
    main()
