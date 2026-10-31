from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.regimes import assign_regimes
from src.train.metrics import bootstrap_bss, bootstrap_delta_bss, summarize_predictions
from src.utils import load_config, project_paths, save_json


def _load_pred(paths, model: str, tag: str, seed: int | None) -> pd.DataFrame | None:
    if seed is None:
        candidates = [
            paths["metrics"] / f"predictions_{model}_{tag}.csv",
        ]
    else:
        candidates = [
            paths["metrics"] / f"predictions_{model}_{tag}_seed{seed}.csv",
            paths["metrics"] / f"predictions_{model}_{tag}.csv",
        ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Storm-clustered bootstrap regime BSS from saved predictions"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--models", nargs="+", default=["tabular", "imagery", "fusion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--prob-col", default="prob_primary")
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    tag = args.tag or Path(args.config).stem

    print(
        "Note: stratified BSS uses the subset climatology "
        "(mean label within the regime), not the pooled test base rate."
    )
    print("Bootstrap resamples whole storms (cluster bootstrap).\n")

    all_rows = []
    all_deltas = []

    for seed in args.seeds:
        frames = {}
        for model in args.models:
            df = _load_pred(paths, model, tag, seed)
            if df is None:
                print(f"missing predictions for {model} seed={seed}")
                continue
            frames[model] = df
        if len(frames) < 1:
            continue

        print(f"=== seed {seed} ===")
        for model, df in frames.items():
            framed = assign_regimes(df)
            y = framed["label"].to_numpy(dtype=float)
            p = framed[args.prob_col].to_numpy(dtype=float)
            storms = framed["storm_id"].to_numpy()
            for axis in ("regime_intensity",):
                for name in sorted(framed[axis].dropna().unique().tolist()):
                    mask = framed[axis].to_numpy() == name
                    if mask.sum() == 0:
                        continue
                    yt, yp, st = y[mask], p[mask], storms[mask]
                    summary = summarize_predictions(yt, yp)
                    boot = bootstrap_bss(yt, yp, n_boot=args.n_boot, clusters=st)
                    n_storms = int(len(np.unique(st)))
                    n_ri = int(yt.sum())
                    row = {
                        "seed": seed,
                        "model": model,
                        "axis": axis,
                        "regime": name,
                        "n": summary["n"],
                        "n_storms": n_storms,
                        "n_ri": n_ri,
                        "ri_rate": summary["ri_rate"],
                        "climatology_ref": "subset_mean_label",
                        "brier": summary["brier"],
                        "bss": summary["brier_skill_vs_climatology"],
                        "pod": summary["pod"],
                        "success_ratio": summary["success_ratio"],
                        **boot,
                    }
                    all_rows.append(row)
                    if name == "Cat1-2_64-95kt":
                        print(
                            f"  {model:<10} Cat1-2 n={row['n']} storms={n_storms} ri={n_ri} "
                            f"bss={row['bss']:.4f} "
                            f"[{row['bss_lo']:.4f}, {row['bss_hi']:.4f}] "
                            f"pod={row['pod']:.3f}"
                        )

        if "tabular" not in frames:
            continue
        tab = assign_regimes(frames["tabular"])
        mask = tab["regime_intensity"].to_numpy() == "Cat1-2_64-95kt"
        print("  paired delta BSS vs tabular (storm-clustered):")
        for other in ("imagery", "fusion"):
            if other not in frames:
                continue
            odf = assign_regimes(frames[other])
            both = tab.loc[mask, ["storm_id", "time", "label", args.prob_col]].rename(
                columns={args.prob_col: "p_tab"}
            ).merge(
                odf[["storm_id", "time", args.prob_col]].rename(
                    columns={args.prob_col: "p_other"}
                ),
                on=["storm_id", "time"],
                how="inner",
            )
            y = both["label"].to_numpy(dtype=float)
            p_tab = both["p_tab"].to_numpy(dtype=float)
            p_other = both["p_other"].to_numpy(dtype=float)
            storms = both["storm_id"].to_numpy()
            delta = bootstrap_delta_bss(
                y, p_other, p_tab, n_boot=args.n_boot, clusters=storms
            )
            rec = {
                "seed": seed,
                "contrast": f"{other}_minus_tabular",
                "n": int(len(y)),
                "n_storms": int(len(np.unique(storms))),
                "n_ri": int(y.sum()),
                **delta,
            }
            all_deltas.append(rec)
            print(
                f"    {other}-tabular: {delta['delta_mean']:.4f} "
                f"[{delta['delta_lo']:.4f}, {delta['delta_hi']:.4f}] "
                f"storms={rec['n_storms']} ri={rec['n_ri']}"
            )

    out = {
        "tag": tag,
        "prob_col": args.prob_col,
        "n_boot": args.n_boot,
        "bootstrap": "storm_clustered",
        "climatology_ref": "subset_mean_label_within_regime",
        "rows": all_rows,
        "cat12_delta_bss": all_deltas,
    }
    out_path = paths["metrics"] / f"regime_bootstrap_{tag}.json"
    save_json(out, out_path)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
