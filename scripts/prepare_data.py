from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prepare import assign_split, build_samples, save_processed
from src.utils import ensure_dirs, load_config, project_paths, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RI samples from TC PRIMED")
    parser.add_argument("--config", default="configs/al062018.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Optional local TC PRIMED root (use on Colab to avoid Drive read drops)",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    paths = project_paths(cfg, ROOT)
    ensure_dirs(paths)

    tag = args.tag or Path(args.config).stem
    subset = cfg.get("subset") or {}
    use_subset = bool(subset)
    raw_root = Path(args.raw_root) if args.raw_root else paths["raw_tcprimed"]

    if use_subset:
        seasons = subset.get("seasons", cfg["train_seasons"])
        basins = subset.get("basins", cfg["basins"])
        storm_ids = subset.get("storm_ids")
        instruments = subset.get("instruments")
    else:
        seasons = sorted(set(cfg["train_seasons"] + cfg["test_seasons"]))
        basins = cfg["basins"]
        storm_ids = None
        instruments = cfg.get("instruments")

    checkpoint_dir = paths["interim"] / f"prepare_{tag}"
    df, images = build_samples(
        raw_root=raw_root,
        image_size=cfg["image_size"],
        ri_threshold_kt=cfg["ri_threshold_kt"],
        ri_period_hours=cfg["ri_period_hours"],
        max_gap_hours=cfg["max_gap_hours"],
        seasons=seasons,
        basins=basins,
        storm_ids=storm_ids,
        instruments=instruments,
        checkpoint_dir=checkpoint_dir,
    )

    df, images = assign_split(
        df,
        images,
        train_seasons=cfg["train_seasons"],
        test_seasons=cfg["test_seasons"],
        within_storm_split=use_subset or set(cfg["train_seasons"]) == set(cfg["test_seasons"]),
        val_fraction=cfg.get("train", {}).get("val_fraction", 0.25),
        seed=cfg.get("train", {}).get("seed", 42),
    )

    if len(df) != len(images):
        raise RuntimeError("internal mismatch between table and images after split")

    saved = save_processed(df, images, paths["processed"], tag=tag)
    summary = {
        "tag": tag,
        "subset": use_subset,
        "n_samples": int(len(df)),
        "n_ri": int(df["ri_label"].sum()),
        "ri_rate": float(df["ri_label"].mean()),
        "splits": df["split"].value_counts().to_dict(),
        "paths": {k: str(v) for k, v in saved.items()},
    }
    save_json(summary, paths["metrics"] / f"prepare_{tag}.json")
    print(summary)


if __name__ == "__main__":
    main()
