from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .tcprimed import (
    find_env_files,
    find_overpass_files,
    load_env_index,
    parse_storm_id,
    read_overpass_sample,
)


TABULAR_FEATURES = [
    "sst_c",
    "shear_low_0_800",
    "shear_high_0_800",
    "potential_intensity_kt",
    "pw_mean",
    "distance_to_land_km",
    "storm_speed",
    "intensity_kt",
    "ir_annulus_std_0",
    "ir_annulus_std_1",
    "ir_eye_max",
    "ir_eye_mean",
    "ir_pct_below_-50",
    "ir_pct_below_-30",
]


def _storm_dirs_from_env(env_files: list[Path]) -> list[Path]:
    return sorted({path.parent for path in env_files})


def _filter_storm_dir(
    storm_dir: Path,
    seasons: Optional[list[int]],
    basins: Optional[list[str]],
    storm_ids: Optional[list[str]],
) -> bool:
    env_files = list(storm_dir.glob("*_env_*.nc"))
    if not env_files:
        return False
    basin, number, season = parse_storm_id(env_files[0])
    if seasons is not None and season not in seasons:
        return False
    if basins is not None and basin not in basins:
        return False
    if storm_ids is not None and number not in storm_ids:
        return False
    return True


def _storm_key(storm_dir: Path, raw_root: Path) -> str:
    try:
        return storm_dir.resolve().relative_to(Path(raw_root).resolve()).as_posix()
    except ValueError:
        return storm_dir.as_posix()


def _key_from_safe(safe: str) -> str:
    return safe.replace("__", "/")


def _load_done(checkpoint_dir: Path) -> set[str]:
    done: set[str] = set()
    path = checkpoint_dir / "done.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            done.update(json.load(f))
    shard_dir = checkpoint_dir / "storms"
    if shard_dir.exists():
        for parquet_path in shard_dir.glob("*.parquet"):
            npy_path = parquet_path.with_suffix(".npy")
            if npy_path.exists():
                done.add(_key_from_safe(parquet_path.stem))
    return done


def _save_done(checkpoint_dir: Path, done: set[str]) -> None:
    merged = _load_done(checkpoint_dir) | set(done)
    path = checkpoint_dir / "done.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(merged), f, indent=2)


def _save_storm_shard(
    checkpoint_dir: Path,
    key: str,
    rows: list[dict[str, Any]],
    images: list[np.ndarray],
) -> None:
    safe = key.replace("/", "__")
    shard_dir = checkpoint_dir / "storms"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame(columns=["storm_id"]).to_parquet(shard_dir / f"{safe}.parquet", index=False)
        np.save(shard_dir / f"{safe}.npy", np.zeros((0, 1, 1, 1), dtype=np.float32))
        return
    df = pd.DataFrame(rows)
    for col in TABULAR_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    df.to_parquet(shard_dir / f"{safe}.parquet", index=False)
    np.save(shard_dir / f"{safe}.npy", np.stack(images).astype(np.float32))


def _load_all_shards(checkpoint_dir: Path, done: set[str]) -> tuple[pd.DataFrame, np.ndarray]:
    frames: list[pd.DataFrame] = []
    images: list[np.ndarray] = []
    shard_dir = checkpoint_dir / "storms"
    for key in sorted(done):
        safe = key.replace("/", "__")
        table_path = shard_dir / f"{safe}.parquet"
        image_path = shard_dir / f"{safe}.npy"
        if not table_path.exists() or not image_path.exists():
            continue
        df = pd.read_parquet(table_path)
        arr = np.load(image_path)
        if len(df) == 0:
            continue
        frames.append(df)
        images.append(arr)
    if not frames:
        raise RuntimeError(
            f"no usable samples found in checkpoint {checkpoint_dir}. "
            "Download more TC PRIMED storms or relax filters."
        )
    return pd.concat(frames, ignore_index=True), np.concatenate(images, axis=0)


def build_samples(
    raw_root: Path,
    image_size: int,
    ri_threshold_kt: float,
    ri_period_hours: int,
    max_gap_hours: float,
    seasons: Optional[list[int]] = None,
    basins: Optional[list[str]] = None,
    storm_ids: Optional[list[str]] = None,
    instruments: Optional[list[str]] = None,
    checkpoint_dir: Optional[Path] = None,
    shard: Optional[tuple[int, int]] = None,
    assemble_only: bool = False,
) -> tuple[Optional[pd.DataFrame], Optional[np.ndarray], bool]:
    """
    Returns (df, images, complete).
    complete=False means more storms are still pending (e.g. another shard worker).
    """
    env_files = find_env_files(raw_root)
    storm_dirs = [
        d
        for d in _storm_dirs_from_env(env_files)
        if _filter_storm_dir(d, seasons, basins, storm_ids)
    ]
    all_keys = [_storm_key(d, raw_root) for d in storm_dirs]

    if shard is not None:
        shard_id, num_shards = shard
        if num_shards < 1 or shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f"invalid shard {shard_id}/{num_shards}")
        indexed = [(i, d) for i, d in enumerate(storm_dirs) if i % num_shards == shard_id]
        assigned_dirs = [d for _, d in indexed]
        print(f"shard {shard_id}/{num_shards}: {len(assigned_dirs)}/{len(storm_dirs)} storms assigned")
    else:
        assigned_dirs = storm_dirs

    done: set[str] = set()
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        done = _load_done(checkpoint_dir)
        print(f"resume: {len(done)}/{len(storm_dirs)} storms already done")

    if assemble_only:
        done = _load_done(checkpoint_dir) if checkpoint_dir is not None else set()
        complete = checkpoint_dir is not None and set(all_keys).issubset(done)
        if not complete:
            missing = len(all_keys) - len(done & set(all_keys))
            print(f"assemble-only: still missing ~{missing} storms")
            return None, None, False
        df, images = _load_all_shards(checkpoint_dir, done)
        return df, images, True

    acc_rows: list[dict[str, Any]] = []
    acc_images: list[np.ndarray] = []

    storm_bar = tqdm(
        assigned_dirs,
        desc="storms",
        mininterval=5.0,
        maxinterval=60.0,
    )
    for storm_dir in storm_bar:
        key = _storm_key(storm_dir, raw_root)
        if checkpoint_dir is not None:
            done = _load_done(checkpoint_dir)
            if key in done:
                continue

        env_path = next(storm_dir.glob("*_env_*.nc"))
        env_index = load_env_index(env_path)
        overpasses = find_overpass_files(storm_dir, instruments=instruments)
        rows: list[dict[str, Any]] = []
        images: list[np.ndarray] = []
        for overpass in overpasses:
            sample = read_overpass_sample(
                overpass,
                env_index=env_index,
                ri_threshold_kt=ri_threshold_kt,
                ri_period_hours=ri_period_hours,
                image_size=image_size,
                max_gap_hours=max_gap_hours,
            )
            if sample is None:
                time.sleep(0.2)
                continue
            image = sample.pop("image")
            rows.append(sample)
            images.append(image)
            time.sleep(0.2)

        if checkpoint_dir is not None:
            _save_storm_shard(checkpoint_dir, key, rows, images)
            _save_done(checkpoint_dir, {key})
            done.add(key)
        else:
            acc_rows.extend(rows)
            acc_images.extend(images)

        storm_bar.set_postfix(storm=key, samples=len(rows), refresh=False)

    if checkpoint_dir is not None:
        done = _load_done(checkpoint_dir)
        complete = set(all_keys).issubset(done)
        if not complete:
            print(
                f"checkpoint incomplete: {len(done)}/{len(all_keys)} storms. "
                "Other shard workers may still be running. "
                "When all are done, run with --assemble-only."
            )
            return None, None, False
        df, images = _load_all_shards(checkpoint_dir, done)
        return df, images, True

    if not acc_rows:
        raise RuntimeError(
            f"no usable samples found under {raw_root}. "
            "Download more TC PRIMED storms or relax filters."
        )
    df = pd.DataFrame(acc_rows)
    for col in TABULAR_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df, np.stack(acc_images).astype(np.float32), True


def assign_split(
    df: pd.DataFrame,
    images: np.ndarray,
    train_seasons: list[int],
    test_seasons: list[int],
    within_storm_split: bool = False,
    val_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = df.copy().reset_index(drop=True)
    if within_storm_split or set(train_seasons) == set(test_seasons):
        storms = sorted(out["storm_id"].unique())
        rng = np.random.default_rng(seed)
        rng.shuffle(storms)
        if len(storms) == 1:
            n = len(out)
            idx = np.arange(n)
            rng.shuffle(idx)
            n_val = max(1, int(round(n * val_fraction)))
            n_test = max(1, int(round(n * val_fraction)))
            split = np.array(["train"] * n, dtype=object)
            split[idx[:n_test]] = "test"
            split[idx[n_test : n_test + n_val]] = "val"
            out["split"] = split
        else:
            n_test = max(1, int(round(len(storms) * val_fraction)))
            n_val = max(1, int(round(len(storms) * val_fraction)))
            test_storms = set(storms[:n_test])
            val_storms = set(storms[n_test : n_test + n_val])

            def _split(sid: str) -> str:
                if sid in test_storms:
                    return "test"
                if sid in val_storms:
                    return "val"
                return "train"

            out["split"] = out["storm_id"].map(_split)
        return out, images

    out["split"] = "drop"
    out.loc[out["season"].isin(train_seasons), "split"] = "train"
    out.loc[out["season"].isin(test_seasons), "split"] = "test"
    train_df = out[out["split"] == "train"]
    if len(train_df) == 0:
        raise RuntimeError("no train samples after season split")
    storms = sorted(train_df["storm_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(storms)
    n_val = max(1, int(round(len(storms) * val_fraction))) if len(storms) > 1 else 0
    val_storms = set(storms[:n_val])
    out.loc[out["storm_id"].isin(val_storms), "split"] = "val"
    keep = out["split"].to_numpy() != "drop"
    out = out.loc[keep].reset_index(drop=True)
    images = images[keep]
    return out, images


def save_processed(
    df: pd.DataFrame,
    images: np.ndarray,
    processed_dir: Path,
    tag: str,
) -> dict[str, Path]:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_path = processed_dir / f"samples_{tag}.parquet"
    image_path = processed_dir / f"images_{tag}.npy"
    meta_path = processed_dir / f"meta_{tag}.json"
    feature_path = processed_dir / f"features_{tag}.json"

    df.to_parquet(table_path, index=False)
    np.save(image_path, images)
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(TABULAR_FEATURES, f, indent=2)
    meta = {
        "n_samples": int(len(df)),
        "n_ri": int(df["ri_label"].sum()),
        "ri_rate": float(df["ri_label"].mean()),
        "splits": df["split"].value_counts().to_dict(),
        "seasons": sorted(map(int, df["season"].unique())),
        "basins": sorted(df["basin"].unique().tolist()),
        "image_shape": list(images.shape),
        "table_path": str(table_path),
        "image_path": str(image_path),
        "feature_path": str(feature_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return {
        "table": table_path,
        "images": image_path,
        "meta": meta_path,
        "features": feature_path,
    }
