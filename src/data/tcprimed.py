from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Optional

import netCDF4 as nc
import numpy as np
from PIL import Image


STORM_RE = re.compile(
    r"TCPRIMED_v\d+r\d+-(?:final|preliminary)_([A-Z]{2})(\d{2})(\d{4})_"
)


def open_dataset(path: Path, retries: int = 8, delay: float = 3.0):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return nc.Dataset(path)
        except (OSError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 == retries:
                break
            time.sleep(delay * (attempt + 1))
    raise last_error


def find_env_files(raw_root: Path) -> list[Path]:
    return sorted(Path(raw_root).rglob("*_env_*.nc"))


def find_overpass_files(
    storm_dir: Path,
    instruments: list[str] | None = None,
) -> list[Path]:
    files = []
    for path in sorted(storm_dir.glob("*.nc")):
        name = path.name
        if "_env_" in name:
            continue
        if instruments:
            if not any(f"_{inst}_" in name for inst in instruments):
                continue
        files.append(path)
    return files


def parse_storm_id(path: Path) -> tuple[str, str, int]:
    match = STORM_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot parse storm id from {path.name}")
    basin, number, season = match.groups()
    return basin, number, int(season)


def _to_float(value: Any) -> Optional[float]:
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return None
    item = arr[0]
    if np.ma.is_masked(item):
        return None
    item = float(item)
    if not np.isfinite(item):
        return None
    return item


def _safe_mean(arr: np.ndarray) -> float:
    data = np.asarray(arr, dtype=float)
    if np.ma.isMaskedArray(data):
        data = np.ma.filled(data, np.nan)
    if data.size == 0 or np.all(~np.isfinite(data)):
        return float("nan")
    return float(np.nanmean(data))


def nearest_time_index(times: np.ndarray, target: float, max_gap_seconds: float) -> Optional[int]:
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return None
    diffs = np.abs(times - target)
    idx = int(np.nanargmin(diffs))
    if not np.isfinite(diffs[idx]) or diffs[idx] > max_gap_seconds:
        return None
    return idx


def load_env_index(env_path: Path) -> dict[str, Any]:
    ds = open_dataset(env_path)
    try:
        diagnostics = ds["diagnostics"]
        storm = ds["storm_metadata"]
        return {
            "path": env_path,
            "diag_time": np.asarray(diagnostics["time"][:], dtype=float),
            "sst": np.asarray(diagnostics["sst"][:], dtype=float),
            "shear_magnitude": np.asarray(diagnostics["shear_magnitude"][:], dtype=float),
            "potential_intensity": np.asarray(
                diagnostics["potential_intensity_empirical"][:], dtype=float
            ),
            "precipitable_water": np.asarray(diagnostics["precipitable_water"][:], dtype=float),
            "distance_to_land": np.asarray(diagnostics["distance_to_land"][:], dtype=float),
            "storm_speed": np.asarray(storm["storm_speed"][:], dtype=float),
            "intensity": np.asarray(storm["intensity"][:], dtype=float),
            "storm_time": np.asarray(storm["time"][:], dtype=float),
        }
    finally:
        ds.close()


def extract_env_features(env_index: dict[str, Any], target_time: float, max_gap_hours: float) -> Optional[dict[str, float]]:
    idx = nearest_time_index(
        env_index["diag_time"],
        target_time,
        max_gap_seconds=max_gap_hours * 3600.0,
    )
    if idx is None:
        return None

    shear = env_index["shear_magnitude"][idx]
    pw = env_index["precipitable_water"][idx]
    sst = _safe_mean(env_index["sst"][idx])
    pot = _safe_mean(env_index["potential_intensity"][idx])
    dist = _to_float(env_index["distance_to_land"][idx])
    speed = _to_float(env_index["storm_speed"][idx]) if idx < len(env_index["storm_speed"]) else None

    storm_idx = nearest_time_index(
        env_index["storm_time"],
        target_time,
        max_gap_seconds=max_gap_hours * 3600.0,
    )
    intensity = (
        _to_float(env_index["intensity"][storm_idx]) if storm_idx is not None else None
    )

    features = {
        "sst_c": sst - 273.15 if np.isfinite(sst) else float("nan"),
        "shear_low_0_800": _safe_mean(shear[0]) if shear.ndim >= 1 else float("nan"),
        "shear_high_0_800": _safe_mean(shear[1]) if shear.ndim >= 1 else float("nan"),
        "potential_intensity_kt": pot,
        "pw_mean": _safe_mean(pw),
        "distance_to_land_km": dist if dist is not None else float("nan"),
        "storm_speed": speed if speed is not None else float("nan"),
        "intensity_kt": intensity if intensity is not None else float("nan"),
    }
    return features


def extract_ir_summary(ir_group) -> Optional[dict[str, float]]:
    flag = _to_float(ir_group["infrared_availability_flag"][:])
    if flag is None or flag < 1:
        return None
    if "IRWIN" not in ir_group.variables:
        return None

    pct = np.asarray(ir_group["percent_pixel_below_threshold"][:], dtype=float).reshape(-1)
    thresholds = np.asarray(ir_group["brightness_temperature_thresholds"][:], dtype=int)
    ann_std = np.asarray(
        ir_group["brightness_temperature_annulus_standard_deviation"][:], dtype=float
    ).reshape(-1)
    eye_max = _to_float(ir_group["brightness_temperature_eye_maximum"][:])
    eye_mean = _to_float(ir_group["brightness_temperature_eye_mean"][:])

    summary = {
        "ir_annulus_std_0": float(ann_std[0]) if ann_std.size > 0 else float("nan"),
        "ir_annulus_std_1": float(ann_std[1]) if ann_std.size > 1 else float("nan"),
        "ir_eye_max": eye_max if eye_max is not None else float("nan"),
        "ir_eye_mean": eye_mean if eye_mean is not None else float("nan"),
    }
    for thr, value in zip(thresholds, pct):
        summary[f"ir_pct_below_{int(thr)}"] = float(value)
    return summary


def load_ir_image(ir_group, image_size: int) -> Optional[np.ndarray]:
    flag = _to_float(ir_group["infrared_availability_flag"][:])
    if flag is None or flag < 1 or "IRWIN" not in ir_group.variables:
        return None
    arr = np.asarray(ir_group["IRWIN"][:], dtype=np.float32)
    if np.ma.isMaskedArray(arr):
        arr = np.ma.filled(arr, np.nan)
    if not np.isfinite(arr).any():
        return None

    finite = arr[np.isfinite(arr)]
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0)
    img = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    out = np.asarray(img, dtype=np.float32) / 255.0
    return out[None, :, :]


def read_overpass_sample(
    path: Path,
    env_index: dict[str, Any],
    ri_threshold_kt: float,
    ri_period_hours: int,
    image_size: int,
    max_gap_hours: float,
) -> Optional[dict[str, Any]]:
    from .labels import ri_label_from_intensity_change

    ds = open_dataset(path)
    try:
        if "infrared" not in ds.groups:
            return None
        ir_summary = extract_ir_summary(ds["infrared"])
        if ir_summary is None:
            return None
        image = load_ir_image(ds["infrared"], image_size=image_size)
        if image is None:
            return None

        osm = ds["overpass_storm_metadata"]
        periods = np.asarray(osm["intensity_change_periods"][:])
        intensity_change = np.asarray(osm["intensity_change"][:])
        label, delta = ri_label_from_intensity_change(
            intensity_change,
            periods,
            hours=ri_period_hours,
            threshold_kt=ri_threshold_kt,
        )
        if label is None:
            return None

        time_val = _to_float(osm["time"][:])
        if time_val is None:
            return None

        env_feats = extract_env_features(env_index, time_val, max_gap_hours=max_gap_hours)
        if env_feats is None:
            return None

        basin, number, season = parse_storm_id(path)
        intensity = _to_float(osm["intensity"][:])
        lat = _to_float(osm["storm_latitude"][:])
        lon = _to_float(osm["storm_longitude"][:])

        sample = {
            "storm_id": f"{basin}{number}{season}",
            "basin": basin,
            "cyclone_number": number,
            "season": season,
            "overpass_file": path.name,
            "time": time_val,
            "latitude": lat if lat is not None else float("nan"),
            "longitude": lon if lon is not None else float("nan"),
            "intensity_kt": intensity if intensity is not None else env_feats["intensity_kt"],
            "delta_v_24h": delta,
            "ri_label": label,
            "image": image,
        }
        sample.update(env_feats)
        sample.update(ir_summary)
        return sample
    finally:
        ds.close()
