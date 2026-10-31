from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _intensity_bin(v: float) -> str:
    if not np.isfinite(v):
        return "unknown"
    if v < 50:
        return "TS_<50kt"
    if v < 64:
        return "TS_50-63kt"
    if v < 96:
        return "Cat1-2_64-95kt"
    return "Cat3+_>=96kt"


def _shear_bin(v: float) -> str:
    if not np.isfinite(v):
        return "unknown"
    if v < 10:
        return "low_<10kt"
    if v < 20:
        return "mod_10-20kt"
    return "high_>=20kt"


def assign_regimes(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if "intensity_kt" in out.columns:
        out["regime_intensity"] = out["intensity_kt"].map(_intensity_bin)
    else:
        out["regime_intensity"] = "unknown"
    shear_col = "shear_high_0_800" if "shear_high_0_800" in out.columns else "shear_low_0_800"
    if shear_col in out.columns:
        out["regime_shear"] = out[shear_col].map(_shear_bin)
    else:
        out["regime_shear"] = "unknown"
    return out


def regime_breakdown(
    table: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    summarize: Callable[[np.ndarray, np.ndarray, str], dict],
    columns: tuple[str, ...] = ("regime_intensity", "regime_shear"),
) -> dict[str, dict[str, dict]]:
    framed = assign_regimes(table)
    out: dict[str, dict[str, dict]] = {}
    for col in columns:
        if col not in framed.columns:
            continue
        buckets: dict[str, dict] = {}
        for name in sorted(framed[col].dropna().unique().tolist()):
            mask = framed[col].to_numpy() == name
            buckets[name] = summarize(y_true[mask], y_prob[mask], f"{col}:{name}")
        out[col] = buckets
    return out
