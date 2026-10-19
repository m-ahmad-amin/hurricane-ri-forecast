from __future__ import annotations

from typing import Optional

import numpy as np


def period_index(periods: np.ndarray, hours: int) -> int:
    periods = np.asarray(periods).astype(int)
    matches = np.where(periods == hours)[0]
    if len(matches) == 0:
        raise ValueError(f"period {hours}h not found in {periods.tolist()}")
    return int(matches[0])


def ri_label_from_intensity_change(
    intensity_change: np.ndarray,
    periods: np.ndarray,
    hours: int = 24,
    threshold_kt: float = 30.0,
) -> tuple[Optional[int], Optional[float]]:
    idx = period_index(periods, hours)
    values = np.asarray(intensity_change, dtype=float).reshape(-1)
    if idx >= values.size:
        return None, None
    value = values[idx]
    if np.ma.is_masked(value) or not np.isfinite(value):
        return None, None
    return int(float(value) >= threshold_kt), float(value)
