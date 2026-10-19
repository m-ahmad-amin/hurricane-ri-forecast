from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    reference: np.ndarray | float | None = None,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bs = brier_score(y_true, y_prob)
    if reference is None:
        clim = float(np.mean(y_true))
        reference = np.full_like(y_true, clim)
    elif np.isscalar(reference):
        reference = np.full_like(y_true, float(reference))
    bs_ref = brier_score(y_true, reference)
    if bs_ref <= 1e-12:
        return float("nan")
    return float(1.0 - bs / bs_ref)


def binary_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_hat = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    tn = int(((y_hat == 0) & (y_true == 0)).sum())
    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    success_ratio = tp / (tp + fp) if (tp + fp) else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pod": pod,
        "success_ratio": success_ratio,
        "far": far,
    }


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    freq = []
    conf = []
    counts = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        centers.append(float(0.5 * (lo + hi)))
        freq.append(float(y_true[mask].mean()))
        conf.append(float(y_prob[mask].mean()))
        counts.append(int(mask.sum()))
    return {
        "bin_centers": centers,
        "observed_frequency": freq,
        "mean_confidence": conf,
        "counts": counts,
    }


def summarize_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    counts = binary_counts(y_true, y_prob, threshold=threshold)
    return {
        "n": int(len(y_true)),
        "ri_rate": float(np.mean(y_true)),
        "brier": brier_score(y_true, y_prob),
        "brier_skill_vs_climatology": brier_skill_score(y_true, y_prob),
        "threshold": threshold,
        **counts,
        "reliability": reliability_curve(y_true, y_prob),
    }


def bootstrap_bss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    values = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        values.append(brier_skill_score(y_true[idx], y_prob[idx]))
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"bss_mean": float("nan"), "bss_lo": float("nan"), "bss_hi": float("nan")}
    return {
        "bss_mean": float(np.mean(values)),
        "bss_lo": float(np.quantile(values, 0.025)),
        "bss_hi": float(np.quantile(values, 0.975)),
    }
