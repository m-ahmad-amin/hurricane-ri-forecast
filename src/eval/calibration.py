from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def probs_to_logits(p: np.ndarray) -> np.ndarray:
    p = _clip_prob(p)
    return np.log(p / (1.0 - p))


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float) / max(temperature, 1e-4)))


def fit_temperature(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    logits = probs_to_logits(probs)

    def nll(temp: float) -> float:
        p = temperature_scale(logits, float(temp))
        p = _clip_prob(p)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


def fit_platt(y_true: np.ndarray, probs: np.ndarray) -> LogisticRegression:
    """1-D logistic (Platt-style) calibrator on predicted probabilities."""
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    x = np.asarray(probs, dtype=float).reshape(-1, 1)
    model.fit(x, np.asarray(y_true, dtype=int))
    return model


def apply_platt(model: LogisticRegression, probs: np.ndarray) -> np.ndarray:
    x = np.asarray(probs, dtype=float).reshape(-1, 1)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def fit_isotonic(y_true: np.ndarray, probs: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(np.asarray(probs, dtype=float), np.asarray(y_true, dtype=float))
    return model


def apply_isotonic(model: IsotonicRegression, probs: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(np.asarray(probs, dtype=float)), dtype=float)
