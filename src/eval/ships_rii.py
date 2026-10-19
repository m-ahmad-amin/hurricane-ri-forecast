from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.train.metrics import brier_skill_score, summarize_predictions


def attach_ships_rii(samples: pd.DataFrame, ships_csv: Path) -> pd.DataFrame:
    ships = pd.read_csv(ships_csv)
    return samples.merge(ships, on=["storm_id", "time"], how="left")


def compare_to_ships_rii(
    y_true: np.ndarray,
    model_prob: np.ndarray,
    ships_prob: np.ndarray,
) -> dict:
    mask = np.isfinite(ships_prob)
    if mask.sum() == 0:
        return {"available": False}
    return {
        "available": True,
        "n": int(mask.sum()),
        "model": summarize_predictions(y_true[mask], model_prob[mask]),
        "ships_rii": summarize_predictions(y_true[mask], ships_prob[mask]),
        "bss_model_vs_ships_rii": brier_skill_score(
            y_true[mask], model_prob[mask], reference=ships_prob[mask]
        ),
    }
