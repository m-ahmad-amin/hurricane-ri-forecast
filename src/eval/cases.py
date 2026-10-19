from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pick_cases(
    table: pd.DataFrame,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, dict]:
    df = table.copy().reset_index(drop=True)
    df["prob"] = probs
    df["pred"] = (df["prob"] >= threshold).astype(int)

    hit = df[(df["ri_label"] == 1) & (df["pred"] == 1)].sort_values("prob", ascending=False)
    miss = df[(df["ri_label"] == 1) & (df["pred"] == 0)].sort_values("prob", ascending=True)
    false_alarm = df[(df["ri_label"] == 0) & (df["pred"] == 1)].sort_values("prob", ascending=False)

    def _row(frame: pd.DataFrame) -> dict | None:
        if len(frame) == 0:
            return None
        row = frame.iloc[0]
        return {
            "storm_id": row["storm_id"],
            "season": int(row["season"]),
            "basin": row["basin"],
            "time": float(row["time"]),
            "prob": float(row["prob"]),
            "ri_label": int(row["ri_label"]),
            "delta_v_24h": float(row["delta_v_24h"]),
            "overpass_file": row["overpass_file"],
            "table_index": int(row.name),
        }

    return {
        "hit": _row(hit),
        "miss": _row(miss),
        "false_alarm": _row(false_alarm),
    }


def save_case_figure(
    images: np.ndarray,
    cases: dict[str, dict | None],
    out_path: Path,
) -> None:
    labels = ["hit", "miss", "false_alarm"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, name in zip(axes, labels):
        case = cases.get(name)
        ax.axis("off")
        if case is None:
            ax.set_title(f"{name}: none")
            continue
        idx = case["table_index"]
        ax.imshow(images[idx, 0], cmap="gray")
        ax.set_title(
            f"{name}\n{case['storm_id']} p={case['prob']:.2f}\n"
            f"dV24={case['delta_v_24h']:.0f} kt"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_reliability_figure(reliability: dict, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    if reliability["bin_centers"]:
        ax.plot(
            reliability["mean_confidence"],
            reliability["observed_frequency"],
            marker="o",
            label="model",
        )
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed RI frequency")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
