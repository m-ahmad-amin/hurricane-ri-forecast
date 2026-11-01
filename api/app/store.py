from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@lru_cache(maxsize=1)
def load_summary() -> dict[str, Any]:
    return json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_index() -> list[dict[str, Any]]:
    return json.loads((DATA_DIR / "index.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_cases() -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    path = DATA_DIR / "cases.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cases[row["case_id"]] = row
    return cases


def filter_index(
    *,
    basin: str | None = None,
    season: int | None = None,
    storm_id: str | None = None,
    ri_only: bool | None = None,
    q: str | None = None,
    min_fusion: float | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    rows = load_index()
    out: list[dict[str, Any]] = []
    ql = (q or "").strip().lower()
    for r in rows:
        if basin and str(r.get("basin", "")).upper() != basin.upper():
            continue
        if season is not None and int(r.get("season") or -1) != season:
            continue
        if storm_id and str(r.get("storm_id", "")).upper() != storm_id.upper():
            continue
        if ri_only is True and int(r.get("ri_label") or 0) != 1:
            continue
        if ri_only is False and int(r.get("ri_label") or 0) != 0:
            continue
        if min_fusion is not None and float(r.get("p_fusion_iso") or 0) < min_fusion:
            continue
        if ql:
            blob = f"{r.get('storm_id','')} {r.get('basin','')} {r.get('time','')}".lower()
            if ql not in blob:
                continue
        out.append(r)
    total = len(out)
    return out[offset : offset + limit], total


def analog_predict(
    *,
    basin: str | None,
    intensity_kt: float | None,
    latitude: float | None,
    k: int = 25,
) -> dict[str, Any]:
    """Experimental live predictor: kNN analogs in the held-out archive."""
    cases = list(load_cases().values())
    scored: list[tuple[float, dict[str, Any]]] = []
    for c in cases:
        dist = 0.0
        if basin and c.get("basin"):
            dist += 0.0 if str(c["basin"]).upper() == basin.upper() else 2.5
        if intensity_kt is not None and c.get("intensity_kt") is not None:
            dist += abs(float(c["intensity_kt"]) - float(intensity_kt)) / 40.0
        if latitude is not None and c.get("latitude") is not None:
            dist += abs(float(c["latitude"]) - float(latitude)) / 20.0
        scored.append((dist, c))
    scored.sort(key=lambda x: x[0])
    top = [c for _, c in scored[:k]]
    if not top:
        return {"ok": False, "error": "no analogs"}

    def mean(key: str) -> float:
        vals = [float(c[key]) for c in top if c.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "ok": True,
        "method": "archive_knn_analog",
        "k": k,
        "note": (
            "Experimental. Uses nearest TC PRIMED archive cases by basin/intensity/latitude. "
            "Not a full live IR+environment inference pipeline."
        ),
        "probabilities": {
            "tabular_iso": round(mean("p_tabular_iso"), 4),
            "imagery_iso": round(mean("p_imagery_iso"), 4),
            "fusion_iso": round(mean("p_fusion_iso"), 4),
        },
        "analogs": [
            {
                "case_id": c["case_id"],
                "storm_id": c["storm_id"],
                "time": c["time"],
                "intensity_kt": c.get("intensity_kt"),
                "ri_label": c.get("ri_label"),
                "p_fusion_iso": c.get("p_fusion_iso"),
            }
            for c in top[:8]
        ],
    }
