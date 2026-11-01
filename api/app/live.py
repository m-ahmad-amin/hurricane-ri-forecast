from __future__ import annotations

from typing import Any

import httpx

from .config import settings
from .store import analog_predict


async def fetch_nhc_current_storms() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        r = await client.get(
            settings.nhc_current_storms_url,
            headers={"User-Agent": "PRIMED-RI-Explorer/1.0 (research; github.com/m-ahmad-amin/hurricane-ri-forecast)"},
        )
        r.raise_for_status()
        payload = r.json()

    storms_out: list[dict[str, Any]] = []
    # NHC JSON shape varies; handle common activeStorms / storm list patterns
    raw_list = (
        payload.get("activeStorms")
        or payload.get("storms")
        or payload.get("currentStorms")
        or []
    )
    if isinstance(payload, list):
        raw_list = payload

    for s in raw_list:
        if not isinstance(s, dict):
            continue
        name = s.get("name") or s.get("stormName") or s.get("id") or "Unknown"
        basin = s.get("basin") or s.get("basinId") or _infer_basin(s)
        lat = _num(s.get("latitude") or s.get("lat") or (s.get("center") or {}).get("lat"))
        lon = _num(s.get("longitude") or s.get("lon") or (s.get("center") or {}).get("lon"))
        wind = _num(s.get("intensity") or s.get("maxWind") or s.get("windSpeed"))
        movement = s.get("movement")
        if wind is None and isinstance(movement, dict):
            wind = _num(movement.get("windSpeed"))
        classif = s.get("classification") or s.get("intensityMisc") or {}
        if wind is None and isinstance(classif, dict):
            wind = _num(classif.get("intensity") or classif.get("wind"))
        storms_out.append(
            {
                "id": str(s.get("id") or s.get("stormNumber") or name),
                "name": str(name),
                "basin": basin,
                "latitude": lat,
                "longitude": lon,
                "intensity_kt": wind,
                "raw": {k: s[k] for k in list(s)[:12]},  # truncated for payload size
            }
        )

    return {
        "source": settings.nhc_current_storms_url,
        "count": len(storms_out),
        "storms": storms_out,
        "disclaimer": (
            "Live storm list from NHC public JSON when available. "
            "Predictions use experimental archive analogs, not operational guidance."
        ),
    }


def _infer_basin(s: dict[str, Any]) -> str | None:
    sid = str(s.get("id") or s.get("basin") or "").upper()
    if sid.startswith("AL") or "ATLANTIC" in sid:
        return "AL"
    if sid.startswith("EP") or "EAST" in sid:
        return "EP"
    if sid.startswith("CP"):
        return "CP"
    return None


def _num(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


async def live_predict_for_storm(storm: dict[str, Any], k: int = 25) -> dict[str, Any]:
    pred = analog_predict(
        basin=storm.get("basin"),
        intensity_kt=storm.get("intensity_kt"),
        latitude=storm.get("latitude"),
        k=k,
    )
    return {
        "storm": {
            "id": storm.get("id"),
            "name": storm.get("name"),
            "basin": storm.get("basin"),
            "latitude": storm.get("latitude"),
            "longitude": storm.get("longitude"),
            "intensity_kt": storm.get("intensity_kt"),
        },
        "prediction": pred,
        "disclaimer": (
            "Experimental live mode. Not NHC/NWS guidance. Not SHIPS-RII. "
            "Full GOES IR + TC PRIMED-matched environment inference is not enabled on this deploy yet."
        ),
    }
