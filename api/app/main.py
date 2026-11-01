from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .live import fetch_nhc_current_storms, live_predict_for_storm
from .store import analog_predict, filter_index, load_cases, load_summary

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "PRIMED RI Explorer API — probabilistic 24h tropical-cyclone rapid intensification "
        "on public TC PRIMED archive cases, plus experimental live analogs."
    ),
)

def _parse_cors_origins(raw: str) -> list[str]:
    """Normalize Render/Netlify env values (quotes, trailing slashes, commas)."""
    out: list[str] = []
    for part in (raw or "*").split(","):
        o = part.strip().strip("\"'")
        if not o:
            continue
        if o != "*":
            o = o.rstrip("/")
        out.append(o)
    return out or ["*"]


origins = _parse_cors_origins(settings.cors_origins)
# No cookies/auth on this API. credentials=False lets both "*" and explicit origins work.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LivePredictBody(BaseModel):
    basin: str | None = Field(default=None, examples=["AL"])
    intensity_kt: float | None = Field(default=None, examples=[65.0])
    latitude: float | None = Field(default=None, examples=[20.5])
    longitude: float | None = None
    name: str | None = None
    id: str | None = None
    k: int = Field(default=25, ge=5, le=100)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/meta")
def meta() -> dict[str, Any]:
    summary = load_summary()
    return {
        **summary,
        "endpoints": {
            "cases": "/v1/cases",
            "case": "/v1/cases/{case_id}",
            "live_storms": "/v1/live/storms",
            "live_predict": "POST /v1/live/predict",
        },
    }


@app.get("/v1/cases")
def list_cases(
    basin: str | None = None,
    season: int | None = None,
    storm_id: str | None = None,
    ri_only: bool | None = None,
    q: str | None = None,
    min_fusion: float | None = Query(default=None, ge=0.0, le=1.0),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    rows, total = filter_index(
        basin=basin,
        season=season,
        storm_id=storm_id,
        ri_only=ri_only,
        q=q,
        min_fusion=min_fusion,
        limit=limit,
        offset=offset,
    )
    return {"total": total, "limit": limit, "offset": offset, "cases": rows}


@app.get("/v1/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    case = load_cases().get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="case not found")
    return {
        "case": case,
        "probabilities": {
            "tabular": {
                "raw": case.get("p_tabular_raw"),
                "isotonic": case.get("p_tabular_iso"),
                "primary": case.get("p_tabular_primary"),
            },
            "imagery": {
                "raw": case.get("p_imagery_raw"),
                "isotonic": case.get("p_imagery_iso"),
                "primary": case.get("p_imagery_primary"),
            },
            "fusion": {
                "raw": case.get("p_fusion_raw"),
                "isotonic": case.get("p_fusion_iso"),
                "primary": case.get("p_fusion_primary"),
            },
        },
        "interpretation": _interpret(case),
        "disclaimer": load_summary().get("disclaimer"),
    }


def _interpret(case: dict[str, Any]) -> dict[str, Any]:
    t = float(case.get("p_tabular_iso") or 0)
    i = float(case.get("p_imagery_iso") or 0)
    f = float(case.get("p_fusion_iso") or 0)
    agree = max(t, i, f) - min(t, i, f) < 0.05
    leader = max(
        ("tabular", t),
        ("imagery", i),
        ("fusion", f),
        key=lambda x: x[1],
    )[0]
    return {
        "models_agree": agree,
        "highest_model": leader,
        "spread": round(max(t, i, f) - min(t, i, f), 4),
        "ri_observed": bool(case.get("ri_label")),
    }


@app.get("/v1/live/storms")
async def live_storms() -> dict[str, Any]:
    try:
        return await fetch_nhc_current_storms()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"NHC fetch failed: {exc}") from exc


@app.post("/v1/live/predict")
async def live_predict(body: LivePredictBody) -> dict[str, Any]:
    storm = {
        "id": body.id or body.name or "manual",
        "name": body.name or body.id or "manual",
        "basin": body.basin,
        "latitude": body.latitude,
        "longitude": body.longitude,
        "intensity_kt": body.intensity_kt,
    }
    if body.intensity_kt is None and body.latitude is None and body.basin is None:
        # try resolve from NHC list by id/name
        try:
            feed = await fetch_nhc_current_storms()
            match = None
            key = (body.id or body.name or "").lower()
            for s in feed.get("storms") or []:
                sid = str(s.get("id", "")).lower()
                sname = str(s.get("name", "")).lower()
                if key and (key in sid or key in sname):
                    match = s
                    break
            if match:
                storm.update(match)
        except Exception:
            pass
    result = await live_predict_for_storm(storm, k=body.k)
    if not result.get("prediction", {}).get("ok"):
        # still return structure
        result["prediction"] = analog_predict(
            basin=storm.get("basin"),
            intensity_kt=storm.get("intensity_kt"),
            latitude=storm.get("latitude"),
            k=body.k,
        )
    return result


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "health": "/health",
        "github": settings.github_url,
    }
