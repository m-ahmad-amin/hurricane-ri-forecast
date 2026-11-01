import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { fmtTime, pct, riskLabel, riskLevel, riskVerdict } from "../lib/risk";

export default function CasePage() {
  const { caseId } = useParams();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showTech, setShowTech] = useState(false);

  useEffect(() => {
    if (!caseId) return;
    api
      .case(caseId)
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, [caseId]);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading…</p>;

  const c = data.case as Record<string, unknown>;
  const probs = data.probabilities as Record<string, Record<string, number>>;
  const p = Number(probs.fusion?.isotonic || 0);
  const level = riskLevel(p);

  return (
    <div className="storms-page">
      <Link className="back-link" to="/explorer">
        ← All storms
      </Link>

      <div className={`verdict-panel risk-${level}`}>
        <div className="verdict-kicker">{String(c.storm_id)} · {fmtTime(c.time)}</div>
        <div className="verdict-chance">{pct(p, 0)}</div>
        <div className={`risk-chip lg ${level}`}>{riskLabel(level)}</div>
        <p className="verdict-text">{riskVerdict(p)}</p>
        <div className="verdict-facts">
          <span>{Number(c.intensity_kt).toFixed(0)} kt</span>
          <span>{String(c.basin)}</span>
          <span>{c.ri_label ? "What happened: did RI" : "What happened: no RI"}</span>
        </div>
      </div>

      <button type="button" className="tech-toggle" onClick={() => setShowTech((v) => !v)}>
        {showTech ? "Hide model details" : "Show model details"}
      </button>

      {showTech && (
        <div className="tech-box">
          <p className="muted">For researchers: three model pathways used to build the score above.</p>
          <div className="tech-row">
            <span>Environment + scalars</span>
            <strong>{pct(probs.tabular?.isotonic, 1)}</strong>
          </div>
          <div className="tech-row">
            <span>Satellite imagery</span>
            <strong>{pct(probs.imagery?.isotonic, 1)}</strong>
          </div>
          <div className="tech-row">
            <span>Combined model</span>
            <strong>{pct(probs.fusion?.isotonic, 1)}</strong>
          </div>
          {(c.sst_c != null || c.shear_low_0_800 != null) && (
            <div className="tech-env">
              {c.sst_c != null && <span>Sea temp {Number(c.sst_c).toFixed(1)}°C</span>}
              {c.shear_low_0_800 != null && <span>Shear {Number(c.shear_low_0_800).toFixed(1)}</span>}
              {c.distance_to_land_km != null && (
                <span>Land {Number(c.distance_to_land_km).toFixed(0)} km</span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
