import { useState } from "react";
import EstimateProgress from "../components/EstimateProgress";
import RiGauge from "../components/RiGauge";
import { api } from "../api";
import { riskLabel, riskLevel, riskVerdict } from "../lib/risk";

export default function ExplorerPage() {
  const [name, setName] = useState("");
  const [basin, setBasin] = useState("AL");
  const [intensity, setIntensity] = useState("75");
  const [lat, setLat] = useState("18");
  const [lon, setLon] = useState("-60");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);

  async function predict(e?: React.FormEvent) {
    e?.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.livePredict({
        id: "user",
        name: name.trim() || "Your storm",
        basin,
        intensity_kt: Number(intensity),
        latitude: Number(lat),
        longitude: Number(lon),
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  const pred = result?.prediction as Record<string, unknown> | undefined;
  const probs = pred?.probabilities as Record<string, number> | undefined;
  const p = Number(probs?.fusion_iso || 0);
  const level = riskLevel(p);
  const hasResult = Boolean(probs) && !busy;

  return (
    <div className="predict-page">
      <div className="predict-intro">
        <h1>Check RI chance</h1>
        <p>Enter what you know about the storm. We will estimate the chance it rapidly intensifies in the next 24 hours.</p>
      </div>

      <div className="predict-layout">
        <form className="predict-form" onSubmit={predict}>
          <label className="field">
            Storm name <span className="opt">(optional)</span>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Idalia" />
          </label>

          <div className="field-row">
            <label className="field">
              Basin
              <div className="choice-row">
                <button
                  type="button"
                  className={`choice ${basin === "AL" ? "on" : ""}`}
                  onClick={() => setBasin("AL")}
                >
                  Atlantic
                </button>
                <button
                  type="button"
                  className={`choice ${basin === "EP" ? "on" : ""}`}
                  onClick={() => setBasin("EP")}
                >
                  East Pacific
                </button>
              </div>
            </label>
          </div>

          <label className="field">
            Current wind speed
            <div className="slider-wrap">
              <input
                type="range"
                min={25}
                max={160}
                step={5}
                value={Number(intensity) || 75}
                onChange={(e) => setIntensity(e.target.value)}
              />
              <strong>{intensity || "-"} kt</strong>
            </div>
          </label>

          <div className="field-row two">
            <label className="field">
              Latitude
              <input
                type="number"
                step="0.1"
                value={lat}
                onChange={(e) => setLat(e.target.value)}
                placeholder="18"
              />
            </label>
            <label className="field">
              Longitude
              <input
                type="number"
                step="0.1"
                value={lon}
                onChange={(e) => setLon(e.target.value)}
                placeholder="-60"
              />
            </label>
          </div>

          {error && <p className="error">{error}</p>}

          <button type="submit" className="btn btn-navy predict-go" disabled={busy}>
            {busy ? "Estimating…" : "Estimate RI chance"}
          </button>
          <p className="form-note">Experimental research estimate. Not official forecast guidance.</p>
        </form>

        <div className={`predict-result-card ${hasResult ? `tone-${level}` : ""} ${busy ? "estimating" : ""}`}>
          {hasResult ? (
            <>
              <RiGauge value={p} />
              <div className={`risk-chip lg ${level}`}>{riskLabel(level)}</div>
              <p className="result-verdict">{riskVerdict(p)}</p>
              <div className="result-meta">
                <span>{name.trim() || "Your storm"}</span>
                <span>{basin === "AL" ? "Atlantic" : "East Pacific"}</span>
                <span>{intensity} kt</span>
              </div>
            </>
          ) : (
            <>
              <RiGauge value={0} empty={!busy} loading={busy} />
              {!busy && <p className="result-hint">Hit estimate to fill the gauge.</p>}
              <EstimateProgress active={busy} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
