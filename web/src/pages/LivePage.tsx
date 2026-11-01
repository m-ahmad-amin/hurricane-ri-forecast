import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import RiGauge from "../components/RiGauge";
import { api } from "../api";
import { riskLabel, riskLevel, riskVerdict } from "../lib/risk";

type Storm = {
  id?: string;
  name?: string;
  basin?: string;
  latitude?: number;
  longitude?: number;
  intensity_kt?: number;
};

export default function LivePage() {
  const [storms, setStorms] = useState<Storm[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Storm | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .liveStorms()
      .then((res) => setStorms((res.storms || []) as Storm[]))
      .catch((e: Error) => setError(e.message));
  }, []);

  async function run(storm: Storm) {
    setBusy(true);
    setSelected(storm);
    setError(null);
    try {
      setResult(
        await api.livePredict({
          id: storm.id,
          name: storm.name,
          basin: storm.basin,
          latitude: storm.latitude,
          longitude: storm.longitude,
          intensity_kt: storm.intensity_kt,
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const pred = result?.prediction as Record<string, unknown> | undefined;
  const probs = pred?.probabilities as Record<string, number> | undefined;
  const p = Number(probs?.fusion_iso || 0);
  const level = riskLevel(p);
  const hasResult = Boolean(probs);

  return (
    <div className="predict-page">
      <div className="predict-intro">
        <h1>Live storms</h1>
        <p>
          Tap an active storm from the NHC feed, or{" "}
          <Link to="/explorer">enter details yourself</Link>.
        </p>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="predict-layout">
        <div className="predict-form">
          {storms.length === 0 ? (
            <p className="muted" style={{ margin: 0 }}>
              No active storms in the feed right now.
            </p>
          ) : (
            <div className="storm-list">
              {storms.map((s) => (
                <button
                  type="button"
                  key={String(s.id || s.name)}
                  className={`storm-row ${selected?.id === s.id ? "on" : ""}`}
                  onClick={() => run(s)}
                  disabled={busy}
                >
                  <span>
                    <strong>{s.name}</strong>
                    <div className="muted" style={{ fontSize: "0.8rem" }}>
                      {s.basin || "—"} · {s.intensity_kt != null ? `${s.intensity_kt} kt` : "n/a"}
                    </div>
                  </span>
                  <span className="btn btn-navy" style={{ padding: "0.45rem 0.85rem" }}>
                    Check
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className={`predict-result-card ${hasResult ? `tone-${level}` : ""}`}>
          <RiGauge value={p} empty={!hasResult} />
          {!hasResult ? (
            <p className="result-hint">{busy ? "Checking…" : "Pick a live storm to fill the gauge."}</p>
          ) : (
            <>
              <div className={`risk-chip lg ${level}`}>{riskLabel(level)}</div>
              <p className="result-verdict">{riskVerdict(p)}</p>
              <div className="result-meta">
                <span>{selected?.name || "Storm"}</span>
                <span>Experimental</span>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
