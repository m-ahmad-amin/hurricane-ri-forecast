type Row = { key: string; label: string; value: number; tone?: "tab" | "img" | "fus" };

export default function ModelBars({ rows }: { rows: Row[] }) {
  const max = Math.max(...rows.map((r) => r.value), 0.01);
  return (
    <div className="bars">
      {rows.map((r) => (
        <div className="bar-row" key={r.key}>
          <div className="bar-meta">
            <span>{r.label}</span>
            <strong>{(r.value * 100).toFixed(1)}%</strong>
          </div>
          <div className="bar-track">
            <div
              className={`bar-fill ${r.tone === "img" ? "img" : r.tone === "fus" ? "fus" : ""}`}
              style={{ width: `${(r.value / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
