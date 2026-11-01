type Props = {
  value: number;
  label: string;
  sub?: string;
  accent?: string;
  size?: number;
};

export default function ProbRing({
  value,
  label,
  sub,
  accent = "#5eead4",
  size = 132,
}: Props) {
  const v = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  const r = 54;
  const c = 2 * Math.PI * r;
  const dash = c * v;

  return (
    <div className="ring" style={{ width: size }}>
      <svg width={size} height={size} viewBox="0 0 140 140" className="ring-svg">
        <circle cx="70" cy="70" r={r} className="ring-track" />
        <circle
          cx="70"
          cy="70"
          r={r}
          className="ring-value"
          style={{
            stroke: accent,
            strokeDasharray: `${dash} ${c}`,
          }}
        />
      </svg>
      <div className="ring-center">
        <strong>{(v * 100).toFixed(1)}%</strong>
        <span>{label}</span>
        {sub ? <em>{sub}</em> : null}
      </div>
    </div>
  );
}
