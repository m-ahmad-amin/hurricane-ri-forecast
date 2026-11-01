import { riskLevel } from "../lib/risk";

type Props = {
  value: number; // 0–1
  empty?: boolean;
};

export default function RiGauge({ value, empty }: Props) {
  const p = empty ? 0 : Math.max(0, Math.min(1, value));
  const level = riskLevel(p);
  const size = 200;
  const stroke = 16;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const dash = empty ? c * 0.08 : c * p;
  const color = level === "high" ? "#dc2626" : level === "watch" ? "#d97706" : "#16a34a";

  return (
    <div className={`ri-gauge ${empty ? "empty" : ""}`} aria-hidden={empty}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#d8dee8" strokeWidth={stroke} />
        <circle
          className="ri-gauge-arc"
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={empty ? "#0a1931" : color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${c - dash}`}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          opacity={empty ? 0.35 : 1}
        />
      </svg>
      <div className="ri-gauge-center">
        {empty ? (
          <>
            <span className="ri-gauge-placeholder">-</span>
            <span className="ri-gauge-sub">RI chance</span>
          </>
        ) : (
          <>
            <span className="ri-gauge-pct">{Math.round(p * 100)}%</span>
            <span className="ri-gauge-sub">RI chance</span>
          </>
        )}
      </div>
    </div>
  );
}
