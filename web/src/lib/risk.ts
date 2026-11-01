export function pct(p: unknown, digits = 0) {
  const n = Number(p);
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(digits)}%`;
}

export function fmtTime(t: unknown) {
  const n = Number(t);
  if (!Number.isFinite(n)) return String(t ?? "-");
  const d = new Date(n * 1000);
  if (Number.isNaN(d.getTime())) return String(t);
  return d.toISOString().replace("T", " ").slice(0, 16) + "Z";
}

/** Main user-facing score: fusion isotonic P(RI in next 24h). */
export function riskLevel(p: number): "low" | "watch" | "high" {
  if (p >= 0.15) return "high";
  if (p >= 0.05) return "watch";
  return "low";
}

export function riskLabel(level: "low" | "watch" | "high") {
  if (level === "high") return "High RI chance";
  if (level === "watch") return "Elevated";
  return "Low RI chance";
}

export function riskVerdict(p: number) {
  const level = riskLevel(p);
  if (level === "high") return "Rapid intensification looks more likely in the next 24 hours.";
  if (level === "watch") return "Some signal of rapid intensification. Worth watching.";
  return "Rapid intensification looks unlikely in the next 24 hours.";
}
