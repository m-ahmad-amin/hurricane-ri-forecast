const API_BASE = (import.meta.env.VITE_API_BASE || "http://localhost:8000").replace(/\/$/, "");

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

export type CaseIndex = {
  case_id: string;
  storm_id: string;
  basin: string;
  season: number;
  time: number | string;
  latitude: number;
  longitude: number;
  intensity_kt: number;
  ri_label: number;
  p_tabular_iso: number;
  p_imagery_iso: number;
  p_fusion_iso: number;
};

export type Meta = {
  n_cases: number;
  n_storms: number;
  n_ri: number;
  ri_rate: number;
  basins: string[];
  seasons: number[];
  disclaimer: string;
  skill_snapshot: Record<string, number>;
  github: string;
};

export const api = {
  base: API_BASE,
  meta: () => get<Meta>("/v1/meta"),
  cases: (params: URLSearchParams) =>
    get<{ total: number; cases: CaseIndex[] }>(`/v1/cases?${params}`),
  case: (id: string) => get<Record<string, unknown>>(`/v1/cases/${encodeURIComponent(id)}`),
  liveStorms: () => get<{ count: number; storms: Record<string, unknown>[]; disclaimer: string }>("/v1/live/storms"),
  livePredict: (body: Record<string, unknown>) =>
    post<Record<string, unknown>>("/v1/live/predict", body),
  health: () => get<{ status: string }>("/health"),
};
