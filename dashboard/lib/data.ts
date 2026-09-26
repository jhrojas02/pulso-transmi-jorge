// Todo lo de este archivo corre EXCLUSIVAMENTE en el servidor (Server
// Components / route handlers de Next.js). Nunca se importa desde un
// componente cliente, así que SUPABASE_SERVICE_KEY y PULSO_API_KEY nunca
// llegan al navegador. No usar variables NEXT_PUBLIC_* para estos secretos.

const SUPABASE_URL = process.env.SUPABASE_URL!;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY!;
const PULSO_API_BASE =
  process.env.PULSO_API_BASE ?? "https://pulso-transmi.72-60-245-2.sslip.io";
const PULSO_API_KEY = process.env.PULSO_API_KEY!;

async function sb(table: string, params: Record<string, string> = {}) {
  const qs = new URLSearchParams(params).toString();
  const url = `${SUPABASE_URL}/rest/v1/${table}${qs ? `?${qs}` : ""}`;
  const res = await fetch(url, {
    headers: {
      apikey: SUPABASE_SERVICE_KEY,
      Authorization: `Bearer ${SUPABASE_SERVICE_KEY}`,
    },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Supabase ${table} -> ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

async function pulso(path: string) {
  const res = await fetch(`${PULSO_API_BASE}${path}`, {
    headers: { Authorization: `Bearer ${PULSO_API_KEY}` },
    cache: "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

export type ChampionEntry = {
  horizon_min: number;
  model_id: string;
  promoted_at: string;
  version?: string;
  trained_at?: string;
  cutoff_train_fin?: string;
  algoritmo?: string;
};

export type StationMetric = {
  station_id: string;
  station_name?: string;
  horizon_min: number;
  accuracy: number | null;
  wape: number | null;
  coverage: number;
  n_evaluable: number;
  n_expected: number;
};

export type HorizonDrift = {
  horizon_min: number;
  cumulative_accuracy: number | null;
  rolling_24h_accuracy: number | null;
  delta: number | null;
};

export type RunRow = {
  run_id: string;
  run_at: string;
  status: string;
  decision_reentrenar: boolean;
  motivo_decision: string;
  model_id: string | null;
};

export type LeaderboardRow = {
  display_name: string;
  accuracy: number | null;
  raw_wape?: number | null;
  coverage: number | null;
  rank: number | null;
  [key: string]: unknown;
};

export type StationGeo = {
  station_id: string;
  station_name: string;
  corridor: string;
  latitude: number;
  longitude: number;
  accuracy: number | null;
};

export type TrendPoint = { t: string; accuracy: number };
export type AccuracyTrend = { horizon_min: number; points: TrendPoint[] };

export type DashboardData = {
  fetchedAt: string;
  collector: { lastSyncAt: string | null; lagMinutes: number | null };
  champions: ChampionEntry[];
  stationMetrics: StationMetric[];
  stationsGeo: StationGeo[];
  trend: AccuracyTrend[];
  drift: HorizonDrift[];
  recentRuns: RunRow[];
  trainingRuns: RunRow[];
  leaderboard: { self: LeaderboardRow | null; top: LeaderboardRow[]; error: string | null };
};

async function getCollectorLag() {
  const rows = await sb("sync_state", { select: "source,cursor_value,updated_at" });
  const row = rows.find((r: any) => r.source === "observations_stream") ?? rows[0] ?? null;
  if (!row) return { lastSyncAt: null, lagMinutes: null };
  const lastSyncAt = row.updated_at as string;
  const lagMinutes = Math.round((Date.now() - new Date(lastSyncAt).getTime()) / 60000);
  return { lastSyncAt, lagMinutes };
}

async function getChampions(): Promise<ChampionEntry[]> {
  const champions = await sb("champion", { select: "*", order: "horizon_min.asc" });
  const modelIds = champions.map((c: any) => c.model_id);
  if (modelIds.length === 0) return [];
  const modelos = await sb("modelo", {
    select: "model_id,version,trained_at,cutoff_train_fin,algoritmo",
    model_id: `in.(${modelIds.join(",")})`,
  });
  const byId = new Map(modelos.map((m: any) => [m.model_id, m]));
  return champions.map((c: any) => ({ ...c, ...(byId.get(c.model_id) ?? {}) }));
}

async function getStationMetrics(): Promise<StationMetric[]> {
  const [metrics, stations] = await Promise.all([
    sb("operational_metric", {
      select: "station_id,horizon_min,accuracy,wape,coverage,n_evaluable,n_expected,computed_at",
      window_kind: "eq.cumulative",
      station_id: "not.is.null",
      order: "computed_at.desc",
    }),
    sb("estacion", { select: "station_id,station_name" }),
  ]);
  const nameById = new Map(stations.map((s: any) => [s.station_id, s.station_name]));
  const seen = new Set<string>();
  const latest: StationMetric[] = [];
  for (const m of metrics) {
    const key = `${m.station_id}::${m.horizon_min}`;
    if (seen.has(key)) continue;
    seen.add(key);
    latest.push({ ...m, station_name: nameById.get(m.station_id) });
  }
  return latest.sort((a, b) => a.station_id.localeCompare(b.station_id) || a.horizon_min - b.horizon_min);
}

async function getStationsGeo(stationMetrics: StationMetric[]): Promise<StationGeo[]> {
  const stations = await sb("estacion", { select: "station_id,station_name,corridor,latitude,longitude" });
  const avgAccByStation = new Map<string, number[]>();
  for (const m of stationMetrics) {
    if (m.accuracy == null) continue;
    const arr = avgAccByStation.get(m.station_id) ?? [];
    arr.push(m.accuracy);
    avgAccByStation.set(m.station_id, arr);
  }
  return stations.map((s: any) => {
    const accs = avgAccByStation.get(s.station_id);
    const accuracy = accs && accs.length ? accs.reduce((a, b) => a + b, 0) / accs.length : null;
    return { ...s, accuracy };
  });
}

async function getAccuracyTrend(): Promise<AccuracyTrend[]> {
  // Solo necesitamos ~72h de historial (downsample a 1h más abajo), así que
  // le pedimos a Supabase que filtre por fecha en la query en vez de traer
  // todo el histórico acumulado (semanas de filas, una por cada cron de
  // ~10 min) y recortarlo después en JS — evita el egress innecesario.
  const since = new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString();
  const rows = await sb("operational_metric", {
    select: "computed_at,horizon_min,accuracy",
    window_kind: "eq.cumulative",
    station_id: "is.null",
    computed_at: `gte.${since}`,
    order: "computed_at.asc",
  });
  const horizons = [15, 30, 45, 60];
  return horizons.map((h) => {
    const forH = rows.filter((r: any) => r.horizon_min === h);
    // Downsample a buckets de 1h (toma el último valor de cada hora) — la
    // tabla tiene una fila por cada despertar del cron (~cada 10 min), casi
    // siempre repitiendo el mismo valor; una hora es suficiente resolución
    // para ver la tendencia sin miles de puntos redundantes.
    const byHour = new Map<string, TrendPoint>();
    for (const r of forH) {
      const d = new Date(r.computed_at);
      d.setMinutes(0, 0, 0);
      byHour.set(d.toISOString(), { t: d.toISOString(), accuracy: r.accuracy });
    }
    const points = Array.from(byHour.values()).sort((a, b) => a.t.localeCompare(b.t));
    return { horizon_min: h, points: points.slice(-72) };
  });
}

async function getDrift(): Promise<HorizonDrift[]> {
  const [cumulative, rolling] = await Promise.all([
    sb("operational_metric", {
      select: "horizon_min,accuracy,computed_at",
      window_kind: "eq.cumulative",
      station_id: "is.null",
      order: "computed_at.desc",
    }),
    sb("operational_metric", {
      select: "horizon_min,accuracy,computed_at",
      window_kind: "eq.rolling_24h",
      station_id: "is.null",
      order: "computed_at.desc",
    }),
  ]);
  const latestByHorizon = (rows: any[]) => {
    const seen = new Set<number>();
    const out = new Map<number, number | null>();
    for (const r of rows) {
      if (seen.has(r.horizon_min)) continue;
      seen.add(r.horizon_min);
      out.set(r.horizon_min, r.accuracy);
    }
    return out;
  };
  const cumByH = latestByHorizon(cumulative);
  const rollByH = latestByHorizon(rolling);
  const horizons = [15, 30, 45, 60];
  return horizons.map((h) => {
    const c = cumByH.get(h) ?? null;
    const r = rollByH.get(h) ?? null;
    return { horizon_min: h, cumulative_accuracy: c, rolling_24h_accuracy: r, delta: c !== null && r !== null ? r - c : null };
  });
}

async function getRuns(): Promise<{ recentRuns: RunRow[]; trainingRuns: RunRow[] }> {
  const rows: RunRow[] = await sb("ejecucion_pipeline", {
    select: "run_id,run_at,status,decision_reentrenar,motivo_decision,model_id",
    order: "run_at.desc",
    limit: "60",
  });
  const trainingRuns = rows.filter((r) => r.decision_reentrenar).slice(0, 5);
  const recentRuns = rows.slice(0, 8);
  return { recentRuns, trainingRuns };
}

async function getLeaderboard(): Promise<DashboardData["leaderboard"]> {
  try {
    const [me, board] = await Promise.all([pulso("/v1/me"), pulso("/v1/leaderboard?window=cumulative")]);
    if (!board) return { self: null, top: [], error: "leaderboard no disponible" };
    const rows: LeaderboardRow[] = board.data ?? [];
    const myName: string | null = me?.display_name ?? null;
    const self = myName
      ? rows.find((r) => String(r.display_name).toLowerCase() === myName.toLowerCase()) ?? null
      : null;
    const top = [...rows].sort((a, b) => (a.rank ?? 999) - (b.rank ?? 999)).slice(0, 8);
    return { self, top, error: null };
  } catch (e: any) {
    return { self: null, top: [], error: String(e?.message ?? e) };
  }
}

export async function getDashboardData(): Promise<DashboardData> {
  const [collector, champions, stationMetrics, drift, runs, leaderboard, trend] = await Promise.all([
    getCollectorLag(),
    getChampions(),
    getStationMetrics(),
    getDrift(),
    getRuns(),
    getLeaderboard(),
    getAccuracyTrend(),
  ]);
  const stationsGeo = await getStationsGeo(stationMetrics);
  return {
    fetchedAt: new Date().toISOString(),
    collector,
    champions,
    stationMetrics,
    stationsGeo,
    trend,
    drift,
    recentRuns: runs.recentRuns,
    trainingRuns: runs.trainingRuns,
    leaderboard,
  };
}
