import { getDashboardData, type StationMetric } from "@/lib/data";
import { accCell, TrendChart, StationMap, LeaderboardBars } from "@/lib/charts";

export const dynamic = "force-dynamic";
export const revalidate = 0;

function fmtMinutesAgo(min: number | null) {
  if (min === null) return "—";
  if (min < 1) return "hace instantes";
  if (min < 60) return `hace ${min} min`;
  const h = Math.floor(min / 60);
  return `hace ${h}h ${min % 60}min`;
}

function collectorStatus(min: number | null) {
  if (min === null) return { dot: "dot-critical", text: "sin datos" };
  if (min <= 20) return { dot: "dot-good", text: "al día" };
  if (min <= 60) return { dot: "dot-warning", text: "con retraso" };
  return { dot: "dot-critical", text: "detenido" };
}

function fmtDate(iso: string | null | undefined) {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("es-CO", { dateStyle: "medium", timeStyle: "short" });
}

export default async function Page() {
  const data = await getDashboardData();
  const collector = collectorStatus(data.collector.lagMinutes);

  const stationIds = Array.from(new Set(data.stationMetrics.map((m) => m.station_id))).sort();
  const horizons = [15, 30, 45, 60];
  const byKey = new Map(data.stationMetrics.map((m) => [`${m.station_id}::${m.horizon_min}`, m]));
  const stationNameById = new Map(data.stationMetrics.map((m) => [m.station_id, m.station_name]));
  const selfName = data.leaderboard.self ? String(data.leaderboard.self.display_name) : null;

  return (
    <main className="page">
      <div className="header">
        <div>
          <h1>Pulso TransMi</h1>
          <div className="subtitle">Dashboard operativo — observabilidad del pipeline en vivo</div>
        </div>
        <span className="badge">
          <span className={`dot ${collector.dot}`} />
          collector {collector.text} · {fmtMinutesAgo(data.collector.lagMinutes)}
        </span>
      </div>

      <section className="section">
        <h2>Leaderboard</h2>
        <div className="tiles">
          <div className="tile">
            <span className="label">Accuracy</span>
            <span className="value">
              {data.leaderboard.self?.accuracy != null ? data.leaderboard.self.accuracy.toFixed(2) : "—"}
            </span>
            <span className="hint">acumulado, cohorte</span>
          </div>
          <div className="tile">
            <span className="label">Cobertura</span>
            <span className="value">
              {data.leaderboard.self?.coverage != null ? `${(Number(data.leaderboard.self.coverage) * 100).toFixed(0)}%` : "—"}
            </span>
            <span className="hint">predicciones con realidad ya evaluada</span>
          </div>
          <div className="tile">
            <span className="label">Posición</span>
            <span className="value">{data.leaderboard.self?.rank ?? "—"}</span>
            <span className="hint">en el leaderboard oficial</span>
          </div>
        </div>
        {data.leaderboard.error && (
          <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
            Leaderboard no disponible ahora mismo ({data.leaderboard.error}).
          </p>
        )}
        {!data.leaderboard.self && !data.leaderboard.error && data.leaderboard.top.length > 0 && (
          <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
            No se pudo emparejar tu fila por nombre — mostrando el top de la cohorte.
          </p>
        )}
        {data.leaderboard.top.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <LeaderboardBars rows={data.leaderboard.top} selfName={selfName} />
          </div>
        )}
      </section>

      <section className="section">
        <h2>Tendencia de accuracy (acumulada, última semana)</h2>
        <div className="trend-grid">
          {data.trend.map((t) => (
            <TrendChart key={t.horizon_min} trend={t} />
          ))}
        </div>
      </section>

      <section className="section">
        <h2>Mapa de estaciones</h2>
        <div className="card map-card">
          <StationMap stations={data.stationsGeo} />
        </div>
      </section>

      <section className="section">
        <h2>Champion vigente</h2>
        <table>
          <thead>
            <tr>
              <th>Horizonte</th>
              <th>Versión</th>
              <th>Entrenado</th>
              <th>Corte de datos</th>
              <th>Promovido</th>
            </tr>
          </thead>
          <tbody>
            {data.champions.map((c) => (
              <tr key={c.horizon_min}>
                <td>+{c.horizon_min} min</td>
                <td className="mono">{c.version ?? "—"}</td>
                <td>{fmtDate(c.trained_at)}</td>
                <td>{c.cutoff_train_fin ?? "—"}</td>
                <td>{fmtDate(c.promoted_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="section">
        <h2>Drift — accuracy acumulada vs. últimas 24h</h2>
        <div className="tiles">
          {data.drift.map((d) => {
            const warn = d.delta !== null && d.delta < -3;
            return (
              <div className="tile" key={d.horizon_min}>
                <span className="label">+{d.horizon_min} min</span>
                <span className="value">
                  {d.rolling_24h_accuracy != null ? d.rolling_24h_accuracy.toFixed(1) : "—"}
                </span>
                <span className="hint">
                  acumulado {d.cumulative_accuracy != null ? d.cumulative_accuracy.toFixed(1) : "—"} ·{" "}
                  <span style={{ color: warn ? "var(--status-critical)" : "var(--status-good)" }}>
                    {d.delta != null ? `${d.delta >= 0 ? "+" : ""}${d.delta.toFixed(1)}` : "—"}
                  </span>
                </span>
              </div>
            );
          })}
        </div>
        <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
          Si la caída promedio supera 3 puntos, el pipeline dispara un reentrenamiento antes de tiempo automáticamente
          (ver <code className="mono">monitor.py</code>) — no hace falta ninguna acción manual.
        </p>
      </section>

      <section className="section">
        <h2>Error por estación y horizonte (accuracy acumulada)</h2>
        <table>
          <thead>
            <tr>
              <th>Estación</th>
              {horizons.map((h) => (
                <th key={h}>+{h} min</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {stationIds.map((sid) => (
              <tr key={sid}>
                <td>
                  {sid} <span className="muted">{stationNameById.get(sid) ?? ""}</span>
                </td>
                {horizons.map((h) => {
                  const m = byKey.get(`${sid}::${h}`) as StationMetric | undefined;
                  const cell = accCell(m?.accuracy ?? null);
                  return (
                    <td key={h}>
                      <span className="cell-accuracy" style={{ background: cell.bg, color: cell.fg }}>
                        {cell.label}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="section">
        <h2>Decisiones de reentrenamiento recientes</h2>
        <div className="card">
          {data.trainingRuns.length === 0 && <p className="muted">Sin corridas de entrenamiento registradas aún.</p>}
          {data.trainingRuns.map((r) => (
            <div className="run-row" key={r.run_id}>
              <div>
                <div>{r.model_id ? `Promovió ${r.model_id}` : "Sin promoción"}</div>
                <div className="meta">{r.motivo_decision?.slice(0, 220)}</div>
              </div>
              <div className="meta" style={{ whiteSpace: "nowrap" }}>{fmtDate(r.run_at)}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="section">
        <h2>Última actividad del pipeline</h2>
        <div className="card">
          {data.recentRuns.map((r) => (
            <div className="run-row" key={r.run_id}>
              <div>
                <span className={`badge`} style={{ marginRight: 8 }}>
                  <span className={`dot ${r.status === "ok" ? "dot-good" : "dot-critical"}`} />
                  {r.status}
                </span>
                {r.decision_reentrenar ? "entrenamiento" : "inferencia"}
              </div>
              <div className="meta">{fmtDate(r.run_at)}</div>
            </div>
          ))}
        </div>
      </section>

      <footer>
        Actualizado {fmtDate(data.fetchedAt)} · datos de solo lectura desde Supabase y la API oficial de Pulso
        TransMi
      </footer>
    </main>
  );
}
