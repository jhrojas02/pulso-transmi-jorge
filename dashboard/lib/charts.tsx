import type { AccuracyTrend, StationGeo } from "./data";

export function accCell(acc: number | null) {
  if (acc === null) return { bg: "var(--gridline)", fg: "var(--text-muted)", label: "—" };
  const t = Math.max(0, Math.min(1, (acc - 60) / 35));
  const steps = ["var(--seq-100)", "var(--seq-250)", "var(--seq-400)", "var(--seq-550)", "var(--seq-700)"];
  const idx = Math.min(steps.length - 1, Math.floor(t * steps.length));
  return { bg: steps[idx], fg: idx >= 2 ? "#ffffff" : "var(--text-primary)", label: acc.toFixed(1) };
}

const SEQ_STEPS_HEX = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];
function accHex(acc: number | null) {
  if (acc === null) return "#c3c2b7";
  const t = Math.max(0, Math.min(1, (acc - 60) / 35));
  const idx = Math.min(SEQ_STEPS_HEX.length - 1, Math.floor(t * SEQ_STEPS_HEX.length));
  return SEQ_STEPS_HEX[idx];
}

export function TrendChart({ trend }: { trend: AccuracyTrend }) {
  const { points, horizon_min } = trend;
  const W = 300;
  const H = 96;
  const padL = 8;
  const padR = 44;
  const padT = 14;
  const padB = 18;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  if (points.length < 2) {
    return (
      <div className="trend-chart">
        <div className="trend-title">+{horizon_min} min</div>
        <div className="trend-empty">Aún no hay suficiente historial</div>
      </div>
    );
  }

  const accs = points.map((p) => p.accuracy);
  const rawMin = Math.min(...accs);
  const rawMax = Math.max(...accs);
  const pad = Math.max(1, (rawMax - rawMin) * 0.2);
  const yMin = Math.max(0, rawMin - pad);
  const yMax = Math.min(100, rawMax + pad);
  const yRange = yMax - yMin || 1;

  const x = (i: number) => padL + (i / (points.length - 1)) * innerW;
  const y = (v: number) => padT + innerH - ((v - yMin) / yRange) * innerH;

  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.accuracy).toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${x(points.length - 1).toFixed(1)},${(padT + innerH).toFixed(1)} L${x(0).toFixed(1)},${(padT + innerH).toFixed(1)} Z`;

  const last = points[points.length - 1];
  const first = points[0];
  const trendDelta = last.accuracy - first.accuracy;

  return (
    <div className="trend-chart">
      <div className="trend-title">
        +{horizon_min} min
        <span className={`trend-delta ${trendDelta >= 0 ? "good" : "bad"}`}>
          {trendDelta >= 0 ? "+" : ""}
          {trendDelta.toFixed(1)}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="trend-svg" role="img" aria-label={`Tendencia de accuracy a +${horizon_min} minutos`}>
        <line x1={padL} y1={padT + innerH} x2={W - padR} y2={padT + innerH} stroke="var(--gridline)" strokeWidth={1} />
        <path d={areaPath} fill="var(--series-blue)" opacity={0.1} stroke="none" />
        <path d={linePath} fill="none" stroke="var(--series-blue)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        <circle cx={x(points.length - 1)} cy={y(last.accuracy)} r={4} fill="var(--series-blue)" stroke="var(--surface-1)" strokeWidth={2} />
        <text x={x(points.length - 1) + 8} y={y(last.accuracy) + 4} fontSize={12} fill="var(--text-primary)" fontWeight={600}>
          {last.accuracy.toFixed(1)}
        </text>
        <text x={padL} y={H - 2} fontSize={9} fill="var(--text-muted)">
          {new Date(first.t).toLocaleString("es-CO", { day: "2-digit", month: "short", hour: "2-digit" })}
        </text>
      </svg>
    </div>
  );
}

export function StationMap({ stations }: { stations: StationGeo[] }) {
  const W = 360;
  const H = 420;
  const pad = 36;

  const lats = stations.map((s) => s.latitude);
  const lons = stations.map((s) => s.longitude);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const latSpan = maxLat - minLat || 0.01;
  const lonSpan = maxLon - minLon || 0.01;

  const x = (lon: number) => pad + ((lon - minLon) / lonSpan) * (W - 2 * pad);
  const y = (lat: number) => pad + ((maxLat - lat) / latSpan) * (H - 2 * pad);

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="map-svg" role="img" aria-label="Mapa de estaciones con accuracy acumulada">
        {stations.map((s) => {
          const cx = x(s.longitude);
          const cy = y(s.latitude);
          const fill = accHex(s.accuracy);
          return (
            <g key={s.station_id}>
              <circle cx={cx} cy={cy} r={10} fill={fill} stroke="var(--surface-1)" strokeWidth={2} />
              <text x={cx} y={cy + 3} fontSize={8} fontWeight={700} textAnchor="middle" fill={fill === "#c3c2b7" ? "var(--text-primary)" : "#ffffff"}>
                {s.accuracy != null ? Math.round(s.accuracy) : "—"}
              </text>
              <text x={cx} y={cy + 22} fontSize={9} textAnchor="middle" fill="var(--text-secondary)">
                {s.station_name.length > 18 ? s.station_name.slice(0, 17) + "…" : s.station_name}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="map-legend">
        <span className="map-legend-label">Accuracy</span>
        {SEQ_STEPS_HEX.map((hex, i) => (
          <span key={i} className="map-legend-swatch" style={{ background: hex }} />
        ))}
        <span className="map-legend-label">bajo → alto</span>
      </div>
    </div>
  );
}

export function LeaderboardBars({
  rows,
  selfName,
}: {
  rows: { display_name: string; accuracy: number | null; rank: number | null }[];
  selfName: string | null;
}) {
  const max = Math.max(1, ...rows.map((r) => r.accuracy ?? 0));
  return (
    <div className="lb-bars">
      {rows.map((r, i) => {
        const isSelf = selfName != null && r.display_name.toLowerCase() === selfName.toLowerCase();
        const pct = ((r.accuracy ?? 0) / max) * 100;
        return (
          <div className="lb-row" key={i}>
            <span className="lb-rank">{r.rank ?? i + 1}</span>
            <span className={`lb-name ${isSelf ? "lb-name-self" : ""}`}>
              {r.display_name}
              {isSelf && <span className="lb-you"> (tú)</span>}
            </span>
            <span className="lb-track">
              <span className="lb-fill" style={{ width: `${pct}%`, background: isSelf ? "var(--status-good)" : "var(--series-blue)" }} />
            </span>
            <span className="lb-value">{r.accuracy != null ? r.accuracy.toFixed(1) : "—"}</span>
          </div>
        );
      })}
    </div>
  );
}
