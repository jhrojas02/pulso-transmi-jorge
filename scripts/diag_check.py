"""Diagnóstico puntual de solo lectura contra Neon (DATABASE_URL) — se
corre desde .github/workflows/diag.yml. No escribe nada."""

from src import pg_client as pg

print("=== champion vigente ===")
champions = pg.select_all("champion", order="horizon_min.asc")
for row in champions:
    print(row)

print("\n=== accuracy operacional rolling_24h por estación (más reciente, TODAS) ===")
all_metrics = pg.select_all(
    "operational_metric",
    select="horizon_min,station_id,accuracy,n_evaluable,computed_at",
    filters={"window_kind": "eq.rolling_24h"},
    order="computed_at.desc",
)
if all_metrics:
    latest_ts = all_metrics[0]["computed_at"]
    latest = [r for r in all_metrics if r["computed_at"] == latest_ts and r["station_id"] is not None]
    latest.sort(key=lambda r: r["accuracy"])
    print(f"computed_at={latest_ts}")
    for r in latest:
        print(f"  {r['station_id']}  h{r['horizon_min']:>2}  accuracy={r['accuracy']:6.2f}  n={r['n_evaluable']}")

print("\n=== promedio por estación ===")
by_station = {}
for r in latest:
    by_station.setdefault(r["station_id"], []).append(r["accuracy"])
for sid, accs in sorted(by_station.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
    print(f"  {sid}: promedio={sum(accs)/len(accs):6.2f}")
