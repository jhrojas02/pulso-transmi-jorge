"""Diagnóstico puntual de solo lectura contra Neon (DATABASE_URL) — se
corre desde .github/workflows/diag.yml porque este sandbox de desarrollo
no tiene salida de red hacia Neon (bloqueada por política del entorno).
No escribe nada; solo imprime a los logs de Actions para poder leerlo
desde ahí."""

from src import pg_client as pg

print("=== champion vigente ===")
for row in pg.select_all("champion", order="horizon_min.asc"):
    print(row)

print("\n=== accuracy operacional rolling_24h, estación 05100 ===")
rows = pg.select_all(
    "operational_metric",
    select="horizon_min,accuracy,n_evaluable,computed_at",
    filters={"window_kind": "eq.rolling_24h", "station_id": "eq.05100"},
    order="computed_at.desc",
)
for row in rows[:8]:
    print(row)

print("\n=== metrica_validacion del último champion de cada horizonte, estación 05100 ===")
for champ in pg.select_all("champion", order="horizon_min.asc"):
    m = pg.select_all(
        "metrica_validacion",
        filters={"model_id": f"eq.{champ['model_id']}", "station_id": "eq.05100"},
    )
    print(f"horizon={champ['horizon_min']} model_id={champ['model_id']}: {m}")

print("\n=== última observación de 05100 en Neon (confirma que sigue creciendo) ===")
last_obs = pg.select_top("observacion", filters={"station_id": "eq.05100"}, order="observed_at.desc", limit=3)
for row in last_obs:
    print(row)
