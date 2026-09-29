"""Diagnóstico puntual de solo lectura contra Neon (DATABASE_URL) — se
corre desde .github/workflows/diag.yml. No escribe nada."""

from src import pg_client as pg

print("=== metrica_validacion del champion vigente para 03000 (control estable) ===")
champions = pg.select_all("champion", order="horizon_min.asc")
for champ in champions:
    m = pg.select_one(
        "metrica_validacion",
        filters={"model_id": f"eq.{champ['model_id']}", "station_id": "eq.03000"},
    )
    print(f"horizon={champ['horizon_min']} model_id={champ['model_id']}: {m}")

print("\n=== historial reciente de ejecucion_pipeline (decisiones de promote.py) ===")
runs = pg.select_all(
    "ejecucion_pipeline",
    select="run_id,run_at,motivo_decision",
    order="run_at.desc",
)
runs = [r for r in runs if r["motivo_decision"] and "promoted" in r["motivo_decision"]][:6]
for r in runs:
    print(f"\n--- {r['run_at']} ---")
    print(r["motivo_decision"][:1500])
