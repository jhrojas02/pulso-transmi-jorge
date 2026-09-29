"""Diagnóstico puntual de solo lectura contra Neon (DATABASE_URL) — se
corre desde .github/workflows/diag.yml porque este sandbox de desarrollo
no tiene salida de red hacia Neon (bloqueada por política del entorno).
No escribe nada; solo imprime a los logs de Actions para poder leerlo
desde ahí."""

from src import pg_client as pg

for sid in ["02300", "05000", "07111"]:
    print(f"\n=== {sid}: predicción (h60) vs real, últimas 20 ===")
    preds = pg.select_all(
        "prediccion",
        select="target_timestamp,demanda_predicha",
        filters={"station_id": f"eq.{sid}", "horizonte": "eq.4"},
        order="target_timestamp.desc",
    )[:20]
    for p in preds:
        obs = pg.select_one(
            "observacion",
            select="demand",
            filters={"station_id": f"eq.{sid}", "observed_at": f"eq.{p['target_timestamp']}"},
        )
        real = obs["demand"] if obs else None
        print(f"  {p['target_timestamp']}  predicho={p['demanda_predicha']:.1f}  real={real}")
