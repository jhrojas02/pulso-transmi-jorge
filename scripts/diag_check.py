"""Diagnóstico puntual de solo lectura contra Neon (DATABASE_URL) — se
corre desde .github/workflows/diag.yml porque este sandbox de desarrollo
no tiene salida de red hacia Neon (bloqueada por política del entorno).
No escribe nada; solo imprime a los logs de Actions para poder leerlo
desde ahí."""

import datetime

from src import pg_client as pg

receipts = pg.select_all(
    "submission_receipt",
    select="cycle_id,status,data_cutoff,received_at",
    filters={"cycle_id": "neq.cyc_practice_20260918"},
)
receipts_sorted = sorted(receipts, key=lambda r: r["data_cutoff"])
print(f"total recibos oficiales: {len(receipts_sorted)}")
print("primer data_cutoff:", receipts_sorted[0]["data_cutoff"])
print("último data_cutoff:", receipts_sorted[-1]["data_cutoff"])

print("\n=== TODOS los huecos entre data_cutoff consecutivos (>70 min = probable ciclo saltado) ===")
prev = None
gaps = []
for r in receipts_sorted:
    dc = datetime.datetime.fromisoformat(r["data_cutoff"].replace("Z", "+00:00")) if isinstance(r["data_cutoff"], str) else r["data_cutoff"]
    if prev is not None:
        delta_min = (dc - prev).total_seconds() / 60
        if delta_min > 70:
            gaps.append((prev.isoformat(), dc.isoformat(), delta_min))
    prev = dc
print(f"huecos >70min encontrados: {len(gaps)}")
for g in gaps:
    print(f"  {g[0]}  ->  {g[1]}   ({g[2]:.0f} min = {g[2]/60:.1f} h)")

print("\n=== estado del reloj ahora ===")
import requests
clock = requests.get("https://pulso-transmi.72-60-245-2.sslip.io/v1/clock", timeout=15).json()
print(clock)
