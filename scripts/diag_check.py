"""Diagnóstico puntual de solo lectura contra Neon (DATABASE_URL) — se
corre desde .github/workflows/diag.yml porque este sandbox de desarrollo
no tiene salida de red hacia Neon (bloqueada por política del entorno).
No escribe nada; solo imprime a los logs de Actions para poder leerlo
desde ahí."""

from src import pg_client as pg

print("=== submission_receipt: conteo y status ===")
receipts = pg.select_all("submission_receipt", select="cycle_id,status,data_cutoff,received_at")
print(f"total recibos: {len(receipts)}")
by_status = {}
for r in receipts:
    by_status[r["status"]] = by_status.get(r["status"], 0) + 1
print("por status:", by_status)

receipts_sorted = sorted(receipts, key=lambda r: r["data_cutoff"])
print("\nprimeros 5:")
for r in receipts_sorted[:5]:
    print(" ", r)
print("últimos 10:")
for r in receipts_sorted[-10:]:
    print(" ", r)

print("\n=== huecos entre data_cutoff consecutivos (>35 min sugiere ciclo saltado) ===")
import datetime
prev = None
gaps = []
for r in receipts_sorted:
    dc = datetime.datetime.fromisoformat(r["data_cutoff"].replace("Z", "+00:00")) if isinstance(r["data_cutoff"], str) else r["data_cutoff"]
    if prev is not None:
        delta_min = (dc - prev).total_seconds() / 60
        if delta_min > 35:
            gaps.append((prev.isoformat(), dc.isoformat(), delta_min))
    prev = dc
print(f"huecos encontrados: {len(gaps)}")
for g in gaps[:20]:
    print(" ", g)
