"""Diagnóstico puntual de solo lectura contra la API de Pulso — se corre
desde .github/workflows/diag.yml. No escribe nada."""

import os

import requests

API_BASE = "https://pulso-transmi.72-60-245-2.sslip.io"
API_KEY = os.environ.get("PULSO_API_KEY", "")
headers = {"Authorization": f"Bearer {API_KEY}"}
ME = "Jorge Horacio Rojas Criollo"

for window in ["cumulative", "rolling_24h"]:
    r = requests.get(f"{API_BASE}/v1/leaderboard", params={"window": window}, headers=headers, timeout=15)
    body = r.json()
    print(f"\n=== leaderboard window={window} (status {r.status_code}) ===")
    print("meta:", {k: v for k, v in body.items() if k != "data"})
    rows = body.get("data", [])
    print(f"total participantes: {len(rows)}")
    me_row = next((row for row in rows if row["display_name"] == ME), None)
    top3 = rows[:3]
    print("\nTop 3:")
    for row in top3:
        print(f"  #{row['rank']:2d}  {row['display_name']:35s}  accuracy={row['accuracy']:.2f}  wape={row['raw_wape']:.4f}  coverage={row['coverage']:.2%}")
    print("\nYo:")
    if me_row:
        print(f"  #{me_row['rank']:2d}  {me_row['display_name']:35s}  accuracy={me_row['accuracy']:.2f}  wape={me_row['raw_wape']:.4f}  coverage={me_row['coverage']:.2%}")
    else:
        print("  no aparezco en esta ventana")
    print("\nInmediatamente arriba y abajo de mí:")
    if me_row:
        idx = rows.index(me_row)
        for row in rows[max(0, idx - 2): idx + 3]:
            marker = " <-- YO" if row is me_row else ""
            print(f"  #{row['rank']:2d}  {row['display_name']:35s}  accuracy={row['accuracy']:.2f}{marker}")
