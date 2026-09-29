"""Diagnóstico puntual de solo lectura contra la API oficial — se corre
desde .github/workflows/diag.yml. No escribe nada."""

import os

import requests

API_BASE = "https://pulso-transmi.72-60-245-2.sslip.io"
API_KEY = os.environ.get("PULSO_API_KEY", "")
headers = {"Authorization": f"Bearer {API_KEY}"}

print("=== /v1/me ===")
r = requests.get(f"{API_BASE}/v1/me", headers=headers, timeout=15)
print(r.status_code, r.text[:2000])

for window in ["cumulative", "rolling_24h"]:
    print(f"\n=== /v1/leaderboard?window={window} ===")
    r = requests.get(f"{API_BASE}/v1/leaderboard", params={"window": window}, headers=headers, timeout=15)
    print(r.status_code, r.text[:4000])
