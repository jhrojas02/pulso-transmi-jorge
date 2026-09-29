"""Diagnóstico puntual de solo lectura contra Neon + API de Pulso — se
corre desde .github/workflows/diag.yml porque este sandbox de desarrollo
no tiene salida de red hacia Neon (bloqueada por política del entorno).
No escribe nada; solo imprime a los logs de Actions para poder leerlo
desde ahí."""

import os

import requests

API_BASE = "https://pulso-transmi.72-60-245-2.sslip.io"
API_KEY = os.environ.get("PULSO_API_KEY", "")
headers = {"Authorization": f"Bearer {API_KEY}"}

print("=== /v1/me ===")
r = requests.get(f"{API_BASE}/v1/me", headers=headers, timeout=15)
print(r.status_code, r.text[:2000])

print("\n=== /v1/leaderboard?window=cumulative ===")
r = requests.get(f"{API_BASE}/v1/leaderboard", params={"window": "cumulative"}, headers=headers, timeout=15)
print(r.status_code, r.text[:4000])

print("\n=== /v1/leaderboard?window=rolling_24h ===")
r = requests.get(f"{API_BASE}/v1/leaderboard", params={"window": "rolling_24h"}, headers=headers, timeout=15)
print(r.status_code, r.text[:4000])

print("\n=== /v1/portal/accuracy-chart ===")
r = requests.get(f"{API_BASE}/v1/portal/accuracy-chart", headers=headers, timeout=15)
print(r.status_code, r.text[:4000])

print("\n=== /v1/portal/dashboard ===")
r = requests.get(f"{API_BASE}/v1/portal/dashboard", headers=headers, timeout=15)
print(r.status_code, r.text[:3000])
