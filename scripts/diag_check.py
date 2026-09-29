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

import json as jsonlib

boards = {}
for window in ["cumulative", "rolling_24h"]:
    r = requests.get(f"{API_BASE}/v1/leaderboard", params={"window": window}, headers=headers, timeout=15)
    boards[window] = {row["display_name"]: row["accuracy"] for row in r.json()["data"]}

print("\n=== comparacion cumulative vs rolling_24h (gap = rolling - cumulative) ===")
names = set(boards["cumulative"]) | set(boards["rolling_24h"])
rows = []
for name in names:
    cum = boards["cumulative"].get(name)
    roll = boards["rolling_24h"].get(name)
    gap = (roll - cum) if (cum is not None and roll is not None) else None
    rows.append((name, cum, roll, gap))
rows.sort(key=lambda r: (r[3] is None, -(r[3] or 0)))
for name, cum, roll, gap in rows:
    cum_s = f"{cum:6.2f}" if cum is not None else "  n/a "
    roll_s = f"{roll:6.2f}" if roll is not None else "  n/a "
    gap_s = f"{gap:+6.2f}" if gap is not None else "   n/a"
    print(f"{name:35s} cumulative={cum_s}  rolling_24h={roll_s}  gap={gap_s}")
