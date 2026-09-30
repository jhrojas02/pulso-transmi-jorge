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

import time

print("\n=== metadata de cada ventana del leaderboard ===")
boards = {}
raw_by_window = {}
for window in ["cumulative", "rolling_24h"]:
    for attempt in range(5):
        r = requests.get(f"{API_BASE}/v1/leaderboard", params={"window": window}, headers=headers, timeout=15)
        if r.status_code == 200:
            break
        print(f"  reintento {attempt} para {window}: {r.status_code}")
        time.sleep(3)
    raw = r.json()
    raw_by_window[window] = raw
    meta = {k: v for k, v in raw.items() if k != "data"}
    print(f"{window}: {jsonlib.dumps(meta, indent=2)}")
    boards[window] = {row["display_name"]: row["accuracy"] for row in raw["data"]}
    time.sleep(2)

print("\n=== nuestra fila completa en cada ventana ===")
for window, raw in raw_by_window.items():
    for row in raw["data"]:
        if row["display_name"] == "Jorge Horacio Rojas Criollo":
            print(f"{window}: {jsonlib.dumps(row, indent=2)}")

print("\n=== fila completa de los top 3 y de quien más subió, ambas ventanas ===")
cum_by_name = {r["display_name"]: r for r in raw_by_window["cumulative"]["data"]}
roll_by_name = {r["display_name"]: r for r in raw_by_window["rolling_24h"]["data"]}
gaps = sorted(
    ((name, roll_by_name[name]["accuracy"] - cum_by_name[name]["accuracy"]) for name in cum_by_name if name in roll_by_name),
    key=lambda x: -x[1],
)
for name, gap in gaps[:5]:
    print(f"-- {name} (gap={gap:+.2f}) --")
    print("cumulative:", jsonlib.dumps(cum_by_name[name], indent=2))
    print("rolling_24h:", jsonlib.dumps(roll_by_name[name], indent=2))

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
