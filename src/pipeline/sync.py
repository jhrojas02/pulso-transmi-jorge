"""Sincronización incremental — capa 1 en producción (Fase "Collector").

Trae observaciones nuevas de /v1/stream/observations desde el cursor
guardado en Supabase (`sync_state`), las sube con upsert idempotente y
avanza el cursor SOLO después de escribir con éxito: si el proceso
muere a mitad de camino, la próxima corrida repite el último tramo en
vez de perderlo (nunca al revés — perder observaciones es peor que
reprocesar unas de más, y el upsert las vuelve inofensivas).

Nota conocida, descubierta en producción: el contexto (clima/eventos)
no se publica vía stream, solo el histórico fijo inicial. Como
`observacion.observed_at` tiene una foreign key hacia
`contexto.observed_at` (supabase/schema.sql), insertar observaciones
de timestamps sin contexto falla con 23503 — la base protegiendo
integridad, no un bug. Antes de insertar observaciones, este módulo
repite (forward-fill) el último contexto conocido hacia cualquier
timestamp nuevo que todavía no tenga fila en `contexto`. Queda
marcado — no es una lectura real del clima futuro.
"""

import os
from datetime import datetime, timezone

import requests

from src import supabase_client as sb

API_BASE = os.environ.get("PULSO_API_BASE", "https://pulso-transmi.72-60-245-2.sslip.io")
SOURCE = "observations_stream"


def _ensure_context_for(observed_ats):
    """Garantiza que cada timestamp en `observed_ats` tenga fila en
    `contexto` (forward-fill del último conocido) antes de que
    `observacion` intente referenciarlo."""
    if not observed_ats:
        return
    existing = sb.select_all("contexto", select="observed_at")
    have = {row["observed_at"] for row in existing}
    missing = sorted(t for t in set(observed_ats) if t not in have)
    if not missing:
        return

    last_context = sb.select_top("contexto", order="observed_at.desc", limit=1)
    if not last_context:
        raise RuntimeError("No hay contexto base para hacer forward-fill; carga el histórico inicial primero (ingest.py).")
    last = last_context[0]

    filled = []
    for t in missing:
        row = dict(last)
        row["observed_at"] = t
        filled.append(row)
    sb.write("contexto", filled, on_conflict="observed_at", merge=False)
    print(f"AVISO: contexto no publicado para {len(missing)} timestamps nuevos; forward-fill desde {last['observed_at']}")


def sync_observations_from_saved_cursor():
    state = sb.select_one("sync_state", filters={"source": f"eq.{SOURCE}"})
    cursor = state["cursor_value"] if state else None

    total = 0
    while True:
        params = {"limit": 5000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{API_BASE}/v1/stream/observations", params=params, timeout=30)
        r.raise_for_status()
        body = r.json()
        rows = body.get("data", [])
        if not rows:
            break

        _ensure_context_for({row["observed_at"] for row in rows})

        payload = [
            {"station_id": row["station_id"], "observed_at": row["observed_at"], "demand": row["demand"]}
            for row in rows
        ]
        sb.write("observacion", payload, on_conflict="station_id,observed_at", merge=False)
        total += len(payload)

        cursor = body.get("next_cursor")
        sb.write(
            "sync_state",
            [{"source": SOURCE, "cursor_value": cursor, "updated_at": datetime.now(timezone.utc).isoformat()}],
            on_conflict="source",
            merge=True,
        )
        if not cursor:
            break

    return total


if __name__ == "__main__":
    n = sync_observations_from_saved_cursor()
    print(f"sync_observations_from_saved_cursor: {n} filas nuevas")
