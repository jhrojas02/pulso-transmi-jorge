"""Sincronización incremental — capa 1 en producción (Fase "Collector").

Trae observaciones nuevas de /v1/stream/observations desde el cursor
guardado en Supabase (`sync_state`), las sube con upsert idempotente y
avanza el cursor SOLO después de escribir con éxito: si el proceso
muere a mitad de camino, la próxima corrida repite el último tramo en
vez de perderlo (nunca al revés — perder observaciones es peor que
reprocesar unas de más, y el upsert las vuelve inofensivas).

Nota conocida, descubierta en producción: el contexto (clima/eventos)
no se publica vía stream, solo el histórico fijo inicial (confirmado
contra el API: /v1/context nunca avanza más allá de 2026-09-09T04:45Z,
por más que pasen días reales — no es un retraso, el dataset
simplemente no publica clima nuevo). Como `observacion.observed_at`
tiene una foreign key hacia `contexto.observed_at`
(supabase/schema.sql), insertar observaciones de timestamps sin
contexto falla con 23503 — la base protegiendo integridad, no un bug.

Antes de insertar observaciones, este módulo completa el contexto
faltante con un ESTIMADO CLIMATOLÓGICO (promedio histórico real por
hora:minuto del día, ver features.climatological_context) — no con la
última lectura real repetida para siempre. Repetir un único valor real
indefinidamente es peor: ese valor deja de representar cualquier
condición real a medida que pasan los días, y el modelo puede terminar
interpretando esa constante arbitraria como si fuera información. La
climatología, en cambio, es un estimado honesto ("lo típico a esta
hora"), explícitamente marcado como tal.
"""

import os
from datetime import datetime, timezone

import pandas as pd
import requests

from src import supabase_client as sb
from src.features import climatological_context, estimate_context_row

API_BASE = os.environ.get("PULSO_API_BASE", "https://pulso-transmi.72-60-245-2.sslip.io")
SOURCE = "observations_stream"

# Último instante con contexto REAL confirmado contra el API (/v1/meta
# -> dataset.history_end). Cualquier fila de `contexto` después de esto
# es, por construcción, un estimado climatológico, nunca una lectura.
REAL_CONTEXT_CUTOFF = pd.Timestamp("2026-09-09T04:45:00Z")


def _ensure_context_for(observed_ats):
    """Garantiza que cada timestamp en `observed_ats` tenga fila en
    `contexto` (estimado climatológico) antes de que `observacion`
    intente referenciarlo.

    Compara por Timestamp, no por string crudo: el stream del API
    serializa como "...Z" y Supabase devuelve "...+00:00" para el
    mismo instante — comparar los strings directamente nunca coincide
    y hace parecer que siempre falta algo, aunque ya esté."""
    if not observed_ats:
        return
    existing = sb.select_all("contexto", select="observed_at", order="observed_at.asc")
    have_ts = {pd.Timestamp(row["observed_at"]) for row in existing}
    missing = sorted(t for t in set(observed_ats) if pd.Timestamp(t) not in have_ts)
    if not missing:
        return

    real_context = sb.select_all(
        "contexto",
        filters={"observed_at": f"lte.{REAL_CONTEXT_CUTOFF.isoformat()}"},
        order="observed_at.asc",
    )
    if not real_context:
        raise RuntimeError("No hay contexto real para calcular climatología; carga el histórico inicial primero (ingest.py).")
    climatology = climatological_context(pd.DataFrame(real_context))

    filled = [estimate_context_row(t, climatology) for t in missing]
    sb.write("contexto", filled, on_conflict="observed_at", merge=False)
    print(f"AVISO: contexto no publicado para {len(missing)} timestamps nuevos; estimado con climatología (promedio histórico real por hora:minuto)")


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
