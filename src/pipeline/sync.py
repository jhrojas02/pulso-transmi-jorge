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

from src import supabase_client as sb
from src.features import climatological_context, estimate_context_row
from src.http_retry import request_with_retry
from src.schema_guard import SchemaDriftError, normalize_observation, report_schema_drift, validate_observations

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

    # El cursor del stream no está avanzando (confirmado 2026-10-03:
    # cursor_value queda en None entre corridas) — el stream re-sirve la
    # misma ventana de ~12-13k filas cada 10min mientras el crecimiento
    # real de la tabla en ese lapso fue de un puñado de filas. El upsert
    # con ON CONFLICT DO NOTHING las ignora del lado de Postgres, pero
    # igual se manda el payload completo a la REST API cada vez — 144
    # veces/día, justo el tipo de carga repetitiva que agotó la cuota del
    # proyecto anterior. Se filtra del lado nuestro ANTES de escribir:
    # solo se manda a Supabase lo que es más nuevo que lo que ya tenemos.
    #
    # select_top, NUNCA select_one, para esto: select_one pagina de a 1
    # fila por request (ver su propio docstring en supabase_client.py) —
    # en una tabla de 65k+ filas SIN filtro eso son 65k+ requests
    # secuenciales. Confirmado en producción: colgó predict.yml varios
    # minutos antes de que se cancelara a mano. select_top usa `limit`
    # de PostgREST — una sola request.
    last_rows = sb.select_top("observacion", select="observed_at", order="observed_at.desc", limit=1)
    known_max = pd.Timestamp(last_rows[0]["observed_at"]) if last_rows else None

    total = 0
    while True:
        params = {"limit": 5000}
        if cursor:
            params["cursor"] = cursor
        r = request_with_retry("GET", f"{API_BASE}/v1/stream/observations", params=params, timeout=30)
        r.raise_for_status()
        body = r.json()
        # "data" ausente (campo renombrado) vs. [] real (nada nuevo todavía)
        # son casos MUY distintos: body.get("data", []) los trataba igual,
        # así que un cambio de formato acá se veía idéntico a "al día" —
        # el sync se habría congelado para siempre, cada 10min, sin ningún
        # error (mismo tipo de hueco que cycle.get("state"), 2026-10-03).
        if "data" not in body:
            e = SchemaDriftError(
                f"stream/observations: falta la clave 'data' en la respuesta — "
                f"claves recibidas: {sorted(body.keys())}"
            )
            report_schema_drift(e)
            raise e
        rows = body["data"]
        if not rows:
            break

        # Validar la forma cruda ANTES de tocar Supabase (ver
        # schema_guard.py) — el profesor avisó que viene un cambio de
        # formato fuerte; esto corre cada 10min sin supervisión, así
        # que debe fallar alto y claro en vez de escribir datos mal
        # interpretados silenciosamente.
        try:
            validate_observations(rows)
        except SchemaDriftError as e:
            report_schema_drift(e)
            raise

        new_rows = rows if known_max is None else [row for row in rows if pd.Timestamp(row["observed_at"]) > known_max]
        if len(new_rows) < len(rows):
            print(f"sync: {len(rows) - len(new_rows)} filas re-servidas por el stream (ya las teníamos) — omitidas antes de escribir")

        if new_rows:
            _ensure_context_for({row["observed_at"] for row in new_rows})
            # normalize_observation (ver schema_guard.py) traduce el formato
            # crudo del API (viejo "demand" plano, o el nuevo
            # measurement.value en string desde schema_version=2,
            # confirmado en producción 2026-10-04) a la forma plana que
            # espera Supabase — un solo lugar sabe que measurement existe.
            # Devuelve None para huecos de datos legítimos (quality=
            # "missing", confirmado en producción 2026-10-04) — se
            # excluyen del payload (observacion.demand es NOT NULL) pero
            # SÍ cuentan para avanzar known_max, si no el sync los pediría
            # de nuevo para siempre sin que nunca dejen de estar vacíos.
            payload = [n for row in new_rows if (n := normalize_observation(row)) is not None]
            if payload:
                sb.write("observacion", payload, on_conflict="station_id,observed_at", merge=False)
                total += len(payload)
            known_max = max(pd.Timestamp(row["observed_at"]) for row in new_rows)

        cursor = body.get("next_cursor")
        # "next_cursor" ausente es NORMAL en la última página real (menos
        # filas que el límite pedido) — pero una página LLENA (exactamente
        # el límite) sin cursor es rara: lo más probable es que haya más
        # datos y el campo se haya renombrado, no que justo haya terminado
        # ahí. Sin esto, "next_cursor" renombrado se ve igual que "ya
        # sincronizado" — el cursor nunca avanza y el sync se congela para
        # siempre repitiendo la misma página.
        if cursor is None and len(rows) >= params["limit"] and "next_cursor" not in body:
            e = SchemaDriftError(
                f"stream/observations: página llena ({len(rows)} filas) sin 'next_cursor' en la respuesta — "
                f"claves recibidas: {sorted(body.keys())}"
            )
            report_schema_drift(e)
            raise e
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
