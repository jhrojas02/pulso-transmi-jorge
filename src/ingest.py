"""Ingesta de datos del API Pulso TransMi hacia Supabase.

Entra: estaciones, contexto (clima/eventos) y observaciones (demanda) tal
cual los expone el API del starter kit — datos ya publicados/observados,
nunca predicciones ni ground truth futuro.
Sale: filas insertadas (upsert idempotente) en las tablas `estacion`,
`contexto` y `observacion` de Supabase — la "capa 1" del esquema en
supabase/schema.sql. No calcula features ni entrena nada.

Riesgo de fuga de datos: ninguno en este paso — es una copia directa del
API a la base, sin mirar el futuro ni usar información que el API no haya
publicado todavía. La fuga, si existe, se introduce más adelante en
features.py (ej. una rolling window que mire hacia adelante), no aquí.

Credenciales: SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY se leen del entorno
(nunca hardcodeadas). En GitHub Actions van como secrets del repo. Con la
service_role key, esta ingesta salta Row Level Security a propósito: es el
único componente autorizado a escribir en estas tablas.
"""

import argparse
import os
import sys

import requests

API_BASE = os.environ.get("PULSO_API_BASE", "https://pulso-transmi.72-60-245-2.sslip.io")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

PAGE_LIMIT = 5000


def _require_supabase_config():
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        sys.exit(
            "Faltan SUPABASE_URL y/o SUPABASE_SERVICE_ROLE_KEY en el entorno. "
            "Nunca se hardcodean: expórtalas antes de correr este script, o "
            "configúralas como secrets en GitHub Actions."
        )


def _api_get(path, **params):
    r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _paginate(path, since=None):
    cursor = None
    params = {"limit": PAGE_LIMIT}
    if since:
        params["start"] = since
    while True:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        body = _api_get(path, **page_params)
        yield from body["data"]
        cursor = body.get("next_cursor")
        if not cursor:
            break


def _supabase_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=ignore-duplicates,return=minimal",
    }


def _supabase_upsert(table, rows, on_conflict, batch_size=2000):
    if not rows:
        return 0
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        r = requests.post(
            url,
            params={"on_conflict": on_conflict},
            headers=_supabase_headers(),
            json=batch,
            timeout=60,
        )
        r.raise_for_status()
    return len(rows)


def _supabase_max(table, column):
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        params={"select": column, "order": f"{column}.desc", "limit": 1},
        headers=_supabase_headers(),
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data[0][column] if data else None


def ingest_stations():
    stations = _api_get("/v1/stations")["data"]
    rows = [
        {
            "station_id": s["station_id"],
            "station_name": s["station_name"],
            "corridor": s["corridor"],
            "latitude": s["latitude"],
            "longitude": s["longitude"],
        }
        for s in stations
    ]
    n = _supabase_upsert("estacion", rows, on_conflict="station_id")
    print(f"estacion: {n} filas")


def ingest_context(since=None):
    rows = list(_paginate("/v1/context", since=since))
    n = _supabase_upsert("contexto", rows, on_conflict="observed_at")
    print(f"contexto: {n} filas (since={since})")


def ingest_observations(since=None):
    rows = list(_paginate("/v1/observations", since=since))
    rows = [
        {"station_id": o["station_id"], "observed_at": o["observed_at"], "demand": o["demand"]}
        for o in rows
    ]
    n = _supabase_upsert("observacion", rows, on_conflict="station_id,observed_at")
    print(f"observacion: {n} filas (since={since})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ignora el cursor incremental y vuelve a traer todo el histórico disponible del API.",
    )
    args = parser.parse_args()

    _require_supabase_config()

    ingest_stations()

    since_context = None if args.full else _supabase_max("contexto", "observed_at")
    ingest_context(since=since_context)

    since_obs = None if args.full else _supabase_max("observacion", "observed_at")
    ingest_observations(since=since_obs)


if __name__ == "__main__":
    main()
