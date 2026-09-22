"""Cliente REST mínimo de Supabase para los workflows de GitHub Actions.

Los módulos interactivos de esta sesión usan las tools MCP de Supabase,
que no existen fuera de aquí. Todo lo que corre en CI (sync.py,
submit_current_cycle.py, promote.py, monitor.py) pasa por este cliente,
que solo habla HTTP con `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` leídos
del entorno (nunca hardcodeados, van como GitHub Actions secrets).
"""

import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

# select_all pagina tablas grandes en decenas de requests seguidas; una
# sola desconexión transitoria (visto en producción: "Connection reset
# by peer" a mitad de una página) no debería tirar la corrida completa.
# Reintenta con backoff tanto errores de conexión como 5xx/429.
_session = requests.Session()
_retry = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
)
_session.mount("https://", HTTPAdapter(max_retries=_retry))
_session.mount("http://", HTTPAdapter(max_retries=_retry))


def _require_config():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "Faltan SUPABASE_URL y/o SUPABASE_SERVICE_KEY en el entorno "
            "(deben venir como secrets de GitHub Actions)."
        )


def _headers(extra=None):
    h = {"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
    if extra:
        h.update(extra)
    return h


def select_all(table, select="*", filters=None, order=None, page_size=1000):
    """Trae todas las filas de `table`, paginando con Range (PostgREST
    no devuelve más de `page_size` filas por request salvo que el
    proyecto lo configure distinto)."""
    _require_config()
    rows = []
    offset = 0
    params = {"select": select}
    if filters:
        params.update(filters)
    if order:
        params["order"] = order
    while True:
        headers = _headers({"Range-Unit": "items", "Range": f"{offset}-{offset + page_size - 1}"})
        r = _session.get(f"{SUPABASE_URL}/rest/v1/{table}", params=params, headers=headers, timeout=60)
        r.raise_for_status()
        batch = r.json()
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def select_one(table, select="*", filters=None, order=None):
    """Para filtros que ya acotan a 0-1 filas (ej. una PK). Para 'la
    fila más reciente' de una tabla grande sin filtro selectivo, usar
    select_top: pedir 1 fila por página vía Range pagina TODA la tabla
    una por una, porque nunca deja de devolver exactamente 1 fila."""
    rows = select_all(table, select=select, filters=filters, order=order, page_size=1)
    return rows[0] if rows else None


def select_top(table, select="*", filters=None, order=None, limit=1):
    """Trae como mucho `limit` filas en una sola request, vía el
    parámetro `limit` de PostgREST (no Range/paginación) — para "dame
    la última fila" de una tabla grande, ordenada."""
    _require_config()
    params = {"select": select, "limit": limit}
    if filters:
        params.update(filters)
    if order:
        params["order"] = order
    r = _session.get(f"{SUPABASE_URL}/rest/v1/{table}", params=params, headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()


def write(table, rows, on_conflict=None, merge=False, batch_size=2000):
    """Inserta/actualiza filas. `merge=True` sobrescribe en conflicto
    (para sync_state, champion); `merge=False` (default) ignora
    duplicados (para cargas idempotentes tipo observacion/contexto)."""
    _require_config()
    if not rows:
        return 0
    resolution = "merge-duplicates" if merge else "ignore-duplicates"
    headers = _headers({"Content-Type": "application/json", "Prefer": f"resolution={resolution},return=minimal"})
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    params = {"on_conflict": on_conflict} if on_conflict else {}
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        r = _session.post(url, params=params, headers=headers, json=batch, timeout=60)
        r.raise_for_status()
    return len(rows)


def storage_upload(bucket, path, data: bytes, content_type="application/octet-stream"):
    _require_config()
    headers = _headers({"Content-Type": content_type, "x-upsert": "true"})
    r = _session.post(f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}", headers=headers, data=data, timeout=60)
    r.raise_for_status()
    return r.json()


def storage_download(bucket, path):
    _require_config()
    r = _session.get(f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}", headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.content
