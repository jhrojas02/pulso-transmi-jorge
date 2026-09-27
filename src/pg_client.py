"""Cliente de Postgres genérico — Plan B si Supabase queda restringido.

Mismo contrato (mismas funciones, misma firma) que supabase_client.py,
para que ningún script del pipeline tenga que cambiar más que la línea
de import (`from src import pg_client as sb` en vez de
`from src import supabase_client as sb`). Funciona contra CUALQUIER
Postgres accesible por connection string — Neon, Railway, RDS, un
Postgres propio — no ata el proyecto a otro proveedor con el mismo
tipo de política de fair use que causó el problema con Supabase.

Credenciales: DATABASE_URL se lee del entorno (nunca hardcodeada),
igual que SUPABASE_URL/SUPABASE_SERVICE_KEY en el cliente original.

Storage: Supabase Storage (para los .joblib de los modelos) no tiene
equivalente directo en un Postgres genérico, así que los modelos se
guardan como bytea en la tabla `model_blob` (ver
supabase/model_blob.sql) — a 7-9MB por modelo esto es perfectamente
razonable para Postgres, y evita depender de un segundo servicio
(Neon, por ejemplo, no ofrece storage de objetos).

Ver docs/plan-b-postgres.md para la guía completa de activación.
"""

import os

import psycopg2
import psycopg2.extras

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _require_config():
    if not DATABASE_URL:
        raise RuntimeError(
            "Falta DATABASE_URL en el entorno (connection string de Neon/Railway/"
            "Postgres). Nunca se hardcodea: expórtala antes de correr, o "
            "configúrala como secret de GitHub Actions."
        )


def _connect():
    _require_config()
    return psycopg2.connect(DATABASE_URL)


def _parse_filter(value):
    """'eq.x' -> ('=', 'x'); 'gte.x' -> ('>=', 'x'); 'lte.x' -> ('<=', 'x').
    Mismo subconjunto de operadores PostgREST que usa el pipeline hoy
    (ver grep de `filters=` en src/) — se agrega el operador que haga
    falta si en el futuro se usa alguno nuevo (ej. `in.`)."""
    op_map = {"eq": "=", "gte": ">=", "lte": "<=", "gt": ">", "lt": "<", "neq": "!="}
    op, _, val = value.partition(".")
    if op not in op_map:
        raise ValueError(f"Operador de filtro no soportado por pg_client: {value!r}")
    return op_map[op], val


def _build_where(filters):
    if not filters:
        return "", []
    clauses, params = [], []
    for col, value in filters.items():
        op, val = _parse_filter(value)
        clauses.append(f'"{col}" {op} %s')
        params.append(val)
    return " WHERE " + " AND ".join(clauses), params


def _build_order(order):
    if not order:
        return ""
    parts = []
    for piece in order.split(","):
        col, _, direction = piece.partition(".")
        direction = "DESC" if direction == "desc" else "ASC"
        parts.append(f'"{col}" {direction}')
    return " ORDER BY " + ", ".join(parts)


def select_all(table, select="*", filters=None, order=None, page_size=1000):
    """Ignora page_size (paginación por Range era un detalle del REST API
    de Supabase); una sola query trae todo lo que pide el filtro, que es
    lo que de verdad importa para el caller."""
    _require_config()
    cols = "*" if select == "*" else ", ".join(f'"{c.strip()}"' for c in select.split(","))
    where_sql, params = _build_where(filters)
    order_sql = _build_order(order)
    query = f"SELECT {cols} FROM {table}{where_sql}{order_sql}"
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]


def select_one(table, select="*", filters=None, order=None):
    rows = select_all(table, select=select, filters=filters, order=order, page_size=1)
    return rows[0] if rows else None


def select_top(table, select="*", filters=None, order=None, limit=1):
    _require_config()
    cols = "*" if select == "*" else ", ".join(f'"{c.strip()}"' for c in select.split(","))
    where_sql, params = _build_where(filters)
    order_sql = _build_order(order)
    query = f"SELECT {cols} FROM {table}{where_sql}{order_sql} LIMIT %s"
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params + [limit])
            return [dict(row) for row in cur.fetchall()]


def write(table, rows, on_conflict=None, merge=False, batch_size=2000):
    """Mismo contrato que supabase_client.write: merge=True sobrescribe
    en conflicto (para sync_state, champion), merge=False ignora
    duplicados (para cargas idempotentes tipo observacion/contexto).
    Asume filas homogéneas dentro de un mismo batch (mismas columnas),
    igual que ya asume el resto del pipeline al construirlas."""
    _require_config()
    if not rows:
        return 0
    cols = list(rows[0].keys())
    col_sql = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join(["%s"] * len(cols))

    conflict_sql = ""
    if on_conflict:
        conflict_cols = ", ".join(f'"{c.strip()}"' for c in on_conflict.split(","))
        if merge:
            update_sql = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols if c not in on_conflict.split(","))
            conflict_sql = f" ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_sql}" if update_sql else f" ON CONFLICT ({conflict_cols}) DO NOTHING"
        else:
            conflict_sql = f" ON CONFLICT ({conflict_cols}) DO NOTHING"

    query = f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}){conflict_sql}"
    with _connect() as conn:
        with conn.cursor() as cur:
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                values = [[row.get(c) for c in cols] for row in batch]
                psycopg2.extras.execute_batch(cur, query, values)
        conn.commit()
    return len(rows)


def storage_upload(bucket, path, data: bytes, content_type="application/octet-stream"):
    """Guarda `data` como bytea en model_blob, bajo la llave `<bucket>/<path>`
    (mismo namespacing que supabase_client.storage_upload, para que el
    resto del pipeline no tenga que saber que ya no hay buckets reales)."""
    _require_config()
    key = f"{bucket}/{path}"
    query = (
        'INSERT INTO model_blob ("path", "data", "updated_at") VALUES (%s, %s, now()) '
        'ON CONFLICT ("path") DO UPDATE SET "data" = EXCLUDED."data", "updated_at" = now()'
    )
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [key, psycopg2.Binary(data)])
        conn.commit()
    return {"path": key}


def storage_download(bucket, path):
    _require_config()
    key = f"{bucket}/{path}"
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT "data" FROM model_blob WHERE "path" = %s', [key])
            row = cur.fetchone()
            if row is None:
                raise FileNotFoundError(f"model_blob: no existe {key!r}")
            return bytes(row[0])
