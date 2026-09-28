"""Plan B: copia todo el histórico de Supabase a un Postgres genérico
(Neon, Railway, etc.) — ver docs/plan-b-postgres.md para la guía
completa de activación paso a paso.

Requiere en el entorno, AL MISMO TIEMPO:
  - SUPABASE_URL / SUPABASE_SERVICE_KEY  (origen, de donde se lee)
  - DATABASE_URL                          (destino, a donde se escribe)

No borra nada en Supabase — es una copia de solo lectura del lado
origen. Se puede correr más de una vez sin duplicar datos (todas las
tablas usan upsert por su llave primaria).

Uso:
    export SUPABASE_URL=...
    export SUPABASE_SERVICE_KEY=...
    export DATABASE_URL=postgresql://...neon.tech/...
    python -m scripts.migrate_to_postgres
"""

import sys

from src import pg_client, supabase_client as sb

# Orden que respeta las foreign keys del schema (estacion/contexto primero,
# lo que depende de ellas después).
TABLES_IN_ORDER = [
    ("estacion", "station_id"),
    ("contexto", "observed_at"),
    ("observacion", "station_id,observed_at"),
    ("modelo", "model_id"),
    ("metrica_validacion", "metric_id"),
    ("ejecucion_pipeline", "run_id"),
    ("prediccion", "prediction_id"),
    ("sync_state", "source"),
    ("submission_receipt", "cycle_id,model_version"),
    ("champion", "horizon_min"),
    ("operational_metric", "metric_id"),
]


def migrate_tables():
    for table, on_conflict in TABLES_IN_ORDER:
        rows = sb.select_all_rest(table)
        if not rows:
            print(f"{table}: sin filas, se salta")
            continue
        n = pg_client.write(table, rows, on_conflict=on_conflict, merge=True)
        print(f"{table}: {n} filas copiadas")


def migrate_models():
    """Copia cada gbm.joblib del bucket `models` de Supabase Storage
    hacia model_blob en el Postgres destino."""
    modelos = sb.select_all_rest("modelo", select="model_id,artifact_uri")
    for row in modelos:
        artifact_uri = row["artifact_uri"]
        if not artifact_uri.startswith("supabase-storage://models/"):
            print(f"  aviso: artifact_uri inesperado, se salta: {artifact_uri}")
            continue
        storage_path = artifact_uri.removeprefix("supabase-storage://models/")
        blob = sb.storage_download_rest("models", storage_path)
        pg_client.storage_upload("models", storage_path, blob)
        print(f"modelo {row['model_id']}: {len(blob)} bytes copiados")


def main():
    print("=== Copiando tablas ===")
    migrate_tables()
    print("\n=== Copiando modelos (Storage -> model_blob) ===")
    migrate_models()
    print("\nListo. Verifica los conteos arriba contra Supabase antes de cortar el tráfico.")


if __name__ == "__main__":
    main()
