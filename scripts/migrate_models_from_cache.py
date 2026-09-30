"""Migración puntual, de una sola vez: copia los 4 .joblib de los
champions actuales desde la caché local de GitHub Actions (.model_cache/,
restaurada por actions/cache con la misma key que usa predict.yml) hacia
la tabla model_blob en la MISMA base de Supabase, vía conexión directa a
Postgres (DATABASE_URL) — sin pasar por la API de Storage, que cuenta
contra la cuota de Cached Egress ya agotada.

Se corre una sola vez desde .github/workflows/migrate-model-blob.yml.
Seguro de repetir (ON CONFLICT DO UPDATE en storage_upload)."""

from pathlib import Path

from src import pg_client

MODEL_CACHE_DIR = Path(".model_cache")

champions = pg_client.select_all(
    "champion", select="horizon_min,model_id", order="horizon_min.asc"
)

for row in champions:
    model_id = row["model_id"]
    cache_path = MODEL_CACHE_DIR / f"{model_id}.joblib"
    if not cache_path.exists():
        print(f"AVISO: +{row['horizon_min']}min ({model_id}) no está en la caché local — "
              f"no se puede migrar sin pasar por Storage. Se deja pendiente.")
        continue
    data = cache_path.read_bytes()
    storage_path = f"{model_id}/gbm.joblib"
    pg_client.storage_upload("models", storage_path, data)
    print(f"+{row['horizon_min']}min ({model_id}): {len(data)} bytes migrados a model_blob.")

print("Listo.")
