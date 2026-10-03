"""Diagnóstico puntual de solo lectura — uso real de Supabase (para no
repetir el incidente de cuota del proyecto anterior). Mide: cuántos
objetos y cuántos MB hay en Storage (bucket "models" — ¿se están
acumulando artefactos viejos sin limpiar?), el estado real del cursor
de sync (¿avanza de verdad o se re-sincroniza lo mismo cada 10min?), y
el tamaño aproximado de las tablas principales. Se corre desde
.github/workflows/diag-supabase-usage.yml. No escribe nada."""

import os

import requests

from src import supabase_client as sb

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]


def _storage_headers():
    return {"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}


def list_storage_objects(bucket, prefix="", limit=1000):
    r = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/list/{bucket}",
        headers={**_storage_headers(), "Content-Type": "application/json"},
        json={"prefix": prefix, "limit": limit, "sortBy": {"column": "name", "order": "asc"}},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def main():
    print("=== Storage: carpetas (una por model_id) en el bucket 'models' ===")
    top_level = list_storage_objects("models")
    folders = [item["name"] for item in top_level if item.get("id") is None]  # las carpetas no tienen "id"
    print(f"carpetas encontradas: {len(folders)}")

    total_bytes = 0
    total_files = 0
    for folder in folders:
        files = list_storage_objects("models", prefix=folder)
        for f in files:
            size = (f.get("metadata") or {}).get("size", 0)
            total_bytes += size
            total_files += 1
        sizes_str = ", ".join(f"{f['name']}={(f.get('metadata') or {}).get('size', 0) / 1e6:.1f}MB" for f in files)
        print(f"  {folder}/: {sizes_str}")

    print(f"\nTOTAL Storage bucket 'models': {total_files} archivos, {total_bytes / 1e6:.1f} MB")
    print("(plan free de Supabase: tope típico de 1GB de Storage total)")

    print("\n=== sync_state: ¿el cursor de verdad avanza? ===")
    state = sb.select_all("sync_state")
    for row in state:
        print(f"  source={row['source']}  cursor_value={row['cursor_value']}  updated_at={row['updated_at']}")

    print("\n=== tamaño aproximado de las tablas principales (conteo de filas, SIN bajar el contenido) ===")
    for table in ["observacion", "contexto", "estacion", "modelo", "champion", "prediccion", "submission_receipt", "ejecucion_pipeline", "metrica_validacion", "operational_metric"]:
        try:
            r = requests.get(
                f"{SUPABASE_URL}/rest/v1/{table}",
                headers={**_storage_headers(), "Range-Unit": "items", "Range": "0-0", "Prefer": "count=exact"},
                params={"select": "*"},
                timeout=30,
            )
            content_range = r.headers.get("Content-Range", "")
            total = content_range.split("/")[-1] if "/" in content_range else "?"
            print(f"  {table}: {total} filas")
        except Exception as e:
            print(f"  {table}: error leyendo ({e})")


if __name__ == "__main__":
    main()
