"""Reintento con backoff corto para llamadas HTTP a servicios externos
(la API del profesor, sobre todo) — descubierto en producción el
2026-10-01: un timeout de 30s contra pulso-transmi.72-60-245-2.sslip.io
(su servidor, no Supabase ni GitHub) tiraba todo el ciclo de
predict.yml sin reintentar, perdiendo las 12 estaciones x 4 horizontes
de ESE ciclo entero por un solo hipo de red transitorio.

Nunca reintenta indefinido: predict.yml corre con timeout-minutes: 8 y
un cron cada 10 min, así que un reintento sin límite podría comerse el
ciclo completo en vez de perder solo este. RETRY_MAX_ATTEMPTS=3 con
backoff corto (2s, 5s) cubre un hipo momentáneo sin arriesgar el
timeout del job."""

import time

import requests

RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (2, 5)  # espera antes del 2do y 3er intento


def request_with_retry(method, url, **kwargs):
    """Igual que requests.request, pero reintenta errores de conexión/
    timeout transitorios (nunca errores HTTP como 404/500 — esos ya los
    maneja el caller con raise_for_status/status_code, son respuestas
    reales del servidor, no un problema de red)."""
    last_exc = None
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            return requests.request(method, url, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            if attempt < RETRY_MAX_ATTEMPTS - 1:
                wait = RETRY_BACKOFF_SECONDS[attempt]
                print(f"  aviso: {method} {url} falló (intento {attempt + 1}/{RETRY_MAX_ATTEMPTS}): {exc!r} -- reintenta en {wait}s")
                time.sleep(wait)
    raise last_exc
