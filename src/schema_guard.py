"""Detector de cambios de formato en el API del profesor.

El profesor avisó que va a meter un drift mucho más fuerte y puede
cambiar el formato de los datos (nombres de campo, unidades,
resolución). Un cambio así puede fallar de dos formas muy distintas:

  1. Campo faltante/tipo incompatible -> KeyError/TypeError feo, varias
     capas abajo de donde pasó -> el pipeline ya falla, pero sin decir
     claramente QUÉ cambió.
  2. Cambio "silencioso" (unidades, por ejemplo) -> nada truena, pero
     el modelo entrena con datos mal escalados sin que nadie se dé
     cuenta hasta que el accuracy ya se desplomó en producción.

Este módulo valida la forma cruda de lo que devuelve el API ANTES de
que entre al resto del pipeline, para el caso 1 decir exactamente qué
cambió (no un traceback genérico) y para el caso 2 atrapar al menos
los cambios de escala groseros (ver _DEMAND_SANITY_MAX). No intenta
adivinar ni arreglar el formato nuevo solo —eso es más peligroso que
fallar, porque un arreglo mal adivinado se ve igual de silencioso que
el problema original— solo detectarlo rápido y avisar.

Aviso automático: si corre en GitHub Actions (GITHUB_TOKEN/
GITHUB_REPOSITORY en el entorno, igual que ya usa monitor.py para
disparar train.yml por drift), abre un issue con el diagnóstico
completo para que se note y se pueda reaccionar en minutos, no días.
Si no hay esas variables (corriendo local), solo imprime fuerte.
"""

import json
import os

import requests

# Rango de demanda plausible por fila (una estación, un timestamp de
# ~15min). Nunca se vio nada cerca de esto en producción (máximo real
# observado ~2400) — generoso A PROPÓSITO (subido de 20000 a 50000,
# 2026-10-03) para no disparar por el drift mucho más fuerte que el
# profesor avisó que viene: un bug real de escala/unidad típicamente
# se sale por varios órdenes de magnitud (x100, x1000), así que sigue
# atrapando eso sin arriesgar bloquear demanda real más intensa.
_DEMAND_SANITY_MAX = 50000

_OBSERVATION_REQUIRED_KEYS = {"station_id", "observed_at", "demand"}
_STATION_REQUIRED_KEYS = {"station_id", "station_name", "corridor", "latitude", "longitude"}
_CONTEXT_REQUIRED_KEYS = {"observed_at"}
_CYCLE_REQUIRED_KEYS = {"state", "cycle_id", "data_cutoff", "targets"}
_TARGET_REQUIRED_KEYS = {"station_id", "target_at"}


class SchemaDriftError(RuntimeError):
    """El API devolvió una forma de datos distinta a la esperada."""


def _describe_mismatch(label, sample, required_keys):
    actual_keys = set(sample.keys())
    missing = required_keys - actual_keys
    extra = actual_keys - required_keys
    parts = [f"{label}: la fila no tiene la forma esperada."]
    if missing:
        parts.append(f"faltan campos: {sorted(missing)}")
    if extra:
        parts.append(f"campos nuevos no esperados: {sorted(extra)}")
    parts.append(f"fila real recibida: {json.dumps(sample, default=str)[:500]}")
    return " | ".join(parts)


def _check_required_keys(label, rows, required_keys):
    if not rows:
        return
    sample = rows[0]
    if not required_keys.issubset(sample.keys()):
        raise SchemaDriftError(_describe_mismatch(label, sample, required_keys))


def validate_stations(rows):
    _check_required_keys("estaciones (/v1/stations)", rows, _STATION_REQUIRED_KEYS)


def validate_context(rows):
    _check_required_keys("contexto (/v1/context o /v1/stream)", rows, _CONTEXT_REQUIRED_KEYS)


def validate_cycle(cycle):
    """Valida /v1/forecast-cycles/current ANTES de mirar cycle["state"].

    Punto ciego real encontrado 2026-10-03: get_current_cycle() usa
    cycle.get("state") != "open" para decidir si hay algo que predecir.
    Si el profesor renombra "state" (parte del cambio de formato que
    avisó que viene), .get() no truena — devuelve None, que != "open",
    así que el pipeline trata CADA ciclo como cerrado, en silencio, para
    siempre, sin ningún error que avise. Validar la forma ANTES de leer
    "state" convierte ese silencio en una alerta clara."""
    label = "ciclo vigente (/v1/forecast-cycles/current)"
    if not _CYCLE_REQUIRED_KEYS.issubset(cycle.keys()):
        raise SchemaDriftError(_describe_mismatch(label, cycle, _CYCLE_REQUIRED_KEYS))
    targets = cycle.get("targets")
    if not isinstance(targets, list) or not targets:
        raise SchemaDriftError(f"{label}: 'targets' no es una lista no vacía — valor: {json.dumps(targets, default=str)[:300]}")
    _check_required_keys(f"{label} (targets[0])", targets, _TARGET_REQUIRED_KEYS)


def validate_observations(rows):
    label = "observaciones (/v1/observations o /v1/stream/observations)"
    _check_required_keys(label, rows, _OBSERVATION_REQUIRED_KEYS)
    for row in rows:
        demand = row.get("demand")
        if not isinstance(demand, (int, float)) or isinstance(demand, bool):
            raise SchemaDriftError(
                f"{label}: 'demand' no es numérico en esta fila (tipo {type(demand).__name__}) — "
                f"fila: {json.dumps(row, default=str)[:500]}"
            )
        if demand < 0 or demand > _DEMAND_SANITY_MAX:
            raise SchemaDriftError(
                f"{label}: 'demand'={demand} está muy fuera de lo plausible (rango sano: 0-{_DEMAND_SANITY_MAX}) — "
                "esto huele a cambio de unidad/escala, no a drift real de demanda. "
                f"fila: {json.dumps(row, default=str)[:500]}"
            )


_ISSUE_TITLE = "Cambio de formato detectado en el API del profesor"


def report_schema_drift(error: SchemaDriftError):
    """Imprime el diagnóstico siempre; si hay credenciales de GitHub
    Actions en el entorno, además abre un issue para que se note sin
    tener que estar revisando logs.

    Dedupe (2026-10-03): predict.yml corre cada 10min — sin esto, un
    formato roto dispararía un issue nuevo cada corrida (144/día) hasta
    que alguien lo arregle, ahogando la señal real en ruido. Se
    reutiliza el issue abierto existente con el mismo título (un
    comentario con el error más reciente) en vez de crear uno nuevo."""
    print(f"::error::CAMBIO DE FORMATO DETECTADO — {error}")

    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        print("(sin GITHUB_TOKEN/GITHUB_REPOSITORY en el entorno — no se puede abrir el issue automático acá)")
        return

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    body = (
        "El pipeline detectó que el API del profesor devolvió datos con una forma "
        "distinta a la esperada (ver `src/schema_guard.py`). Esto paró la corrida a "
        "propósito en vez de seguir con datos posiblemente mal interpretados.\n\n"
        f"```\n{error}\n```\n\n"
        "Revisar qué cambió exactamente en el API y ajustar `src/ingest.py`/"
        "`src/pipeline/sync.py` (y `schema_guard.py` si el formato nuevo es el "
        "definitivo) antes de reintentar."
    )
    try:
        existing = requests.get(
            f"https://api.github.com/repos/{repo}/issues",
            headers=headers,
            params={"state": "open", "per_page": 20},
            timeout=15,
        )
        existing_issue = None
        if existing.ok:
            for issue in existing.json():
                if issue.get("title") == _ISSUE_TITLE and "pull_request" not in issue:
                    existing_issue = issue
                    break
        if existing_issue:
            r = requests.post(
                existing_issue["comments_url"],
                headers=headers,
                json={"body": f"Volvió a pasar:\n\n{body}"},
                timeout=15,
            )
            if r.status_code == 201:
                print(f"Comentario agregado al issue ya abierto: {existing_issue.get('html_url')}")
            else:
                print(f"No se pudo comentar en el issue existente ({r.status_code}): {r.text[:300]}")
            return
    except requests.RequestException as e:
        print(f"No se pudo chequear issues existentes (error de red), se intenta crear uno nuevo igual: {e}")

    try:
        r = requests.post(
            f"https://api.github.com/repos/{repo}/issues",
            headers=headers,
            json={"title": _ISSUE_TITLE, "body": body},
            timeout=15,
        )
        if r.status_code in (200, 201):
            print(f"Issue de diagnóstico creado: {r.json().get('html_url')}")
        else:
            print(f"No se pudo crear el issue de diagnóstico ({r.status_code}): {r.text[:300]}")
    except requests.RequestException as e:
        print(f"No se pudo crear el issue de diagnóstico (error de red): {e}")
