"""Inferencia + submission — Fase 4 de la guía metodológica.

Consulta el ciclo de predicción vigente (`/v1/forecast-cycles/current`),
arma el batch de predicciones que ese ciclo pidió exactamente (ni una
estación/horizonte de más) con el modelo campeón — híbrido: por cada
(estación, horizonte) usa naive o GBM v2, según lo que ganó en
validación (artifacts/training_summary.json) — y lo envía a
POST /v1/submissions con Idempotency-Key.

Riesgo de fuga de datos: las features de cada estación se calculan con
`build_feature_frame` sobre el histórico hasta `data_cutoff` (mismo
código que en entrenamiento, no una reimplementación aparte) y se toma
la fila cuyo `observed_at == data_cutoff`. Si esa fila no existe (el
collector incremental no llegó hasta el cutoff todavía), la estación se
salta y se avisa — nunca se inventa un valor con datos más viejos sin
decirlo.

Limitación conocida: `hour`/`day_of_week` describen el momento del
`data_cutoff`, no el del `target_at` futuro (así se entrenó). Con
horizontes de máximo 60 min casi siempre coinciden, pero cerca de la
medianoche puede haber un desfase de una hora — pendiente para una
v3 si se justifica por el impacto medido.
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import build_feature_frame
from src.train import FEATURE_COLS

API_BASE = os.environ.get("PULSO_API_BASE", "https://pulso-transmi.72-60-245-2.sslip.io")
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
MODEL_VERSION = "v2"
TRAINING_DATA_END = "2026-09-01T00:00:00Z"  # cutoff_train_fin registrado en Supabase.modelo


def fetch_current_cycle():
    clock = requests.get(f"{API_BASE}/v1/clock", timeout=15).json()
    cycle = requests.get(f"{API_BASE}/v1/forecast-cycles/current", timeout=15).json()
    return clock, cycle


def load_winner_by_station():
    summary = json.loads((ARTIFACTS_DIR / "training_summary.json").read_text())
    return {row["horizon_min"]: row["winner_by_station"] for row in summary}


def load_gbm_models():
    models = {}
    for path in ARTIFACTS_DIR.glob("gbm_h*.joblib"):
        horizon_min = int(path.stem.replace("gbm_h", ""))
        bundle = joblib.load(path)
        models[horizon_min] = bundle  # {"model":..., "station_categories":[...]}
    return models


def naive_tables(base_df):
    """Reconstruye las tablas del baseline naive con TODO el histórico
    disponible (no solo train/val de la validación): en inferencia real
    no hay motivo para descartar datos recientes, a diferencia de la
    evaluación donde el corte temporal existe para no hacer trampa."""
    group_cols = ["station_id", "hour", "day_of_week"]
    lookup = base_df.groupby(group_cols)["target_demand"].mean()
    station_mean = base_df.groupby("station_id")["target_demand"].mean()
    return lookup, station_mean


def predict_batch(base_df, cutoff_row_by_station, targets, winner_by_horizon, gbm_models, naive_lookup, naive_station_mean):
    lookup, station_mean = naive_lookup, naive_station_mean
    predictions = []
    skipped = []

    for t in targets:
        sid = t["station_id"]
        horizon_min = t["horizon_minutes"]
        feat_row = cutoff_row_by_station.get(sid)
        if feat_row is None:
            skipped.append(sid)
            continue

        winner = winner_by_horizon.get(horizon_min, {}).get(sid, "naive")
        if winner == "gbm" and horizon_min in gbm_models:
            bundle = gbm_models[horizon_min]
            X = pd.DataFrame([feat_row[FEATURE_COLS]])
            X["station_id"] = pd.Categorical(X["station_id"], categories=bundle["station_categories"])
            value = float(bundle["model"].predict(X)[0])
        else:
            key = (sid, int(feat_row["hour"]), int(feat_row["day_of_week"]))
            value = lookup.get(key, station_mean.get(sid, 0.0))

        value = max(0.0, float(value))
        if not np.isfinite(value):
            skipped.append(sid)
            continue

        predictions.append({
            "station_id": sid,
            "target_at": t["target_at"],
            "value": round(value, 2),
        })

    return predictions, skipped


def build_submission(cycle, predictions, git_commit):
    return {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": f"jorge_{uuid.uuid4().hex[:12]}",
        "data_cutoff": cycle["data_cutoff"],
        "model": {
            "version": MODEL_VERSION,
            "trained_at": "2026-09-18T21:10:00Z",
            "training_data_end": TRAINING_DATA_END,
            "git_commit": git_commit,
        },
        "predictions": predictions,
    }


def submit(payload, api_key, idempotency_key):
    r = requests.post(
        f"{API_BASE}/v1/submissions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Idempotency-Key": idempotency_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", default="observations.json")
    parser.add_argument("--context", default="context.json")
    parser.add_argument("--api-key", default=os.environ.get("PULSO_API_KEY"))
    parser.add_argument("--git-commit", default=os.environ.get("GITHUB_SHA", "unknown"))
    parser.add_argument("--dry-run", action="store_true", help="Arma el batch pero no lo envía.")
    args = parser.parse_args()

    clock, cycle = fetch_current_cycle()
    if clock.get("state") != "open" and cycle.get("state") != "open":
        print(f"No hay ciclo abierto ahora mismo (clock={clock.get('state')}, cycle={cycle.get('state')}). No se envía nada.")
        return

    print(f"Ciclo vigente: {cycle['cycle_id']}  data_cutoff={cycle['data_cutoff']}  closes_at={cycle.get('closes_at')}")
    print(f"Targets solicitados: {len(cycle['targets'])}")

    observations = pd.DataFrame(json.load(open(args.observations)))
    context = pd.DataFrame(json.load(open(args.context)))
    base = build_feature_frame(observations, context)

    cutoff = pd.Timestamp(cycle["data_cutoff"])
    cutoff_rows = base[base["observed_at"] == cutoff]
    cutoff_row_by_station = {row["station_id"]: row for _, row in cutoff_rows.iterrows()}
    print(f"Estaciones con feature_row en data_cutoff: {len(cutoff_row_by_station)}/12")

    naive_lookup, naive_station_mean = naive_tables(base)
    gbm_models = load_gbm_models()
    winner_by_horizon = load_winner_by_station()

    predictions, skipped = predict_batch(
        base, cutoff_row_by_station, cycle["targets"], winner_by_horizon, gbm_models, naive_lookup, naive_station_mean
    )
    if skipped:
        print(f"AVISO: {len(skipped)} predicciones no se pudieron generar (sin feature_row en cutoff): {skipped}")

    if len(predictions) != len(cycle["targets"]):
        print(f"Batch incompleto: {len(predictions)}/{len(cycle['targets'])} targets cubiertos. No se envía (la API rechaza batches parciales).")
        return

    payload = build_submission(cycle, predictions, args.git_commit)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    if not args.api_key:
        print("Falta --api-key o la variable de entorno PULSO_API_KEY. No se envía.")
        return

    idempotency_key = f"{cycle['cycle_id']}_{payload['client_run_id']}"
    resp = submit(payload, args.api_key, idempotency_key)
    print(f"POST /v1/submissions -> {resp.status_code}")
    print(resp.text[:2000])


if __name__ == "__main__":
    main()
