"""Inferencia + submission del ciclo vigente — workflow cada 10 min.

Sigue el pseudocódigo de la guía operativa v2.0 al pie de la letra:

    sync_observations_from_saved_cursor()
    cycle = get_current_cycle()
    if cycle is None: exit_successfully()
    model = load_promoted_model()
    if receipt_exists(cycle.id, model.version): exit_successfully()
    rows = build_features_as_of(cycle.data_cutoff, cycle.targets)
    predictions = model.predict(rows)
    validate_exact_targets(predictions, cycle.targets)
    key = stable_key(cycle.id, model.version, predictions)
    receipt = post_submission(...)
    save_receipt(receipt)

Cada paso corresponde 1:1 a una función de este módulo, en el mismo
orden, para que se pueda auditar contra el pseudocódigo del PDF.
"""

import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import requests

from src import supabase_client as sb
from src.features import build_feature_frame
from src.pipeline.sync import sync_observations_from_saved_cursor
from src.train import FEATURE_COLS

API_BASE = os.environ.get("PULSO_API_BASE", "https://pulso-transmi.72-60-245-2.sslip.io")
PULSO_API_KEY = os.environ.get("PULSO_API_KEY", "")
MODELS_BUCKET = "models"
HORIZONS = [15, 30, 45, 60]


def get_current_cycle():
    """La API es la autoridad: el 404 no_open_cycle (o un estado que no
    sea "open") es un resultado normal, no un error — el workflow debe
    terminar en verde sin intentar nada más."""
    r = requests.get(f"{API_BASE}/v1/forecast-cycles/current", timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    cycle = r.json()
    if cycle.get("state") != "open":
        return None
    return cycle


def load_promoted_model():
    """Lee el puntero `champion` (uno por horizonte), descarga cada
    artefacto de Supabase Storage y arma un objeto único con todo lo
    necesario para predecir. Nunca asume "el último archivo local" —
    todo sale de Supabase, que es lo único que persiste entre
    ejecuciones del workflow (cada corrida arranca de un checkout
    limpio, sin artifacts/ en disco)."""
    champion_rows = sb.select_all("champion", order="horizon_min.asc")
    if len(champion_rows) != len(HORIZONS):
        raise RuntimeError(f"champion incompleto: {len(champion_rows)}/{len(HORIZONS)} horizontes tienen puntero")

    models = {}
    winner_by_horizon = {}
    versions = set()
    trained_ats = set()
    training_data_ends = set()
    for row in champion_rows:
        model_id = row["model_id"]
        modelo_row = sb.select_one("modelo", filters={"model_id": f"eq.{model_id}"})
        if modelo_row is None:
            raise RuntimeError(f"champion apunta a model_id inexistente en modelo: {model_id}")
        versions.add(modelo_row["version"])
        trained_ats.add(modelo_row["trained_at"])
        training_data_ends.add(modelo_row["cutoff_train_fin"])
        feature_list = modelo_row["feature_list"]
        winner_by_horizon[row["horizon_min"]] = feature_list.get("winner_by_station", {})

        artifact_uri = modelo_row["artifact_uri"]
        assert artifact_uri.startswith("supabase-storage://models/"), f"artifact_uri inesperado: {artifact_uri}"
        storage_path = artifact_uri.removeprefix("supabase-storage://models/")
        blob = sb.storage_download(MODELS_BUCKET, storage_path)
        bundle = joblib.load(io.BytesIO(blob))
        bundle["_model_id"] = model_id
        models[row["horizon_min"]] = bundle  # {"model":..., "station_categories":[...], "_model_id":...}

    if len(versions) != 1:
        print(f"AVISO: los champions de distintos horizontes tienen versiones distintas: {versions}")
    model_version = sorted(versions)[0] if versions else "unknown"
    trained_at = sorted(trained_ats)[-1] if trained_ats else None
    training_data_end_date = sorted(training_data_ends)[-1] if training_data_ends else None
    training_data_end = f"{training_data_end_date}T00:00:00Z" if training_data_end_date else None

    return {
        "models": models,
        "winner_by_horizon": winner_by_horizon,
        "version": model_version,
        "trained_at": trained_at,
        "training_data_end": training_data_end,
    }


def receipt_exists(cycle_id, model_version):
    row = sb.select_one(
        "submission_receipt",
        filters={"cycle_id": f"eq.{cycle_id}", "model_version": f"eq.{model_version}"},
    )
    return row is not None


def _forward_fill_context(context_rows, cutoff_ts, cutoff_str):
    """El contexto no se publica vía stream (solo el histórico fijo).
    Si `cutoff_ts` no tiene fila de contexto, se repite el último
    contexto conocido — documentado aquí y en el log, nunca en
    silencio. Compara por Timestamp (no por string crudo): Supabase y
    el API de Pulso TransMi pueden serializar el mismo instante con
    formato distinto ("...Z" vs "...+00:00")."""
    if not context_rows:
        return context_rows
    have_ts = {pd.Timestamp(r["observed_at"]) for r in context_rows}
    if cutoff_ts in have_ts:
        return context_rows
    last = max(context_rows, key=lambda r: pd.Timestamp(r["observed_at"]))
    print(f"AVISO: contexto faltante en data_cutoff={cutoff_str}; forward-fill desde {last['observed_at']}")
    row = dict(last)
    row["observed_at"] = cutoff_str
    return context_rows + [row]


def build_features_as_of(data_cutoff, targets):
    observations = sb.select_all("observacion", select="station_id,observed_at,demand")
    context = sb.select_all("contexto")

    cutoff_ts = pd.Timestamp(data_cutoff)
    context = _forward_fill_context(context, cutoff_ts, data_cutoff)

    obs_df = pd.DataFrame(observations)
    ctx_df = pd.DataFrame(context)
    base = build_feature_frame(obs_df, ctx_df)

    cutoff_rows = base[base["observed_at"] == cutoff_ts]
    return {row["station_id"]: row for _, row in cutoff_rows.iterrows()}, base


def predict_targets(model, cutoff_row_by_station, targets):
    lookup = model["_naive_lookup"]
    station_mean = model["_naive_station_mean"]
    predictions = []
    for t in targets:
        sid = t["station_id"]
        horizon_min = t["horizon_minutes"]
        feat_row = cutoff_row_by_station.get(sid)
        if feat_row is None:
            continue

        winner = model["winner_by_horizon"].get(horizon_min, {}).get(sid, "naive")
        has_nan_features = bool(feat_row[FEATURE_COLS].isna().any())
        if winner == "gbm" and horizon_min in model["models"] and not has_nan_features:
            bundle = model["models"][horizon_min]
            X = pd.DataFrame([feat_row[FEATURE_COLS]])
            X["station_id"] = pd.Categorical(X["station_id"], categories=bundle["station_categories"])
            value = float(bundle["model"].predict(X)[0])
        else:
            key = (sid, int(feat_row["hour"]), int(feat_row["day_of_week"]))
            value = float(lookup.get(key, station_mean.get(sid, 0.0)))

        value = max(0.0, value)
        predictions.append({
            "station_id": sid,
            "target_at": t["target_at"],
            "horizon_minutes": horizon_min,
            "value": round(value, 2) if np.isfinite(value) else None,
        })
    return predictions


def validate_exact_targets(predictions, targets):
    expected = {(t["station_id"], t["target_at"]) for t in targets}
    got = {(p["station_id"], p["target_at"]) for p in predictions}
    if got != expected:
        missing = expected - got
        extra = got - expected
        raise ValueError(f"Batch no coincide con los targets del ciclo. Faltan: {missing}. De más: {extra}")
    for p in predictions:
        if p["value"] is None or p["value"] < 0 or not np.isfinite(p["value"]):
            raise ValueError(f"Predicción inválida (no finita o negativa): {p}")


def stable_key(cycle_id, model_version):
    return f"{cycle_id}::{model_version}"


def post_submission(cycle, model, predictions, git_commit):
    model_version = model["version"]
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": f"gha_{cycle['cycle_id']}_{model_version}"[:128],
        "data_cutoff": cycle["data_cutoff"],
        "model": {
            "version": model_version,
            "trained_at": model.get("trained_at"),
            "training_data_end": model.get("training_data_end"),
            "git_commit": git_commit,
        },
        "predictions": [{"station_id": p["station_id"], "target_at": p["target_at"], "value": p["value"]} for p in predictions],
    }
    idem_key = stable_key(cycle["cycle_id"], model_version)
    r = requests.post(
        f"{API_BASE}/v1/submissions",
        headers={
            "Authorization": f"Bearer {PULSO_API_KEY}",
            "Idempotency-Key": idem_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    return r, idem_key, payload


def save_receipt(cycle, model_version, resp_json, idem_key):
    sb.write(
        "submission_receipt",
        [{
            "cycle_id": cycle["cycle_id"],
            "model_version": model_version,
            "submission_id": resp_json["submission_id"],
            "attempt": resp_json.get("attempt", 1),
            "status": resp_json.get("status", "accepted"),
            "payload_hash": resp_json.get("payload_hash", ""),
            "data_cutoff": cycle["data_cutoff"],
            "predictions_sent": resp_json.get("predictions_received", 0),
            "idempotency_key": idem_key,
            "received_at": resp_json.get("received_at", datetime.now(timezone.utc).isoformat()),
        }],
        on_conflict="cycle_id,model_version",
        merge=True,
    )


def save_run_and_predictions(cycle, model_version, models_by_horizon, predictions, status):
    run_id = f"run_{uuid.uuid4().hex[:16]}"
    sb.write(
        "ejecucion_pipeline",
        [{
            "run_id": run_id,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "cursor_hasta": cycle["data_cutoff"],
            "status": status,
            "drift_metric": None,
            "decision_reentrenar": False,
            "motivo_decision": "inferencia periódica (predict.yml)",
            "model_id": None,
        }],
        on_conflict="run_id",
    )
    if not predictions:
        return
    rows = []
    for p in predictions:
        model_id = models_by_horizon.get(p["horizon_minutes"], {}).get("_model_id")
        rows.append({
            "prediction_id": f"pred_{uuid.uuid4().hex[:16]}",
            "run_id": run_id,
            "model_id": model_id,
            "station_id": p["station_id"],
            "target_timestamp": p["target_at"],
            "horizonte": p["horizon_minutes"] // 15,
            "demanda_predicha": p["value"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
    rows = [r for r in rows if r["model_id"]]
    if rows:
        sb.write("prediccion", rows, on_conflict="prediction_id")


def main():
    git_commit = os.environ.get("GITHUB_SHA", "unknown")

    n_synced = sync_observations_from_saved_cursor()
    print(f"sync: {n_synced} observaciones nuevas")

    cycle = get_current_cycle()
    if cycle is None:
        print("No hay ciclo abierto ahora mismo. Fin correcto.")
        return

    print(f"Ciclo vigente: {cycle['cycle_id']}  data_cutoff={cycle['data_cutoff']}  closes_at={cycle.get('closes_at')}")

    model = load_promoted_model()

    if receipt_exists(cycle["cycle_id"], model["version"]):
        print(f"Ya existe recibo para {cycle['cycle_id']} + {model['version']}. Fin correcto (sin reenviar).")
        return

    cutoff_row_by_station, base = build_features_as_of(cycle["data_cutoff"], cycle["targets"])
    print(f"Estaciones con feature_row en data_cutoff: {len(cutoff_row_by_station)}/12")

    model["_naive_lookup"] = base.groupby(["station_id", "hour", "day_of_week"])["target_demand"].mean()
    model["_naive_station_mean"] = base.groupby("station_id")["target_demand"].mean()

    predictions = predict_targets(model, cutoff_row_by_station, cycle["targets"])

    try:
        validate_exact_targets(predictions, cycle["targets"])
    except ValueError as e:
        print(f"Batch inválido, no se envía: {e}")
        save_run_and_predictions(cycle, model["version"], {}, [], status="error")
        sys.exit(1)

    resp, idem_key, payload = post_submission(cycle, model, predictions, git_commit)
    print(f"POST /v1/submissions -> {resp.status_code}")

    if resp.status_code in (200, 201):
        resp_json = resp.json()
        save_receipt(cycle, model["version"], resp_json, idem_key)
        models_by_horizon = {h: {"_model_id": bundle.get("_model_id")} for h, bundle in model["models"].items()}
        save_run_and_predictions(cycle, model["version"], models_by_horizon, predictions, status="ok")
        print(f"Recibo guardado: {resp_json.get('submission_id')}")
    elif resp.status_code == 404:
        print("404 no_open_cycle al enviar (el ciclo cerró mientras predecíamos). Fin correcto.")
    elif resp.status_code == 409:
        print("409: ciclo cerrado, en conflicto o límite de intentos alcanzado. No se reintenta a ciegas.")
    elif resp.status_code == 422:
        print(f"422: el batch viola el contrato — revisar ensamblaje, no el modelo. Detalle: {resp.text[:500]}")
        sys.exit(1)
    elif resp.status_code == 429:
        print("429: exceso de solicitudes. El próximo despertar del cron reintenta con la misma llave.")
    else:
        print(f"Respuesta inesperada ({resp.status_code}): {resp.text[:500]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
