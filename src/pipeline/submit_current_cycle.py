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
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import supabase_client as sb
from src import train as train_mod
from src.features import build_feature_frame, climatological_context, estimate_context_row, target_time_features
from src.http_retry import request_with_retry
from src.pipeline.sync import sync_observations_from_saved_cursor
from src.schema_guard import SchemaDriftError, report_schema_drift, validate_cycle
from src.train import FEATURE_COLS, weighted_naive_tables

API_BASE = os.environ.get("PULSO_API_BASE", "https://pulso-transmi.72-60-245-2.sslip.io")
PULSO_API_KEY = os.environ.get("PULSO_API_KEY", "")
MODELS_BUCKET = "models"
HORIZONS = [15, 30, 45, 60]
# Egress de Supabase Storage: cada modelo GBM (gbm.joblib) pesa 7-9 MB (ensemble
# de 3 HistGradientBoostingRegressor, max_depth=8). Este módulo corre cada 10
# min, 144 veces al día — bajar los 4 champions COMPLETOS en cada corrida
# (~34MB/ciclo) sale a ~5GB/día aunque el champion casi nunca cambie. Con esto
# en su lugar, solo se descarga de Storage cuando el model_id del champion
# realmente cambió desde la última corrida (una promoción real) — el resto de
# las veces se lee del caché local, que persiste entre corridas vía
# actions/cache en predict.yml (ver ese workflow).
MODEL_CACHE_DIR = Path(".model_cache")


def get_current_cycle():
    """La API es la autoridad: el 404 no_open_cycle (o un estado que no
    sea "open") es un resultado normal, no un error — el workflow debe
    terminar en verde sin intentar nada más."""
    r = request_with_retry("GET", f"{API_BASE}/v1/forecast-cycles/current", timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    cycle = r.json()
    # Validar la forma ANTES de leer "state" (ver schema_guard.validate_cycle):
    # si el profesor renombra ese campo, cycle.get("state") nunca truena —
    # devuelve None, que != "open", y el pipeline trataría cada ciclo como
    # cerrado en silencio, para siempre, sin ninguna alerta.
    try:
        validate_cycle(cycle)
    except SchemaDriftError as e:
        report_schema_drift(e)
        raise
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
    blend_weight_by_horizon = {}
    feature_cols_by_horizon = {}
    boost_validated_by_horizon = {}
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
        # Cada horizonte guarda su propia lista de features (feature_list["features"]),
        # tal como las vio SU modelo al entrenar — nunca la constante FEATURE_COLS
        # del código actual. Los horizontes se reentrenan y promueven en momentos
        # distintos (ver promote.py: cada uno se compara y decide por separado), así
        # que en cualquier momento pueden convivir champions entrenados con
        # versiones de features distintas del feature engineering.
        feature_cols_by_horizon[row["horizon_min"]] = feature_list["features"]
        mix_weights_by_station = feature_list.get("mix_weights_by_station")
        if mix_weights_by_station is None:
            # Champion de antes del blend de 3 vías (naive/fast/GBM): mismo
            # peso de GBM que tenía, fast=0 (ver promote.load_champion_bundle).
            blend_weight_by_station = feature_list.get("blend_weight_by_station")
            if blend_weight_by_station is None:
                blend_weight_by_station = {
                    sid: (1.0 if winner == "gbm" else 0.0)
                    for sid, winner in feature_list.get("winner_by_station", {}).items()
                }
            mix_weights_by_station = {sid: {"gbm": w, "fast": 0.0} for sid, w in blend_weight_by_station.items()}
        blend_weight_by_horizon[row["horizon_min"]] = mix_weights_by_station
        # Champion de antes de la autovalidación del boost (2026-09-30):
        # sin este campo, ninguna estación pasa el filtro — comportamiento
        # seguro por defecto (sin boost) hasta reentrenar con código nuevo.
        boost_validated_by_horizon[row["horizon_min"]] = set(feature_list.get("boost_validated_stations", []))

        artifact_uri = modelo_row["artifact_uri"]
        assert artifact_uri.startswith("supabase-storage://models/"), f"artifact_uri inesperado: {artifact_uri}"
        storage_path = artifact_uri.removeprefix("supabase-storage://models/")

        cache_path = MODEL_CACHE_DIR / f"{model_id}.joblib"
        if cache_path.exists():
            bundle = joblib.load(cache_path)
        else:
            blob = sb.storage_download(MODELS_BUCKET, storage_path)
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(blob)
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
        "blend_weight_by_horizon": blend_weight_by_horizon,
        "feature_cols_by_horizon": feature_cols_by_horizon,
        "boost_validated_by_horizon": boost_validated_by_horizon,
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
    """Red de seguridad: normalmente sync.py ya deja `contexto` con una
    fila (climatológica) para cada timestamp nuevo antes de llegar
    aquí. Si por algún motivo `cutoff_ts` sigue sin fila, se estima con
    la misma climatología (promedio histórico real por hora:minuto),
    nunca repitiendo la última lectura real — ver features.py y
    pipeline/sync.py para el razonamiento completo. Compara por
    Timestamp, no por string crudo: Supabase y el API pueden serializar
    el mismo instante con formato distinto ("...Z" vs "...+00:00")."""
    if not context_rows:
        return context_rows
    have_ts = {pd.Timestamp(r["observed_at"]) for r in context_rows}
    if cutoff_ts in have_ts:
        return context_rows
    print(f"AVISO: contexto faltante en data_cutoff={cutoff_str}; estimado con climatología")
    climatology = climatological_context(pd.DataFrame(context_rows))
    row = estimate_context_row(cutoff_str, climatology)
    return context_rows + [row]


OBSERVATIONS_LOOKBACK_DAYS = 21  # Egress de Supabase: este módulo corre cada 10
# min y bajaba `observacion` COMPLETA cada vez (miles de filas que solo crecen
# con el tiempo) — el mayor consumidor de egress del proyecto, muy por encima
# del fix de la fecha en el dashboard. lag_672 (la feature más larga) solo
# necesita 7 días hacia atrás, así que con eso alcanzaría — se sube a 21 para
# no degradar demasiado el naive ponderado (weighted_naive_tables, half-life
# 14 días: a los 21 días de historia, comparado con usar TODO el histórico,
# la diferencia mediana en la tabla de lookup es ~3.8 sobre una escala de
# demanda de 20-2000; a 14 días sube a ~6.3 — validado localmente antes de
# este cambio). Sigue acotado en el tiempo: nunca vuelve a crecer sin límite
# aunque `observacion` seguirá creciendo para siempre.
NAIVE_LOOKBACK_DAYS = 21


def build_features_as_of(data_cutoff, targets):
    since = (pd.Timestamp(data_cutoff) - pd.Timedelta(days=max(OBSERVATIONS_LOOKBACK_DAYS, NAIVE_LOOKBACK_DAYS))).isoformat()
    observations = sb.select_all(
        "observacion",
        select="station_id,observed_at,demand",
        filters={"observed_at": f"gte.{since}"},
        order="observed_at.asc,station_id.asc",
    )
    context = sb.select_all("contexto", order="observed_at.asc")

    cutoff_ts = pd.Timestamp(data_cutoff)
    context = _forward_fill_context(context, cutoff_ts, data_cutoff)

    obs_df = pd.DataFrame(observations)
    ctx_df = pd.DataFrame(context)
    base = build_feature_frame(obs_df, ctx_df)

    # Solo diagnóstico/log (ver bug corregido 2026-10-01 en
    # predict_targets): YA NO filtra qué estaciones reciben boost — eso lo
    # decide únicamente model["boost_validated_by_horizon"], el resultado
    # ya validado con backtest real al entrenar. Se recalcula cada ciclo
    # igual, para que el log refleje el drift del momento.
    drifted_stations = train_mod.detect_drifted_stations(obs_df, cutoff_ts)

    cutoff_rows = base[base["observed_at"] == cutoff_ts]
    return {row["station_id"]: row for _, row in cutoff_rows.iterrows()}, base, drifted_stations


def predict_targets(model, cutoff_row_by_station, targets):
    """Por cada target combina la fila de features del cutoff (lags,
    rolling, clima — lo que se sabe HOY) con target_hour/target_day_of_week
    calculados directo de `target_at` (lo único que describe el momento
    FUTURO que se predice) — mismo esquema que usó el entrenamiento.

    Mezcla naive, fast (persistencia rolling_mean_4h) y GBM con los pesos
    congelados por estación (ver train.hybrid_predict) en vez de elegir uno
    solo — mismo criterio que usó el entrenamiento para esta versión de
    champion. El boost se aplica siempre que la estación esté en
    `model["boost_validated_by_horizon"]` para ese horizonte — el champion
    ya la validó con backtest real al entrenar (ver train.run()), igual
    que lo evalúa promote.py al comparar candidato vs. champion.
    `drifted_stations` (recalculado en cada ciclo, ver
    build_features_as_of) ya NO filtra el boost (bug encontrado
    2026-10-01): es el detector de PROMEDIO DIARIO, el mismo que se
    demostró insuficiente para quiebres rápidos dentro de un día (caso
    03000, ver train.py) — como se recalcula cada 10 min, una estación
    podía entrar y salir de ese set ciclo a ciclo aunque el champion ya
    tuviera el boost validado, aplicándolo de forma intermitente en vivo
    en vez de consistente como en el backtest que lo validó. Brecha real
    medida: el mismo champion daba 39.27% reconstruido en backtest
    (promote.champion_accuracy_on, sin este filtro extra) contra 18.8% en
    las predicciones realmente entregadas (con el filtro). Se sigue
    calculando y logueando `drifted_stations` como diagnóstico, nunca
    como gate."""
    lookup = model["_naive_lookup"]
    station_mean = model["_naive_station_mean"]
    # Último recurso para una estación que ni siquiera tiene media histórica
    # propia (nunca vista, o el profe la agrega hoy en medio de más drift) —
    # el promedio entre estaciones sigue siendo mejor que dejar el target sin
    # responder: ver el `continue` que esto reemplaza más abajo.
    global_fallback = float(station_mean.mean()) if len(station_mean) else 0.0
    predictions = []
    for t in targets:
        sid = t["station_id"]
        horizon_min = t["horizon_minutes"]
        feat_row = cutoff_row_by_station.get(sid)
        if feat_row is None:
            # Sin fila de features (estación nueva hoy, o sin ninguna
            # observación aún en el cutoff): antes esto hacía `continue` y
            # dejaba el target sin responder, lo que revienta
            # validate_exact_targets ("Faltan: {...}") y aborta el envío
            # de TODO el ciclo — las otras 11 estaciones se quedaban sin
            # predicción también por culpa de una sola. Mejor responder
            # con el mejor naive disponible (media de la estación si existe,
            # si no la media global) que perder el ciclo completo.
            naive_value = float(station_mean.get(sid, global_fallback))
            predictions.append({
                "station_id": sid,
                "target_at": t["target_at"],
                "horizon_minutes": horizon_min,
                "value": round(max(0.0, naive_value), 2),
            })
            continue

        tgt = target_time_features(t["target_at"])
        combined = {**feat_row.to_dict(), **tgt}

        key = (sid, tgt["target_hour"], tgt["target_day_of_week"])
        naive_value = float(lookup.get(key, station_mean.get(sid, 0.0)))
        fast_raw = combined.get("rolling_mean_4h")
        fast_value = float(fast_raw) if pd.notna(fast_raw) else naive_value
        boost_raw = combined.get("lag_1")
        lag_2_raw = combined.get("lag_2")
        lag_1h_raw = combined.get("lag_1h")
        if pd.notna(boost_raw) and pd.notna(lag_2_raw) and pd.notna(lag_1h_raw):
            # Misma señal que train.hybrid_predict (nunca duplicar la fórmula
            # a mano en los dos lados — ver docstring de blended_boost_signal):
            # mezcla la extrapolación de tendencia SUAVIZADA con el promedio
            # simple de los últimos 2 puntos, para no sobrecorregir cuando el
            # drift es una oscilación rápida en vez de una tendencia sostenida.
            boost_value = float(train_mod.blended_boost_signal(
                [boost_raw], [lag_2_raw], [lag_1h_raw], horizon_min // 15,
            )[0])
        elif pd.notna(boost_raw) and pd.notna(lag_1h_raw):
            boost_value = float(train_mod.trend_extrapolated_signal_smoothed(
                [boost_raw], [lag_1h_raw], horizon_min // 15,
            )[0])
        elif pd.notna(boost_raw):
            boost_value = float(boost_raw)
        else:
            boost_value = naive_value

        mix = model["blend_weight_by_horizon"].get(horizon_min, {}).get(sid, {"gbm": 0.0, "fast": 0.0})
        w_gbm, w_fast = mix["gbm"], mix["fast"]
        # w_boost es aparte de w_fast: multiplica a boost_value (tendencia),
        # nunca a fast_value (rolling_mean_4h) — mismo diseño que
        # train.hybrid_predict, ver ese docstring para el porqué.
        w_boost = 0.0
        w_gbm_eff = w_gbm
        boost_validated = model.get("boost_validated_by_horizon", {}).get(horizon_min, set())
        if sid in boost_validated:
            w_boost = train_mod.compute_fast_boost([naive_value], [boost_value], [True], horizon_min)[0]
            # El boost puede comerle espacio a w_gbm, no solo al sobrante —
            # mismo cambio y mismo motivo que train.hybrid_predict (ver ese
            # docstring): nunca toca w_naive/w_fast, solo reduce w_gbm.
            w_boost = min(w_boost, max(0.0, 1.0 - w_fast))
            room_original = max(0.0, 1.0 - w_gbm - w_fast)
            excess = max(0.0, w_boost - room_original)
            w_gbm_eff = max(0.0, w_gbm - excess)
        feature_cols = model["feature_cols_by_horizon"].get(horizon_min, FEATURE_COLS)
        has_nan_features = any(pd.isna(combined.get(c)) for c in feature_cols)
        if w_gbm_eff > 0 and horizon_min in model["models"] and not has_nan_features:
            bundle = model["models"][horizon_min]
            X = pd.DataFrame([{c: combined[c] for c in feature_cols}])
            X["station_id"] = pd.Categorical(X["station_id"], categories=bundle["station_categories"])
            gbm_value = float(bundle["model"].predict(X)[0])
        else:
            gbm_value, w_gbm_eff = 0.0, 0.0
        w_naive = 1.0 - w_gbm_eff - w_fast - w_boost
        value = w_naive * naive_value + w_fast * fast_value + w_boost * boost_value + w_gbm_eff * gbm_value

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
    # Idempotency-Key ya hace este POST seguro de reintentar: si el primer
    # intento sí llegó al servidor pero la respuesta se perdió en el
    # camino (timeout de lectura, no de conexión), el servidor reconoce
    # la misma key y no duplica la entrega.
    idem_key = stable_key(cycle["cycle_id"], model_version)
    r = request_with_retry(
        "POST",
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

    cutoff_row_by_station, base, drifted_stations = build_features_as_of(cycle["data_cutoff"], cycle["targets"])
    print(f"Estaciones con feature_row en data_cutoff: {len(cutoff_row_by_station)}/12")
    if drifted_stations:
        print(f"Quiebre de demanda detectado (auto): {drifted_stations}")

    # Misma función que usa train.py (weighted_naive_tables) para que el
    # naive en vivo pondere por recencia igual que el que se evaluó y
    # promovió — nunca una copia propia del cálculo, que fue justo el tipo
    # de divergencia entre entrenamiento e inferencia que ya rompió
    # producción una vez (ver feature_cols_by_horizon más arriba).
    model["_naive_lookup"], model["_naive_station_mean"] = weighted_naive_tables(base, "hour", "day_of_week")

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
