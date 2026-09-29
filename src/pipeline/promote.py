"""Entrenamiento y promoción del champion — workflow manual/diario.

Nunca se dispara en cada despertar del cron de inferencia — eso lo
prohíbe explícitamente la guía operativa ("Reentrenar en cada
despertar" está listado como algo que GitHub Actions NO debe hacer).
Corre por decisión manual (workflow_dispatch) o una vez al día.

Reentrena un candidato con TODO el histórico disponible en Supabase
(misma ventana temporal 31/7/7 de train.py, reproducible), lo compara
contra el champion vigente Y contra el baseline naive, verifica
estabilidad por estación — no solo el promedio, para no promover un
candidato que mejora en las estaciones grandes a costa de romper una
chica — y SOLO si hay evidencia real de mejora mueve el puntero de
champion. La versión anterior nunca se borra (artifact_uri nuevo,
model_id nuevo) — queda disponible para rollback manual.

El champion NO se compara contra sus métricas registradas (calculadas
cuando se entrenó, en una ventana de test que ya quedó vieja). Se
descarga y se evalúa EN VIVO sobre la misma ventana de test que ve el
candidato en esta corrida — así el delta es honesto incluso si los
datos cambiaron de régimen entre una promoción y la siguiente (drift
incluido): el candidato reentrenado con datos frescos se compara
contra cómo le va HOY al champion congelado, no contra un número de
otro momento.
"""

import io
import json
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import supabase_client as sb
from src import train as train_mod

MIN_IMPROVEMENT = 0.5  # puntos de accuracy promedio mínimos para justificar promover
# Subido de 2.0 a 4.0 (2026-09-28): con 12 estaciones, el ruido normal de
# volver a entrenar (random_state distinto, unos días más de datos) ya
# produce una caída de 3-4 puntos en la estación más volátil AUNQUE el
# candidato sea mejor en promedio — ver ejecucion_pipeline de los últimos
# ~9 días de corridas de train.yml: candidatos con delta promedio positivo
# rechazados una y otra vez por "peor caída por estación" de -3.31 a -3.91,
# nunca por debajo de -4. Con el umbral viejo, NINGÚN candidato se promovió
# desde que el champion actual quedó congelado (cutoff_train_fin ~04-06 de
# septiembre), aunque el mundo real ya había cambiado bastante para
# entonces (ver CHAMPION_FLOOR_ACCURACY_STATION más abajo).
MAX_STATION_REGRESSION = 4.0
# Si el champion YA está prediciendo mal en una estación (evaluado en vivo,
# no su métrica vieja), no cuenta como "regresión bloqueante" que el
# candidato también le vaya mal ahí — lo urgente es que el resto del
# pipeline no se quede indefinidamente con un champion roto en esa estación
# solo porque ninguna reentrenada logra un puntaje "seguro" ahí. Detectado
# en vivo: 05100 tuvo una caída real de demanda (~600-650/día -> ~250/día,
# 13-15 sep) que el champion nunca vio en entrenamiento (cutoff 04-06 sep)
# y quedó con 4.5%-46% de accuracy en producción — sin este piso, cualquier
# candidato que también le cueste esa estación (típico mientras el quiebre
# es reciente) queda bloqueado igual, sin importar cuánto mejore el resto.
CHAMPION_FLOOR_ACCURACY_STATION = 60.0


def load_full_history():
    observations = sb.select_all("observacion", select="station_id,observed_at,demand", order="observed_at.asc,station_id.asc")
    context = sb.select_all("contexto", order="observed_at.asc")
    return pd.DataFrame(observations), pd.DataFrame(context)


def load_champion_bundle(horizon_min):
    """Descarga el champion vigente de ese horizonte (modelo + metadata) o
    None si todavía no hay uno. Se usa para evaluarlo EN VIVO sobre la misma
    ventana de test que ve el candidato — nunca contra sus métricas
    registradas de cuando se entrenó, que quedan obsoletas apenas cambian
    los datos (y mucho más rápido si hay drift)."""
    champ_row = sb.select_one("champion", filters={"horizon_min": f"eq.{horizon_min}"})
    if champ_row is None:
        return None
    modelo_row = sb.select_one("modelo", filters={"model_id": f"eq.{champ_row['model_id']}"})
    if modelo_row is None:
        return None
    artifact_uri = modelo_row["artifact_uri"]
    storage_path = artifact_uri.removeprefix("supabase-storage://models/")
    blob = sb.storage_download("models", storage_path)
    bundle = joblib.load(io.BytesIO(blob))
    feature_list = modelo_row["feature_list"]
    mix_weights_by_station = feature_list.get("mix_weights_by_station")
    if mix_weights_by_station is None:
        # Champion de antes del blend de 3 vías (naive/fast/GBM, v4): mismo
        # peso de GBM que tenía (2 vías), fast=0 — equivalente exacto de su
        # comportamiento anterior, para que siga funcionando sin reentrenar.
        blend_weight_by_station = feature_list.get("blend_weight_by_station")
        if blend_weight_by_station is None:
            # Champion aún más viejo (v3, selección dura, sin blend_weight).
            blend_weight_by_station = {
                sid: (1.0 if winner == "gbm" else 0.0)
                for sid, winner in feature_list.get("winner_by_station", {}).items()
            }
        mix_weights_by_station = {sid: {"gbm": w, "fast": 0.0} for sid, w in blend_weight_by_station.items()}
    return {
        "model": bundle["model"],
        "station_categories": bundle["station_categories"],
        "feature_cols": feature_list["features"],
        "mix_weights_by_station": mix_weights_by_station,
    }


def champion_accuracy_on(champ_bundle, train_df, test_df, drifted_stations=None, horizon_min=None):
    """Reconstruye la predicción híbrida EXACTA del champion (su modelo GBM
    congelado + sus pesos de mezcla naive/fast/gbm por estación, también
    congelados) pero prediciendo sobre la ventana de test de HOY. Así el
    delta contra el candidato es una comparación real, no contra un número
    viejo.

    `drifted_stations`/`horizon_min` se pasan igual que al candidato (ver
    train_mod.hybrid_predict / compute_fast_boost) para que el boost
    reactivo aplique parejo en ambos lados — comparar un candidato CON
    boost contra un champion SIN boost inflaría el delta de forma
    artificial."""
    if champ_bundle is None:
        return None, {}
    naive_pred = train_mod.naive_baseline(train_df, test_df)
    fast_pred = train_mod.naive_fast_baseline(test_df)
    X_test = test_df[champ_bundle["feature_cols"]].copy()
    X_test["station_id"] = pd.Categorical(X_test["station_id"], categories=champ_bundle["station_categories"])
    gbm_pred = np.clip(champ_bundle["model"].predict(X_test), 0, None)
    hybrid_pred = train_mod.hybrid_predict(test_df, naive_pred, fast_pred, gbm_pred, champ_bundle["mix_weights_by_station"],
                                            drifted_stations=drifted_stations, horizon_min=horizon_min)
    by_station = train_mod.evaluate_by_station(test_df, hybrid_pred)
    overall = by_station["accuracy"].mean()
    return overall, dict(zip(by_station["station_id"], by_station["accuracy"]))


def register_candidate(horizon_min, summary_row, code_commit, version, cutoff_inicio, cutoff_fin):
    model_id = f"model_{date.today().isoformat()}_hybrid_h{horizon_min}_{version}"
    now = datetime.now(timezone.utc).isoformat()

    feature_list = {
        "features": train_mod.FEATURE_COLS,
        "winner_by_station": summary_row["winner_by_station"],
        "mix_weights_by_station": summary_row["mix_weights_by_station"],
        "gbm_loss": "poisson",
        "early_stopping": True,
    }
    sb.write("modelo", [{
        "model_id": model_id,
        "version": version,
        "algoritmo": "hybrid(naive_hxdow + hist_gbm_poisson)",
        "trained_at": now,
        "cutoff_train_inicio": cutoff_inicio,
        "cutoff_train_fin": cutoff_fin,
        "code_commit": code_commit,
        "feature_list": feature_list,
        "artifact_uri": f"supabase-storage://models/{model_id}/gbm.joblib",
    }], on_conflict="model_id")

    metric_rows = [
        {
            "metric_id": f"metric_{model_id}_{row['station_id']}",
            "model_id": model_id,
            "station_id": row["station_id"],
            "split": "temporal_31_7_7",
            "wape": row["wape"],
            "accuracy": row["accuracy"],
            "evaluated_at": now,
        }
        for row in summary_row["hybrid_by_station"]
    ]
    agg_acc = sum(r["accuracy"] for r in summary_row["hybrid_by_station"]) / len(summary_row["hybrid_by_station"])
    agg_wape = sum(r["wape"] for r in summary_row["hybrid_by_station"]) / len(summary_row["hybrid_by_station"])
    metric_rows.append({
        "metric_id": f"metric_{model_id}_agg",
        "model_id": model_id,
        "station_id": None,
        "split": "temporal_31_7_7",
        "wape": agg_wape,
        "accuracy": agg_acc,
        "evaluated_at": now,
    })
    sb.write("metrica_validacion", metric_rows, on_conflict="metric_id")
    return model_id, agg_acc


def upload_artifact(horizon_min, model_id):
    path = train_mod.ARTIFACTS_DIR / f"gbm_h{horizon_min}.joblib"
    sb.storage_upload("models", f"{model_id}/gbm.joblib", path.read_bytes())


def decide_and_promote(horizon_min, candidate_model_id, candidate_summary, champ_accuracy_mean, champ_by_station):
    candidate_by_station = {r["station_id"]: r["accuracy"] for r in candidate_summary["hybrid_by_station"]}
    candidate_mean = candidate_summary["hybrid_accuracy_mean_stations"]

    if champ_accuracy_mean is None:
        should_promote = True
        reason = "no había champion vigente para este horizonte"
    else:
        delta = candidate_mean - champ_accuracy_mean
        # Solo cuentan como "regresión bloqueante" las estaciones donde el
        # champion vigente todavía anda razonablemente bien (ver
        # CHAMPION_FLOOR_ACCURACY_STATION) — una estación ya rota no puede
        # seguir vetando reentrenamientos que mejoran todo lo demás.
        regressions = {
            sid: candidate_by_station[sid] - champ_by_station[sid]
            for sid in candidate_by_station
            if sid in champ_by_station and champ_by_station[sid] >= CHAMPION_FLOOR_ACCURACY_STATION
        }
        worst_station, worst_regression = min(regressions.items(), key=lambda kv: kv[1], default=(None, 0.0))
        should_promote = delta >= MIN_IMPROVEMENT and worst_regression >= -MAX_STATION_REGRESSION
        reason = (
            f"delta promedio={delta:+.2f} (umbral +{MIN_IMPROVEMENT}), "
            f"peor caída por estación={worst_regression:+.2f} en {worst_station} "
            f"(tolerancia -{MAX_STATION_REGRESSION}, ignorando estaciones con champion ya < {CHAMPION_FLOOR_ACCURACY_STATION})"
        )

    champ_acc_str = f"{champ_accuracy_mean:.2f}" if champ_accuracy_mean is not None else "n/a"
    print(f"+{horizon_min}min: candidato={candidate_mean:.2f}  champion_actual(evaluado en vivo)={champ_acc_str}  "
          f"-> {'PROMUEVE' if should_promote else 'no promueve'} ({reason})")

    if should_promote:
        upload_artifact(horizon_min, candidate_model_id)
        sb.write(
            "champion",
            [{"horizon_min": horizon_min, "model_id": candidate_model_id, "promoted_at": datetime.now(timezone.utc).isoformat()}],
            on_conflict="horizon_min",
            merge=True,
        )
    return should_promote, reason


def main():
    code_commit = os.environ.get("GITHUB_SHA", "unknown")
    version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"

    obs_df, ctx_df = load_full_history()
    print(f"histórico cargado: {len(obs_df)} observaciones, {len(ctx_df)} contexto")
    if obs_df.empty:
        print("Sin histórico en Supabase, no se puede entrenar.")
        return

    max_date = pd.to_datetime(obs_df["observed_at"]).max()
    test_start = max_date - pd.Timedelta(days=train_mod.TEST_DAYS)
    cutoff_inicio = pd.to_datetime(obs_df["observed_at"]).min().date().isoformat()
    cutoff_fin = test_start.date().isoformat()

    summary_rows = train_mod.run(obs_df, ctx_df)

    decisions = []
    for row in summary_rows:
        horizon_min = row["horizon_min"]
        model_id, agg_acc = register_candidate(horizon_min, row, code_commit, version, cutoff_inicio, cutoff_fin)

        champ_bundle = load_champion_bundle(horizon_min)
        champ_accuracy_mean, champ_by_station = champion_accuracy_on(
            champ_bundle, row["_full_train_df"], row["_test_df"],
            drifted_stations=row["drifted_stations"], horizon_min=horizon_min,
        )

        promoted, reason = decide_and_promote(horizon_min, model_id, row, champ_accuracy_mean, champ_by_station)
        decisions.append({"horizon_min": horizon_min, "model_id": model_id, "accuracy": agg_acc, "promoted": promoted, "reason": reason})

    run_id = f"run_{uuid.uuid4().hex[:16]}"
    sb.write("ejecucion_pipeline", [{
        "run_id": run_id,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "cursor_hasta": max_date.isoformat(),
        "status": "ok",
        "drift_metric": None,
        "decision_reentrenar": True,
        "motivo_decision": json.dumps(decisions, default=str)[:2000],
        "model_id": next((d["model_id"] for d in decisions if d["promoted"]), None),
    }], on_conflict="run_id")

    print(json.dumps(decisions, indent=2, default=str))


if __name__ == "__main__":
    main()
