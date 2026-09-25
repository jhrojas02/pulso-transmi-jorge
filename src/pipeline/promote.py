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
MAX_STATION_REGRESSION = 2.0  # ninguna estación puede empeorar más que esto vs. el champion actual


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
    blend_weight_by_station = feature_list.get("blend_weight_by_station")
    if blend_weight_by_station is None:
        # Champion anterior a la mezcla suave (v3): equivalente exacto de su
        # selección dura, para que siga funcionando sin reentrenar antes.
        blend_weight_by_station = {
            sid: (1.0 if winner == "gbm" else 0.0)
            for sid, winner in feature_list.get("winner_by_station", {}).items()
        }
    return {
        "model": bundle["model"],
        "station_categories": bundle["station_categories"],
        "feature_cols": feature_list["features"],
        "blend_weight_by_station": blend_weight_by_station,
    }


def champion_accuracy_on(champ_bundle, train_df, test_df):
    """Reconstruye la predicción híbrida EXACTA del champion (su modelo GBM
    congelado + sus pesos de mezcla naive/gbm por estación, también
    congelados) pero prediciendo sobre la ventana de test de HOY. Así el
    delta contra el candidato es una comparación real, no contra un número
    viejo."""
    if champ_bundle is None:
        return None, {}
    naive_pred = train_mod.naive_baseline(train_df, test_df)
    X_test = test_df[champ_bundle["feature_cols"]].copy()
    X_test["station_id"] = pd.Categorical(X_test["station_id"], categories=champ_bundle["station_categories"])
    gbm_pred = np.clip(champ_bundle["model"].predict(X_test), 0, None)
    hybrid_pred = train_mod.hybrid_predict(test_df, naive_pred, gbm_pred, champ_bundle["blend_weight_by_station"])
    by_station = train_mod.evaluate_by_station(test_df, hybrid_pred)
    overall = by_station["accuracy"].mean()
    return overall, dict(zip(by_station["station_id"], by_station["accuracy"]))


def register_candidate(horizon_min, summary_row, code_commit, version, cutoff_inicio, cutoff_fin):
    model_id = f"model_{date.today().isoformat()}_hybrid_h{horizon_min}_{version}"
    now = datetime.now(timezone.utc).isoformat()

    feature_list = {
        "features": train_mod.FEATURE_COLS,
        "winner_by_station": summary_row["winner_by_station"],
        "blend_weight_by_station": summary_row["blend_weight_by_station"],
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
        worst_regression = min(
            (candidate_by_station[sid] - champ_by_station[sid] for sid in candidate_by_station if sid in champ_by_station),
            default=0.0,
        )
        should_promote = delta >= MIN_IMPROVEMENT and worst_regression >= -MAX_STATION_REGRESSION
        reason = (
            f"delta promedio={delta:+.2f} (umbral +{MIN_IMPROVEMENT}), "
            f"peor caída por estación={worst_regression:+.2f} (tolerancia -{MAX_STATION_REGRESSION})"
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
        champ_accuracy_mean, champ_by_station = champion_accuracy_on(champ_bundle, row["_full_train_df"], row["_test_df"])

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
