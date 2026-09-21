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
"""

import json
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src import supabase_client as sb
from src import train as train_mod

MIN_IMPROVEMENT = 0.5  # puntos de accuracy promedio mínimos para justificar promover
MAX_STATION_REGRESSION = 2.0  # ninguna estación puede empeorar más que esto vs. el champion actual


def load_full_history():
    observations = sb.select_all("observacion", select="station_id,observed_at,demand")
    context = sb.select_all("contexto")
    return pd.DataFrame(observations), pd.DataFrame(context)


def current_champion_metrics():
    champion_rows = sb.select_all("champion", order="horizon_min.asc")
    result = {}
    for row in champion_rows:
        model_id = row["model_id"]
        agg = sb.select_one("metrica_validacion", filters={"model_id": f"eq.{model_id}", "station_id": "is.null"})
        by_station_rows = sb.select_all(
            "metrica_validacion", filters={"model_id": f"eq.{model_id}", "station_id": "not.is.null"}
        )
        result[row["horizon_min"]] = {
            "model_id": model_id,
            "accuracy": agg["accuracy"] if agg else None,
            "by_station": {r["station_id"]: r["accuracy"] for r in by_station_rows},
        }
    return result


def register_candidate(horizon_min, summary_row, code_commit, version, cutoff_inicio, cutoff_fin):
    model_id = f"model_{date.today().isoformat()}_hybrid_h{horizon_min}_{version}"
    now = datetime.now(timezone.utc).isoformat()

    feature_list = {
        "features": train_mod.FEATURE_COLS,
        "winner_by_station": summary_row["winner_by_station"],
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


def decide_and_promote(horizon_min, candidate_model_id, candidate_summary, champion_info):
    candidate_by_station = {r["station_id"]: r["accuracy"] for r in candidate_summary["hybrid_by_station"]}
    candidate_mean = candidate_summary["hybrid_accuracy_mean_stations"]
    champ = champion_info.get(horizon_min)

    if champ is None or champ["accuracy"] is None:
        should_promote = True
        reason = "no había champion registrado para este horizonte"
    else:
        delta = candidate_mean - champ["accuracy"]
        worst_regression = min(
            (candidate_by_station[sid] - champ["by_station"][sid] for sid in candidate_by_station if sid in champ["by_station"]),
            default=0.0,
        )
        should_promote = delta >= MIN_IMPROVEMENT and worst_regression >= -MAX_STATION_REGRESSION
        reason = (
            f"delta promedio={delta:+.2f} (umbral +{MIN_IMPROVEMENT}), "
            f"peor caída por estación={worst_regression:+.2f} (tolerancia -{MAX_STATION_REGRESSION})"
        )

    champ_acc_str = f"{champ['accuracy']:.2f}" if champ and champ["accuracy"] is not None else "n/a"
    print(f"+{horizon_min}min: candidato={candidate_mean:.2f}  champion_actual={champ_acc_str}  "
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
    champion_info = current_champion_metrics()

    decisions = []
    for row in summary_rows:
        horizon_min = row["horizon_min"]
        model_id, agg_acc = register_candidate(horizon_min, row, code_commit, version, cutoff_inicio, cutoff_fin)
        promoted, reason = decide_and_promote(horizon_min, model_id, row, champion_info)
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
