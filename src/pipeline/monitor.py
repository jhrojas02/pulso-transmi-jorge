"""Evaluación continua — accuracy, cobertura y drift.

Une las predicciones guardadas en `prediccion` con la realidad en
`observacion` cuando ya está disponible, calcula WAPE/Accuracy por
estación y agregado en dos ventanas (acumulada y rolling 24h) y
registra la cobertura (qué fracción de las predicciones ya hechas
tienen realidad para compararse). Se guarda en `operational_metric`
como bitácora — el futuro dashboard de Vercel (bono) puede leer de ahí
directo.

Este módulo NO decide reentrenar por sí solo — solo dice, con
evidencia, qué está pasando. Esa decisión vive en promote.py, que
compara candidato vs. champion cada vez que corre (manual o diario),
nunca en cada despertar del cron de inferencia.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from src import supabase_client as sb
from src.train import wape_accuracy


def realized_predictions():
    preds = sb.select_all(
        "prediccion",
        select="station_id,target_timestamp,horizonte,demanda_predicha,model_id,generated_at",
        order="prediction_id.asc",
    )
    if not preds:
        return pd.DataFrame()
    obs = sb.select_all("observacion", select="station_id,observed_at,demand", order="observed_at.asc,station_id.asc")
    pred_df = pd.DataFrame(preds)
    obs_df = pd.DataFrame(obs)
    return pred_df.merge(
        obs_df, left_on=["station_id", "target_timestamp"], right_on=["station_id", "observed_at"], how="inner"
    )


def _metric_rows(merged, window_kind, now):
    rows = []
    for horizon_idx, g in merged.groupby("horizonte"):
        horizon_min = int(horizon_idx) * 15
        for station_id, gs in g.groupby("station_id"):
            wape, acc = wape_accuracy(gs["demand"], gs["demanda_predicha"])
            rows.append({
                "metric_id": f"opmetric_{window_kind}_{horizon_min}_{station_id}_{int(now.timestamp())}",
                "computed_at": now.isoformat(),
                "window_kind": window_kind,
                "horizon_min": horizon_min,
                "station_id": station_id,
                "wape": wape,
                "accuracy": acc,
                "n_evaluable": len(gs),
            })
        wape, acc = wape_accuracy(g["demand"], g["demanda_predicha"])
        rows.append({
            "metric_id": f"opmetric_{window_kind}_{horizon_min}_agg_{int(now.timestamp())}",
            "computed_at": now.isoformat(),
            "window_kind": window_kind,
            "horizon_min": horizon_min,
            "station_id": None,
            "wape": wape,
            "accuracy": acc,
            "n_evaluable": len(g),
        })
    return rows


def main():
    merged = realized_predictions()
    if merged.empty:
        print("Sin predicciones con realidad ya observada todavía. Nada que evaluar.")
        return

    all_preds = sb.select_all("prediccion", select="horizonte", order="prediction_id.asc")
    total_by_horizon = pd.DataFrame(all_preds).groupby("horizonte").size().to_dict() if all_preds else {}

    now = datetime.now(timezone.utc)
    # El escenario corre sobre un reloj VIRTUAL (GET /v1/clock -> virtual_now),
    # muy distinto de la hora real — hoy real 2026-09-25, virtual_now cerca de
    # 2026-09-13. Comparar target_timestamp (que vive en el tiempo virtual)
    # contra el reloj real dejaba `rolling` siempre vacío: nunca hubo un solo
    # target en las "últimas 24h" reales. Usamos como referencia el target
    # más reciente que ya tiene realidad observada, que sí vive en el mismo
    # tiempo virtual que target_timestamp.
    target_ts = pd.to_datetime(merged["target_timestamp"])
    virtual_now = target_ts.max()
    since_24h = virtual_now - timedelta(hours=24)
    rolling = merged[target_ts >= since_24h]

    rows = _metric_rows(merged, "cumulative", now) + _metric_rows(rolling, "rolling_24h", now)
    for row in rows:
        horizon_idx = row["horizon_min"] // 15
        expected = total_by_horizon.get(horizon_idx, row["n_evaluable"])
        row["n_expected"] = int(expected)
        row["coverage"] = round(row["n_evaluable"] / expected, 4) if expected else 0.0

    if rows:
        sb.write("operational_metric", rows, on_conflict="metric_id")

    for r in rows:
        if r["window_kind"] == "cumulative" and r["station_id"] is None:
            print(f"+{r['horizon_min']}min cumulative: accuracy={r['accuracy']:.2f}  "
                  f"coverage={r['coverage']:.1%}  (n={r['n_evaluable']}/{r['n_expected']})")


if __name__ == "__main__":
    main()
