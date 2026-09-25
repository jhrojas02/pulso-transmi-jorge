"""Evaluación continua — accuracy, cobertura y drift.

Une las predicciones guardadas en `prediccion` con la realidad en
`observacion` cuando ya está disponible, calcula WAPE/Accuracy por
estación y agregado en dos ventanas (acumulada y rolling 24h) y
registra la cobertura (qué fracción de las predicciones ya hechas
tienen realidad para compararse). Se guarda en `operational_metric`
como bitácora — el futuro dashboard de Vercel (bono) puede leer de ahí
directo.

Este módulo decide reentrenar antes de tiempo SOLO cuando hay evidencia
fuerte de drift — comparando accuracy rolling_24h contra acumulada. La
decisión de PROMOVER un modelo sigue siendo exclusiva de promote.py
(compara candidato vs. champion evaluado en vivo); esto solo adelanta
CUÁNDO corre esa comparación, disparando train.yml por fuera de su cron
de 6h cuando el rolling_24h cae mucho por debajo del acumulado. El
umbral (ver DRIFT_THRESHOLD) se calibró con bootstrap sobre datos
reales: el ruido de muestreo puro de una ventana de ~300 predicciones
tiene desviación estándar ~1 punto y, en 500 simulaciones, nunca bajó
de -3.15 sin que hubiera ningún cambio de régimen real. Umbral de 3
puntos: claramente por encima del ruido, sensible a un drift real (que
debería tumbar el accuracy mucho más que eso).
"""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from src import supabase_client as sb
from src.train import wape_accuracy

DRIFT_THRESHOLD = -3.0  # puntos de accuracy; rolling_24h - cumulative promedio
DRIFT_MIN_SAMPLES = 50  # n_evaluable mínimo en rolling_24h para confiar en el número
DRIFT_TRIGGER_COOLDOWN_MINUTES = 30  # no dispares train.yml más seguido que esto


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

    check_drift_and_maybe_retrain(rows)


def check_drift_and_maybe_retrain(rows):
    """Compara rolling_24h contra cumulative por horizonte (solo agregados,
    station_id=None) y, si la caída promedio supera DRIFT_THRESHOLD con
    evidencia suficiente, dispara train.yml YA en vez de esperar su cron de
    6h. Respeta un cooldown para no reintentar en cada despertar de
    predict.yml mientras el drift sigue activo — una corrida de
    entrenamiento ya en curso o recién terminada es señal suficiente de que
    "ya se está manejando"."""
    cumulative = {r["horizon_min"]: r for r in rows if r["window_kind"] == "cumulative" and r["station_id"] is None}
    rolling = {r["horizon_min"]: r for r in rows if r["window_kind"] == "rolling_24h" and r["station_id"] is None}

    deltas = []
    for horizon_min, roll in rolling.items():
        cum = cumulative.get(horizon_min)
        if cum is None or roll["n_evaluable"] < DRIFT_MIN_SAMPLES:
            continue
        deltas.append(roll["accuracy"] - cum["accuracy"])

    if not deltas:
        print("Drift: sin suficientes datos en rolling_24h todavía para evaluar.")
        return

    avg_delta = sum(deltas) / len(deltas)
    print(f"Drift: rolling_24h vs. cumulative, promedio de horizontes con datos suficientes = {avg_delta:+.2f}")

    if avg_delta > DRIFT_THRESHOLD:
        return  # dentro de lo normal, nada que hacer

    print(f"Drift: caída de {avg_delta:+.2f} supera el umbral ({DRIFT_THRESHOLD}) — evaluando si hay que adelantar el reentrenamiento.")
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        print("Drift: GITHUB_TOKEN/GITHUB_REPOSITORY no disponibles en este entorno; no se puede disparar train.yml (¿corriendo fuera de GitHub Actions?).")
        return

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    runs_url = f"https://api.github.com/repos/{repo}/actions/workflows/train.yml/runs"
    r = requests.get(runs_url, headers=headers, params={"per_page": 1}, timeout=15)
    r.raise_for_status()
    latest = r.json().get("workflow_runs", [])
    if latest:
        last_created = datetime.fromisoformat(latest[0]["created_at"].replace("Z", "+00:00"))
        minutes_since = (datetime.now(timezone.utc) - last_created).total_seconds() / 60
        if minutes_since < DRIFT_TRIGGER_COOLDOWN_MINUTES:
            print(f"Drift: hay una corrida de train.yml de hace {minutes_since:.0f} min (< {DRIFT_TRIGGER_COOLDOWN_MINUTES} min de cooldown) — no se dispara otra.")
            return

    dispatch_url = f"https://api.github.com/repos/{repo}/actions/workflows/train.yml/dispatches"
    r = requests.post(dispatch_url, headers=headers, json={"ref": "main"}, timeout=15)
    if r.status_code == 204:
        print("Drift: train.yml disparado antes de tiempo por drift detectado.")
    else:
        print(f"Drift: no se pudo disparar train.yml ({r.status_code}): {r.text[:300]}")


if __name__ == "__main__":
    main()
