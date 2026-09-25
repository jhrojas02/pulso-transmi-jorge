"""Entrenamiento y validación temporal — Fase 3 de la guía metodológica.

Entra: feature_vector (de features.py) por horizonte de predicción.
Sale: métricas WAPE/Accuracy por estación y agregadas, para el baseline
naive y el candidato de gradient boosting, en cada uno de los 4
horizontes oficiales (+15, +30, +45, +60 min). Guarda los modelos
entrenados (joblib) y un resumen de métricas.

Validación: partición TEMPORAL en 3 bloques (31 días train / 7 días
validación / 7 días test), nunca aleatoria — mezclar futuro y pasado
inflaría la métrica de forma artificial (el modelo "vería" el futuro
durante el entrenamiento). La validación decide, por estación, si el
naive o el GBM gana ahí; el test —nunca tocado en esa decisión— da la
métrica final.

Nota sobre el early stopping del GBM: internamente separa su propio
10% de validación DEL BLOQUE DE ENTRENAMIENTO (aleatorio, no temporal)
para decidir cuándo parar de agregar árboles. No es fuga hacia el
target real (esas filas nunca se usan para ajustar hojas, solo para
medir cuándo parar), pero es una simplificación: idealmente esa
decisión también sería temporal. Se acepta porque el propósito del
early stopping es reducir sobreajuste frente a un `max_iter` fijo
elegido a mano, no maximizar accuracy a toda costa.

Riesgo de fuga de datos: el corte train/test se hace por fecha ANTES de
calcular cualquier estadístico (medias del baseline, hiperparámetros).
El baseline naive se calcula solo con datos de train; aplicarlo con
datos de test incluidos sería fuga directa de la respuesta.
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import build_feature_frame, shift_target_for_horizon

TEST_DAYS = 7
VALIDATION_DAYS = 7
HORIZONS_MIN = [15, 30, 45, 60]
FEATURE_COLS = [
    "station_id", "hour", "day_of_week", "is_weekend",
    "target_hour", "target_day_of_week",
    "lag_1", "lag_2", "lag_4_96", "lag_672", "rolling_mean_24h", "rolling_std_24h",
    "momentum_vs_ayer",
    "rain_mm", "temperature_c", "event_intensity",
]
N_ENSEMBLE = 3  # cuántos HistGradientBoostingRegressor se promedian (bagging)
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


class BaggedGBM:
    """Promedio de varios HistGradientBoostingRegressor con distinto
    random_state. Mismo modelo base, mismas features — solo reduce la
    varianza de la predicción (cada árbol individual depende un poco
    del orden aleatorio en que exploró los splits); no es una forma de
    "hacer trampa" con datos adicionales, cada sub-modelo ve
    exactamente el mismo train_df que uno solo vería."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds.mean(axis=1)


def wape_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    wape = np.abs(y_true - y_pred).sum() / y_true.sum()
    accuracy = 100 * max(0.0, 1 - wape)
    return wape, accuracy


def evaluate_by_station(test_df, y_pred):
    out = test_df[["station_id", "target_demand"]].copy()
    out["y_pred"] = y_pred
    rows = []
    for station_id, g in out.groupby("station_id"):
        wape, acc = wape_accuracy(g["target_demand"], g["y_pred"])
        rows.append({"station_id": station_id, "wape": wape, "accuracy": acc})
    return pd.DataFrame(rows)


def naive_baseline(train_df, test_df):
    """Promedio histórico por (station_id, target_hour, target_day_of_week)
    —la hora y día del MOMENTO QUE SE PREDICE, no del corte—, solo con
    train. Es lo que un baseline estacional debería usar: "¿qué pasó
    otras veces a esta hora/día?", sea cual sea el horizonte."""
    group_cols = ["station_id", "target_hour", "target_day_of_week"]
    lookup = train_df.groupby(group_cols)["target_demand"].mean().rename("y_pred")
    station_mean = train_df.groupby("station_id")["target_demand"].mean().rename("y_pred")

    merged = test_df.merge(lookup, on=group_cols, how="left")
    missing = merged["y_pred"].isna()
    if missing.any():
        fallback = test_df.loc[missing, "station_id"].map(station_mean)
        merged.loc[missing, "y_pred"] = fallback.values
    return merged["y_pred"].to_numpy()


def gbm_candidate(train_df, test_df, station_categories):
    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()
    X_train["station_id"] = pd.Categorical(X_train["station_id"], categories=station_categories)
    X_test["station_id"] = pd.Categorical(X_test["station_id"], categories=station_categories)

    models = []
    for i in range(N_ENSEMBLE):
        m = HistGradientBoostingRegressor(
            categorical_features=["station_id"],
            loss="poisson",  # la demanda es un conteo no-negativo, no un error gaussiano
            max_iter=2000,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=30,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42 + i,
        )
        m.fit(X_train, train_df["target_demand"])
        models.append(m)

    model = BaggedGBM(models)
    y_pred = np.clip(model.predict(X_test), 0, None)
    return model, y_pred


def hybrid_predict(test_df, naive_pred, gbm_pred, winner_by_station):
    """Aplica, fila a fila, el modelo que ganó esa estación en validación."""
    naive_pred = np.asarray(naive_pred)
    gbm_pred = np.asarray(gbm_pred)
    use_gbm = test_df["station_id"].map(winner_by_station).eq("gbm").to_numpy()
    return np.where(use_gbm, gbm_pred, naive_pred)


def run(observations: pd.DataFrame, context: pd.DataFrame):
    base = build_feature_frame(observations, context)
    max_date = base["observed_at"].max()
    test_start = max_date - pd.Timedelta(days=TEST_DAYS)
    val_start = test_start - pd.Timedelta(days=VALIDATION_DAYS)
    station_categories = sorted(base["station_id"].unique().tolist())

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    summary_rows = []

    for horizon_min in HORIZONS_MIN:
        horizon_steps = horizon_min // 15
        df_h = shift_target_for_horizon(base, horizon_steps).dropna(
            subset=["lag_1", "lag_2", "lag_4_96", "lag_672", "rolling_mean_24h", "rolling_std_24h", "momentum_vs_ayer"]
        )

        fit_train_df = df_h[df_h["observed_at"] < val_start]
        val_df = df_h[(df_h["observed_at"] >= val_start) & (df_h["observed_at"] < test_start)]
        full_train_df = df_h[df_h["observed_at"] < test_start]  # train + validation
        test_df = df_h[df_h["observed_at"] >= test_start]

        # Paso 1 — elegir el ganador por estación SOLO con validación
        # (fit_train_df -> predice val_df), nunca se toca test_df aquí.
        val_naive_pred = naive_baseline(fit_train_df, val_df)
        _, val_gbm_pred = gbm_candidate(fit_train_df, val_df, station_categories)
        val_naive_acc = evaluate_by_station(val_df, val_naive_pred).set_index("station_id")["accuracy"]
        val_gbm_acc = evaluate_by_station(val_df, val_gbm_pred).set_index("station_id")["accuracy"]
        winner_by_station = {
            sid: ("gbm" if val_gbm_acc[sid] >= val_naive_acc[sid] else "naive")
            for sid in val_gbm_acc.index
        }

        # Paso 2 — reentrenar con train+validación y evaluar UNA sola vez
        # sobre test, ya con la selección de Paso 1 congelada.
        naive_pred = naive_baseline(full_train_df, test_df)
        naive_by_station = evaluate_by_station(test_df, naive_pred)
        naive_overall_wape, naive_overall_acc = wape_accuracy(test_df["target_demand"], naive_pred)

        model, gbm_pred = gbm_candidate(full_train_df, test_df, station_categories)
        gbm_by_station = evaluate_by_station(test_df, gbm_pred)
        gbm_overall_wape, gbm_overall_acc = wape_accuracy(test_df["target_demand"], gbm_pred)

        hybrid_pred = hybrid_predict(test_df, naive_pred, gbm_pred, winner_by_station)
        hybrid_by_station = evaluate_by_station(test_df, hybrid_pred)
        hybrid_overall_wape, hybrid_overall_acc = wape_accuracy(test_df["target_demand"], hybrid_pred)

        # Se guarda el modelo JUNTO con el orden de categorías de
        # station_id que vio en entrenamiento: HistGradientBoostingRegressor
        # codifica la categórica por posición, no por el string, así que
        # predict.py debe reconstruir exactamente el mismo orden o las
        # predicciones quedarían mal asignadas sin ningún error visible.
        model_path = ARTIFACTS_DIR / f"gbm_h{horizon_min}.joblib"
        joblib.dump({"model": model, "station_categories": station_categories}, model_path)

        summary_rows.append({
            "horizon_min": horizon_min,
            "n_train": len(full_train_df),
            "n_test": len(test_df),
            "winner_by_station": winner_by_station,
            "naive_accuracy_mean_stations": naive_by_station["accuracy"].mean(),
            "gbm_accuracy_mean_stations": gbm_by_station["accuracy"].mean(),
            "hybrid_accuracy_mean_stations": hybrid_by_station["accuracy"].mean(),
            "naive_by_station": naive_by_station.to_dict("records"),
            "gbm_by_station": gbm_by_station.to_dict("records"),
            "hybrid_by_station": hybrid_by_station.to_dict("records"),
            # No serializables (DataFrames) — para que promote.py pueda evaluar
            # el champion vigente en ESTA MISMA ventana de test, en vez de
            # comparar contra sus métricas registradas de cuando se entrenó
            # (que quedan obsoletas apenas los datos cambian, y mucho más si
            # hay drift). Se filtran antes de escribir el JSON de resumen.
            "_full_train_df": full_train_df,
            "_test_df": test_df,
        })

        print(f"\n=== Horizonte +{horizon_min} min ===")
        print(f"train+val={len(full_train_df)} filas, test={len(test_df)} filas (val usada solo para elegir modelo por estación)")
        print(f"Naive  — accuracy promedio por estación: {naive_by_station['accuracy'].mean():.2f}")
        print(f"GBM    — accuracy promedio por estación: {gbm_by_station['accuracy'].mean():.2f}")
        print(f"Hybrid — accuracy promedio por estación: {hybrid_by_station['accuracy'].mean():.2f}  (gana GBM en: {sum(1 for v in winner_by_station.values() if v=='gbm')}/12 estaciones)")

    summary_path = ARTIFACTS_DIR / "training_summary.json"
    serializable_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in summary_rows]
    summary_path.write_text(json.dumps(serializable_rows, indent=2, default=str))
    print(f"\nResumen guardado en {summary_path}")
    return summary_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", default="observations.json")
    parser.add_argument("--context", default="context.json")
    args = parser.parse_args()

    obs = pd.DataFrame(json.load(open(args.observations)))
    ctx = pd.DataFrame(json.load(open(args.context)))
    run(obs, ctx)
