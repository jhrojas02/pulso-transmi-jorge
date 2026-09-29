"""Entrenamiento y validación temporal — Fase 3 de la guía metodológica.

Entra: feature_vector (de features.py) por horizonte de predicción.
Sale: métricas WAPE/Accuracy por estación y agregadas, para el baseline
naive y el candidato de gradient boosting, en cada uno de los 4
horizontes oficiales (+15, +30, +45, +60 min). Guarda los modelos
entrenados (joblib) y un resumen de métricas.

Validación: partición TEMPORAL en 3 bloques (31 días train / 7 días
validación / 7 días test), nunca aleatoria — mezclar futuro y pasado
inflaría la métrica de forma artificial (el modelo "vería" el futuro
durante el entrenamiento). La validación decide, por estación, cuánto
pesa cada candidato (naive estacional, GBM, y persistencia de corto
plazo — ver blend_weights_3way) en la mezcla híbrida por grid search,
en vez de una elección dura de "gana uno solo". El test —nunca tocado
en esa decisión— da la métrica final.

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

# REVERTIDO a 7+7=14 el mismo día (2026-09-29): se probó bajar a 3+2=5 para
# que el entrenamiento alcanzara más rápido un quiebre real de demanda
# (05100), y sí ayudó a ESO — pero con solo 3 días de test (~288 filas por
# estación) la estimación de "delta promedio" que decide la promoción
# (MIN_IMPROVEMENT=0.5 en promote.py) quedó tan ruidosa que empezó a dejar
# pasar candidatos peores por pura casualidad estadística: el champion de
# 03000 (estación ESTABLE, sin ningún quiebre) decía 83-85% en su propia
# métrica de test, pero cayó a 68.5% en accuracy operacional real unas
# horas después de promovido — confirmado comparando metrica_validacion
# contra operational_metric el mismo día. Combinado con que monitor.py
# dispara train.yml cada ~30 min por drift (el propio ruido alimentaba más
# drift, que disparaba más reentrenos), esto empeoró el accuracy real en
# la mayoría de las estaciones, no solo en la que se quería arreglar. La
# ventana de 14 días es más lenta para alcanzar un quiebre nuevo, pero la
# fiabilidad de la decisión de promoción importa más que la velocidad.
TEST_DAYS = 7
VALIDATION_DAYS = 7
HORIZONS_MIN = [15, 30, 45, 60]
FEATURE_COLS = [
    "station_id", "hour", "day_of_week", "is_weekend",
    "target_hour", "target_day_of_week",
    "lag_1", "lag_2", "lag_4_96", "lag_672", "rolling_mean_24h", "rolling_std_24h",
    "rolling_mean_4h", "rolling_std_4h",
    "momentum_vs_ayer", "drift_4h_vs_24h",
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


NAIVE_HALFLIFE_DAYS = 14  # a una observación de hace 14 días le pesa la mitad
# que una de hoy; a las 4 semanas, un cuarto. Antes el naive promediaba TODO
# el histórico por igual, así que un cambio real de régimen (ej. la caída
# de demanda de madrugada en 05100, ver docs de la corrida de monitoreo)
# quedaba diluido entre semanas de datos ya obsoletos y el baseline tardaba
# muchísimo en "enterarse". Con decaimiento exponencial, las observaciones
# recientes pesan más sin descartar el histórico viejo de golpe (que sigue
# aportando en estaciones estables). Validado: no empeora el backtest en
# régimen estable (la ponderación es casi plana cuando no hay quiebre),
# y en teoría reacciona más rápido cuando sí lo hay.


def weighted_naive_tables(df, hour_col, dow_col, halflife_days=NAIVE_HALFLIFE_DAYS):
    """Tablas de lookup (por (station_id, hour_col, dow_col) y por
    station_id solo) para el baseline naive, ponderadas por recencia
    desde el `observed_at` más reciente de `df`. Función única para que
    entrenamiento (naive_baseline, con hour_col/dow_col = target_hour/
    target_day_of_week) e inferencia en vivo (submit_current_cycle.py,
    con hour_col/dow_col = hour/day_of_week del cutoff) construyan
    EXACTAMENTE la misma tabla — antes cada lado tenía su propia copia
    del cálculo (sin ponderar, en el caso de inferencia), lo que
    hubiera repetido el mismo tipo de bug que ya rompió producción una
    vez con las features del GBM (ver commit que agrega
    feature_cols_by_horizon)."""
    ref_date = df["observed_at"].max()
    age_days = (ref_date - df["observed_at"]).dt.total_seconds() / 86400
    weight = np.exp(-np.log(2) * age_days / halflife_days)

    weighted = df.assign(_w=weight, _wy=weight * df["target_demand"])
    grouped = weighted.groupby(["station_id", hour_col, dow_col])[["_w", "_wy"]].sum()
    lookup = grouped["_wy"] / grouped["_w"]

    station_grouped = weighted.groupby("station_id")[["_w", "_wy"]].sum()
    station_mean = station_grouped["_wy"] / station_grouped["_w"]
    return lookup, station_mean


def naive_baseline(train_df, test_df):
    """Promedio histórico por (station_id, target_hour, target_day_of_week)
    —la hora y día del MOMENTO QUE SE PREDICE, no del corte—, solo con
    train, ponderado por recencia (ver weighted_naive_tables). Es lo que
    un baseline estacional debería usar: "¿qué pasó otras veces a esta
    hora/día, dándole más peso a lo reciente?", sea cual sea el
    horizonte."""
    group_cols = ["station_id", "target_hour", "target_day_of_week"]
    lookup, station_mean = weighted_naive_tables(train_df, "target_hour", "target_day_of_week")

    # merge() siempre resetea el índice a 0..N-1, distinto del índice
    # original de test_df (que arrastra los huecos de los dropna previos) —
    # por eso el fallback se resuelve sobre `merged` (mismo índice que
    # `missing`), nunca sobre `test_df` directamente. Indexar test_df con
    # `missing` desalinea filas en silencio (o revienta, según el caso) en
    # cuanto el índice de test_df no es un RangeIndex contiguo desde 0 —
    # nunca lo era en la ventana de test real (arrastra los índices de
    # `base` completo), solo pasaba desapercibido porque con el histórico
    # completo casi ninguna fila cae en el fallback (bug encontrado
    # validando el blend de 3 vías con un subconjunto chico de datos).
    merged = test_df.merge(lookup.rename("y_pred"), on=group_cols, how="left")
    missing = merged["y_pred"].isna()
    if missing.any():
        fallback = merged.loc[missing, "station_id"].map(station_mean)
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
            max_depth=8,
            # Subido de 15 a 30 (2026-09-29): barrido validado con datos
            # reales (07111, 02300, 03000, 4 horizontes c/u) — 15 sigue
            # siendo el peor de todo el barrido (min_samples_leaf 15 a 70);
            # 25-40 mejora de forma consistente y sin que ninguna estación
            # empeore feo (promedio +0.05 a +0.18pts, vs. -27pts que dejó
            # el boost dinámico que sí se descartó). Por encima de 40 el
            # promedio ya empieza a caer (demasiada regularización) — 30
            # queda en el centro de la zona que mejoró en el barrido.
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


def naive_fast_baseline(df):
    """Persistencia de corto plazo: usa `rolling_mean_4h` del momento del
    corte tal cual, sin importar el horizonte, como tercer candidato de la
    mezcla (junto a naive_baseline y GBM). Reacciona a un quiebre real de
    demanda en horas, algo que naive_baseline NUNCA logra sin importar el
    halflife: con ~1 observación por (estación, hora, día-de-semana) por
    día, bajar el halflife de 14 a 3 días casi no mueve el accuracy
    (validado localmente sobre la caída real de la estación 05100,
    13-15 sep 2026: 15.44 -> 15.75 de accuracy en la ventana de la caída).
    rolling_mean_4h en cambio promedia ~16 observaciones de CUALQUIER hora
    de las últimas 4h, así que sí capta el nivel nuevo casi de inmediato
    — a costa de ser más ruidoso en estaciones sin quiebre (por eso el
    peso que se le da, ver blend_weights_3way, sale de validación por
    estación y no es fijo)."""
    return df["rolling_mean_4h"].to_numpy()


def blend_weights_3way(val_df, naive_pred, fast_pred, gbm_pred, grid_step=0.1):
    """Por estación, busca por grid search (barato: ~66 combinaciones con
    paso 0.1) los pesos (w_gbm, w_fast) que maximizan accuracy en
    VALIDACIÓN de la mezcla w_gbm*gbm + w_fast*fast + (1-w_gbm-w_fast)*naive
    — nunca en test. Reemplaza blend_weights_from_validation (mezcla
    naive/GBM únicamente) porque agregar naive_fast_baseline como tercer
    candidato es, de todo lo evaluado, lo único que de verdad mejora el
    accuracy durante un quiebre real de demanda (ver naive_fast_baseline);
    en una estación estable el grid search converge solo a w_fast≈0 (se
    validó explícitamente que un peso fijo de rolling_mean_4h para TODAS
    las estaciones cuesta 4-27 puntos en una estación sin quiebre, de ahí
    la necesidad de que salga de validación por estación en vez de un
    número global)."""
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, grid_step), 2)
    df = pd.DataFrame({
        "station_id": val_df["station_id"].to_numpy(),
        "y_true": val_df["target_demand"].to_numpy(),
        "naive": np.asarray(naive_pred),
        "fast": np.asarray(fast_pred),
        "gbm": np.asarray(gbm_pred),
    })
    weights_by_station = {}
    for sid, g in df.groupby("station_id"):
        best_w_gbm, best_w_fast, best_acc = 0.0, 0.0, -np.inf
        for w_gbm in grid:
            for w_fast in grid[grid <= 1.0 - w_gbm + 1e-9]:
                w_naive = 1.0 - w_gbm - w_fast
                pred = w_naive * g["naive"] + w_fast * g["fast"] + w_gbm * g["gbm"]
                _, acc = wape_accuracy(g["y_true"], pred)
                if acc > best_acc:
                    best_w_gbm, best_w_fast, best_acc = float(w_gbm), float(w_fast), acc
        weights_by_station[sid] = {"gbm": best_w_gbm, "fast": best_w_fast}
    return weights_by_station


def hybrid_predict(test_df, naive_pred, fast_pred, gbm_pred, mix_weights_by_station):
    """Mezcla los 3 candidatos por estación según mix_weights_by_station
    ({"gbm": w_gbm, "fast": w_fast}, peso naive implícito = 1-w_gbm-w_fast)
    — ver blend_weights_3way.

    Nota (2026-09-28): se probó y se descartó un boost dinámico que subía
    w_fast en caliente según qué tan lejos estaba rolling_mean_4h de
    rolling_mean_24h en la fila que se predice, para reaccionar sin
    esperar a que una futura ventana de validación "se entere" de un
    quiebre real (el peso base sí tiene ese rezago — ver blend_weights_3way).
    Empeoraba MUCHO en estaciones sin quiebre (validado con datos reales:
    -27 puntos en 03000, estación estable) porque rolling_mean_4h se
    desvía de rolling_mean_24h todo el tiempo por el ciclo normal de
    horas pico/valle — esa señal no distingue estacionalidad esperada de
    un quiebre real de régimen, así que confundía casi cualquier hora
    pico con una "emergencia". Para hacer esto bien haría falta comparar
    contra el nivel esperado a ESA hora (no contra el promedio de 24h
    parejo), que es justo el problema que ya resuelve naive_baseline —
    quedó pendiente como Fase 2 en vez de forzarlo a medias."""
    naive_pred, fast_pred, gbm_pred = np.asarray(naive_pred), np.asarray(fast_pred), np.asarray(gbm_pred)
    if mix_weights_by_station:
        n = len(mix_weights_by_station)
        default = {
            "gbm": sum(w["gbm"] for w in mix_weights_by_station.values()) / n,
            "fast": sum(w["fast"] for w in mix_weights_by_station.values()) / n,
        }
    else:
        default = {"gbm": 0.5, "fast": 0.0}
    sids = test_df["station_id"]
    w_gbm = sids.map(lambda s: mix_weights_by_station.get(s, default)["gbm"]).to_numpy()
    w_fast = sids.map(lambda s: mix_weights_by_station.get(s, default)["fast"]).to_numpy()
    w_naive = 1.0 - w_gbm - w_fast
    return w_naive * naive_pred + w_fast * fast_pred + w_gbm * gbm_pred


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
            subset=[
                "lag_1", "lag_2", "lag_4_96", "lag_672",
                "rolling_mean_24h", "rolling_std_24h", "rolling_mean_4h", "rolling_std_4h",
                "momentum_vs_ayer", "drift_4h_vs_24h",
            ]
        )

        fit_train_df = df_h[df_h["observed_at"] < val_start]
        val_df = df_h[(df_h["observed_at"] >= val_start) & (df_h["observed_at"] < test_start)]
        full_train_df = df_h[df_h["observed_at"] < test_start]  # train + validation
        test_df = df_h[df_h["observed_at"] >= test_start]

        # Paso 1 — elegir los pesos de mezcla por estación SOLO con
        # validación (fit_train_df -> predice val_df), nunca se toca
        # test_df aquí.
        val_naive_pred = naive_baseline(fit_train_df, val_df)
        val_fast_pred = naive_fast_baseline(val_df)
        _, val_gbm_pred = gbm_candidate(fit_train_df, val_df, station_categories)
        val_naive_acc = evaluate_by_station(val_df, val_naive_pred).set_index("station_id")["accuracy"]
        val_gbm_acc = evaluate_by_station(val_df, val_gbm_pred).set_index("station_id")["accuracy"]
        winner_by_station = {
            sid: ("gbm" if val_gbm_acc[sid] >= val_naive_acc[sid] else "naive")
            for sid in val_gbm_acc.index
        }
        mix_weights_by_station = blend_weights_3way(val_df, val_naive_pred, val_fast_pred, val_gbm_pred)

        # Paso 2 — reentrenar con train+validación y evaluar UNA sola vez
        # sobre test, ya con la selección de Paso 1 congelada.
        naive_pred = naive_baseline(full_train_df, test_df)
        naive_by_station = evaluate_by_station(test_df, naive_pred)
        naive_overall_wape, naive_overall_acc = wape_accuracy(test_df["target_demand"], naive_pred)

        fast_pred = naive_fast_baseline(test_df)
        fast_by_station = evaluate_by_station(test_df, fast_pred)

        model, gbm_pred = gbm_candidate(full_train_df, test_df, station_categories)
        gbm_by_station = evaluate_by_station(test_df, gbm_pred)
        gbm_overall_wape, gbm_overall_acc = wape_accuracy(test_df["target_demand"], gbm_pred)

        hybrid_pred = hybrid_predict(test_df, naive_pred, fast_pred, gbm_pred, mix_weights_by_station)
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
            "mix_weights_by_station": mix_weights_by_station,
            "naive_accuracy_mean_stations": naive_by_station["accuracy"].mean(),
            "fast_accuracy_mean_stations": fast_by_station["accuracy"].mean(),
            "gbm_accuracy_mean_stations": gbm_by_station["accuracy"].mean(),
            "hybrid_accuracy_mean_stations": hybrid_by_station["accuracy"].mean(),
            "naive_by_station": naive_by_station.to_dict("records"),
            "fast_by_station": fast_by_station.to_dict("records"),
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
        print(f"Fast   — accuracy promedio por estación: {fast_by_station['accuracy'].mean():.2f}  (persistencia rolling_mean_4h, ver naive_fast_baseline)")
        print(f"GBM    — accuracy promedio por estación: {gbm_by_station['accuracy'].mean():.2f}")
        n_fast_used = sum(1 for w in mix_weights_by_station.values() if w["fast"] > 0)
        print(f"Hybrid — accuracy promedio por estación: {hybrid_by_station['accuracy'].mean():.2f}  "
              f"(gana GBM en: {sum(1 for v in winner_by_station.values() if v=='gbm')}/12 estaciones, "
              f"usa algo de Fast en: {n_fast_used}/12 estaciones)")

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
