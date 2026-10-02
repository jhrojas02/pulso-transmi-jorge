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
from catboost import CatBoostRegressor

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
    "momentum_vs_ayer", "drift_4h_vs_24h", "short_slope",
    "rain_mm", "temperature_c", "event_intensity",
]
N_ENSEMBLE = 5  # cuántos HistGradientBoostingRegressor se promedian (bagging).
# Subido de 3 a 5 (2026-09-30): +15min quedaba a centésimas del umbral de
# promoción, un margen del orden de la varianza normal de reentrenar — más
# miembros de bagging reduce esa varianza (nunca cambia la señal real que
# el modelo aprende, solo promedia más semillas), a costa de ~67% más
# tiempo de entrenamiento del GBM. Se prueba con backtest real antes de
# confiar en que ayuda, igual que cualquier otro cambio.
CATBOOST_N_ENSEMBLE = 3  # bagging más chico que el sklearn GBM (N_ENSEMBLE=5):
# CatBoost solo se entrena para la SELECCIÓN por estación (ver
# gbm_model_type_by_station en run()), no reemplaza al sklearn GBM en todas
# partes, así que se prioriza mantener el tiempo total de entrenamiento
# razonable sobre exprimir el último punto de varianza.
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


DRIFT_RECENT_DAYS = 5  # "ahora" para la detección de quiebre: últimos 5 días
DRIFT_LOOKBACK_DAYS = 19  # "antes" para comparar: los 14 días previos a esos 5
# (día -19 a día -5 respecto al dato más reciente)
DRIFT_THRESHOLD_PCT = 20.0  # cambio de nivel medio >=20% = quiebre real, no
# ruido normal del ciclo diario/semanal (validado a mano el 2026-09-29:
# 05100 -45.7%, 07111 +29.9%, 05000 +38.9% vs. el resto de estaciones
# estables en ±1-17%, la mayoría bajo 13%)

DRIFT_FAST_RECENT_DAYS = 1  # segunda mirada, más corta, para quiebres que
# empiezan literalmente el último día — la comparación de 5 días (arriba)
# los diluye y no los detecta hasta que llevan varios días (encontrado el
# 2026-09-29 con 02300 +253.4% y 03000 -53.4% en 1 día, invisibles en la
# ventana de 5 días).
DRIFT_FAST_THRESHOLD_PCT = 50.0  # más alto que el de 5 días a propósito:
# con solo 1 día de muestra el ruido normal ya es más grande (validado:
# una estación estable mostró -22.5% en 1 día sin ser un quiebre real) —
# 50% deja fuera ese ruido pero sigue capturando saltos reales grandes.


def detect_drifted_stations(observations, as_of, recent_days=DRIFT_RECENT_DAYS,
                             lookback_days=DRIFT_LOOKBACK_DAYS, threshold_pct=DRIFT_THRESHOLD_PCT):
    """Detección automática, por estación, de un quiebre reciente de nivel
    de demanda. Combina DOS ventanas: la de `recent_days` (por defecto 5,
    ve quiebres sostenidos varios días) y una segunda más corta
    (DRIFT_FAST_RECENT_DAYS=1, con su propio umbral más alto,
    DRIFT_FAST_THRESHOLD_PCT) que ve un salto brusco de un solo día que la
    ventana de 5 días todavía diluye. Reemplaza tener que detectar esto a
    mano estación por estación (así se encontraron 05100/07111/05000 el
    2026-09-29, y 02300/03000 el 2026-09-30) — se recalcula en cada
    corrida para que una estación nueva con quiebre se trate igual sin
    tener que hardcodear su station_id. Una estación cuenta como drifted
    si CUALQUIERA de las dos ventanas la marca."""
    obs = observations.copy()
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)
    as_of = pd.Timestamp(as_of)

    def _pct_change(win_recent_days):
        recent_cut = as_of - pd.Timedelta(days=win_recent_days)
        lookback_cut = as_of - pd.Timedelta(days=lookback_days)
        recent = obs[obs["observed_at"] > recent_cut]
        older = obs[(obs["observed_at"] <= recent_cut) & (obs["observed_at"] > lookback_cut)]
        recent_mean = recent.groupby("station_id")["demand"].mean()
        older_mean = older.groupby("station_id")["demand"].mean()
        common = recent_mean.index.intersection(older_mean.index)
        return 100 * (recent_mean[common] - older_mean[common]) / older_mean[common]

    pct_slow = _pct_change(recent_days)
    pct_fast = _pct_change(DRIFT_FAST_RECENT_DAYS)

    drifted_slow = pct_slow[pct_slow.abs() >= threshold_pct]
    drifted_fast = pct_fast[pct_fast.abs() >= DRIFT_FAST_THRESHOLD_PCT]

    combined = drifted_slow.combine_first(drifted_fast)
    return combined.round(1).to_dict()


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


class BaggedCatBoost:
    """Igual que BaggedGBM pero para CatBoostRegressor — necesita
    station_id como string (no pd.Categorical), CatBoost maneja la
    categórica internamente vía cat_features."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        X = X.copy()
        X["station_id"] = X["station_id"].astype(str)
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds.mean(axis=1)


def gbm_candidate_catboost(train_df, test_df, station_categories):
    """Candidato alternativo al GBM de sklearn — hallazgo del 2026-10-01
    (inspirado en info PÚBLICA del leaderboard: un compañero con accuracy
    muy alto reportaba usar CatBoost). Validado con backtest real contra
    la MISMA ventana de 7 días que usa promote.py (ver test_catboost.py):
    en general es PEOR que el sklearn GBM (-0.3 a -1.5 puntos en los 4
    horizontes), pero dramáticamente MEJOR específicamente para 05100
    (+15 a +20 puntos, consistente en los 4 horizontes) y algo mejor para
    03000/09000. Por eso nunca reemplaza al GBM de sklearn por completo —
    se usa solo para la selección por estación en run() (ver
    gbm_model_type_by_station), igual que ya se hace gbm-vs-naive."""
    cat_idx = FEATURE_COLS.index("station_id")
    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()
    X_train["station_id"] = X_train["station_id"].astype(str)
    X_test["station_id"] = X_test["station_id"].astype(str)

    models = []
    for i in range(CATBOOST_N_ENSEMBLE):
        m = CatBoostRegressor(
            loss_function="Poisson",  # misma razón que loss="poisson" en gbm_candidate
            depth=8,
            iterations=600,
            learning_rate=0.05,
            cat_features=[cat_idx],
            random_seed=42 + i,
            verbose=False,
            early_stopping_rounds=20,
        )
        m.fit(X_train, train_df["target_demand"])
        models.append(m)

    model = BaggedCatBoost(models)
    y_pred = np.clip(model.predict(X_test), 0, None)
    return model, y_pred


def gbm_candidate_mae(train_df, test_df, station_categories):
    """Tercer candidato — hallazgo del 2026-10-01: el GBM de producción
    entrena con pérdida Poisson (optimiza la MEDIA esperada), pero el
    portal califica con WAPE (error absoluto, tipo L1) — eso se minimiza
    con la MEDIANA, no la media. En un régimen tan volátil, la media se
    deja arrastrar por los picos extremos; la mediana es más robusta.
    Validado con backtest real contra la MISMA ventana de 7 días que usa
    promote.py: ayuda en CASI todas las estaciones (+1 a +3.7 puntos en
    02300/05000/09122/etc, los 4 horizontes), pero empeora mucho
    específicamente en 05100 (-7.6 a -15.6) — el patrón INVERSO de
    CatBoost (que ayuda fuerte justo en 05100). Por eso tampoco reemplaza
    nada por completo: es un tercer candidato para la selección por
    estación en run() (ver gbm_model_type_by_station), junto a sklearn-
    Poisson y CatBoost-Poisson."""
    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()
    X_train["station_id"] = pd.Categorical(X_train["station_id"], categories=station_categories)
    X_test["station_id"] = pd.Categorical(X_test["station_id"], categories=station_categories)

    models = []
    for i in range(N_ENSEMBLE):
        m = HistGradientBoostingRegressor(
            categorical_features=["station_id"],
            loss="absolute_error",  # optimiza MAE/mediana en vez de Poisson/media
            max_iter=2000,
            learning_rate=0.05,
            max_depth=8,
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


class DeltaGBM:
    """Envuelve un BaggedGBM entrenado sobre el DELTA (target_demand -
    lag_1) en vez del nivel absoluto — predict(X) reconstruye el nivel
    (lag_1 + delta_predicho, clipeado a 0) para tener la MISMA interfaz
    que los demás candidatos (sklearn/catboost/sklearn_mae), así
    PerStationGBM los mezcla sin saber que este predice distinto. `X`
    siempre trae `lag_1` porque ya es parte de FEATURE_COLS."""

    def __init__(self, bagged_model):
        self.bagged_model = bagged_model

    def predict(self, X):
        delta_pred = self.bagged_model.predict(X)
        return np.clip(X["lag_1"].to_numpy(dtype=float) + delta_pred, 0, None)


def gbm_candidate_delta(train_df, test_df, station_categories):
    """Cuarto candidato — hallazgo del 2026-10-02: con saltos de demanda
    de 2-4x en un solo ciclo (ver investigación de accuracy real en
    producción, ~37-41% vs. 65-79% de backtest), el GBM tiene que
    "recordar" el nivel absoluto completo de cada estación Y aprender la
    dinámica del salto al mismo tiempo. Reentrenar sobre el DELTA
    (target_demand - lag_1) en vez del nivel deja que lag_1 cargue la
    magnitud base (ya disponible gratis, sin que el modelo tenga que
    aprenderla) y el GBM se concentra solo en la parte difícil: cuánto
    cambia. Validado con backtest real sobre la MISMA ventana de 7 días
    que usa promote.py, con datos frescos hasta 2026-09-19 19:00: gana en
    los 4 horizontes (+8.1 en +15min, +5.6 en +30min, +3.0 en +45min,
    +0.5 en +60min), con las mayores ganancias justo en las estaciones de
    ráfaga (02300/05000/05100: +10 a +16 puntos en +15min). Por eso se
    suma como candidato más a la selección por estación (ver
    gbm_model_type_by_station), nunca reemplaza al resto."""
    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()
    X_train["station_id"] = pd.Categorical(X_train["station_id"], categories=station_categories)
    X_test["station_id"] = pd.Categorical(X_test["station_id"], categories=station_categories)
    delta_train = train_df["target_demand"] - train_df["lag_1"]

    models = []
    for i in range(N_ENSEMBLE):
        m = HistGradientBoostingRegressor(
            categorical_features=["station_id"],
            loss="squared_error",  # el delta puede ser negativo, Poisson no aplica aquí
            max_iter=2000,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=30,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42 + i,
        )
        m.fit(X_train, delta_train)
        models.append(m)

    model = DeltaGBM(BaggedGBM(models))
    y_pred = model.predict(X_test)
    return model, y_pred


DEFAULT_GBM_TYPE = "sklearn"


class PerStationGBM:
    """Modelo híbrido por estación: cada estación usa el tipo de GBM que
    ganó en SU PROPIA validación — sklearn HistGradientBoostingRegressor
    (Poisson), CatBoost (Poisson), o sklearn con pérdida MAE (ver
    gbm_candidate_catboost/gbm_candidate_mae) — ver
    gbm_model_type_by_station en run(). Nunca "todo a un solo tipo": cada
    candidato gana fuerte solo en un puñado de estaciones y pierde en el
    resto (CatBoost: 05100/03000/09000; MAE: casi todas MENOS 05100), así
    que la selección es por estación, igual que naive-vs-gbm
    (winner_by_station) y los pesos de mezcla (mix_weights_by_station).

    `models_by_type` es un dict {"sklearn": modelo, "catboost": modelo,
    "sklearn_mae": modelo, ...} — generaliza a cualquier número de
    candidatos sin tocar esta clase de nuevo. Backward-compat: un
    champion viejo (2026-10-01, antes de MAE) serializado con los
    atributos `sklearn_model`/`catboost_model` en vez de
    `models_by_type` se sigue sirviendo bien — ver predict()."""

    def __init__(self, models_by_type, station_type_map):
        self.models_by_type = models_by_type
        self.station_type_map = station_type_map  # station_id (str) -> nombre del tipo

    def predict(self, X):
        X = X.reset_index(drop=True)
        station_ids = X["station_id"].astype(str)
        models_by_type = getattr(self, "models_by_type", None)
        if models_by_type is None:
            # Champion de antes de models_by_type (solo sklearn/catboost).
            models_by_type = {"sklearn": self.sklearn_model, "catboost": self.catboost_model}
        type_map = self.station_type_map
        types = station_ids.map(lambda sid: type_map.get(sid, DEFAULT_GBM_TYPE))
        preds = np.empty(len(X), dtype=float)
        for type_name, model in models_by_type.items():
            mask = (types == type_name).to_numpy()
            if mask.any():
                preds[mask] = model.predict(X.loc[mask])
        return preds


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


MAX_GBM_WEIGHT_DRIFTED = 0.5  # PROBADO Y DESCARTADO (2026-09-30) — no se usa
# (blend_weights_3way ya no recibe drifted_stations). La hipótesis era que
# capar w_gbm en estaciones con quiebre les dejaría más margen para
# naive/fast/boost, pero el backtest real mostró lo contrario: en 05100
# el GBM SÍ aportaba más que el naive/fast base aunque la estación
# estuviera en quiebre (sin_boost bajó de 46-54 a 35-40 al aplicar el
# tope), y el efecto se replicó en los 4 horizontes — los 4 candidatos
# resultantes quedaron 0.9 a 1.1 puntos POR DEBAJO del champion (antes
# +0.1 a +3.35), ninguno promovió. Queda la función con el parámetro
# opcional por si alguien quiere retomarlo con un tope distinto, pero
# 0.5 aplicado de forma pareja a todas las estaciones con quiebre es
# demasiado agresivo — no repetir sin evidencia nueva.


def blend_weights_3way(val_df, naive_pred, fast_pred, gbm_pred, grid_step=0.05,
                        drifted_stations=None, max_gbm_drifted=MAX_GBM_WEIGHT_DRIFTED):
    """Por estación, busca por grid search (barato: ~231 combinaciones con
    paso 0.05 — subido de 0.1 el 2026-09-30, +15min quedaba a centésimas
    del umbral de promoción y una grilla más fina solo puede igualar o
    mejorar lo que ya elegía la gruesa, nunca empeorarlo) los pesos
    (w_gbm, w_fast) que maximizan accuracy en
    VALIDACIÓN de la mezcla w_gbm*gbm + w_fast*fast + (1-w_gbm-w_fast)*naive
    — nunca en test. Reemplaza blend_weights_from_validation (mezcla
    naive/GBM únicamente) porque agregar naive_fast_baseline como tercer
    candidato es, de todo lo evaluado, lo único que de verdad mejora el
    accuracy durante un quiebre real de demanda (ver naive_fast_baseline);
    en una estación estable el grid search converge solo a w_fast≈0 (se
    validó explícitamente que un peso fijo de rolling_mean_4h para TODAS
    las estaciones cuesta 4-27 puntos en una estación sin quiebre, de ahí
    la necesidad de que salga de validación por estación en vez de un
    número global). `drifted_stations` es opcional: si se pasa, capa el
    w_gbm que el grid search puede elegir para esas estaciones (ver
    MAX_GBM_WEIGHT_DRIFTED)."""
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
        gbm_cap = max_gbm_drifted if (drifted_stations and sid in drifted_stations) else 1.0
        best_w_gbm, best_w_fast, best_acc = 0.0, 0.0, -np.inf
        for w_gbm in grid[grid <= gbm_cap + 1e-9]:
            for w_fast in grid[grid <= 1.0 - w_gbm + 1e-9]:
                w_naive = 1.0 - w_gbm - w_fast
                pred = w_naive * g["naive"] + w_fast * g["fast"] + w_gbm * g["gbm"]
                _, acc = wape_accuracy(g["y_true"], pred)
                if acc > best_acc:
                    best_w_gbm, best_w_fast, best_acc = float(w_gbm), float(w_fast), acc
        weights_by_station[sid] = {"gbm": best_w_gbm, "fast": best_w_fast}
    return weights_by_station


def trend_extrapolated_signal(lag_1, lag_2, horizon_steps):
    """Señal alternativa a lag_1 puro para el boost reactivo: extrapola la
    pendiente de muy corto plazo (`lag_1 - lag_2`, el cambio en los
    últimos 15min) `horizon_steps` pasos hacia adelante, en vez de asumir
    que el nivel se queda congelado en lag_1 (2026-09-30, investigando por
    qué 05100 seguía con accuracy muy negativa incluso con el boost ya
    aplicado: su demanda sigue cayendo CICLO A CICLO, no dio un solo salto
    y ya — lag_1 corrige el nivel absoluto, pero en +45/+60min ya vuelve a
    llegar tarde porque asume que el valor de hace 15min sigue vigente 3-4
    pasos después. 03000 tiene el mismo problema en los horizontes donde
    el boost actual está desactivado (ver boost_validated_stations).

    Debe pasar por la MISMA autovalidación por estación/horizonte que
    cualquier otra señal de boost (ver run(): compara con backtest real
    contra la versión sin boost) — nunca se asume que extrapolar la
    pendiente ayuda; en una estación cuyo quiebre ya se está frenando,
    podría sobrecorregir."""
    lag_1 = np.asarray(lag_1, dtype=float)
    lag_2 = np.asarray(lag_2, dtype=float)
    slope = lag_1 - lag_2
    return np.clip(lag_1 + slope * horizon_steps, 0, None)


TREND_DAMPING_PHI = 0.85  # amortigua la extrapolación en horizontes largos
# (Holt damped trend) — 2026-09-30: 02300/03000/05000/05100/06000/07111 tienen
# un ciclo diario fuerte (pico ~10:30-13:00, casi 0 de noche); durante la
# bajada/subida de ese pico la pendiente de 1h es enorme y NO sigue siendo así
# 45-60min después (el nivel se aplana, no sigue cayendo/subiendo en línea
# recta) — extrapolar esa pendiente sin amortiguar sobreestima sistemáticamente
# en +45/+60min. phi=0.85 (cada paso adicional pesa 0.85 del anterior, geométrico
# en vez de lineal) validado con backtest real sobre datos crudos de las 6
# estaciones con quiebre: gana en TODOS los horizontes >=45min de las 6
# estaciones (hasta +1.63 en 07111 +60min), neutral en +15/+30min (peor caso
# -0.35 en 02300 +15min). Barrido de phi en {1.0(sin amortiguar), 0.9, 0.85,
# 0.8, 0.7}: 0.85 fue el mejor promedio (71.16 vs. 70.79 sin amortiguar).

TREND_MAX_RELATIVE_DEVIATION = 0.5  # tope adicional sobre lag_1 — 2026-09-30:
# aun amortiguada, 03000 tiene transiciones tan extremas (ej. 116->969 en
# ~2.5h) que la extrapolación seguía sobrecorrigiendo en +45/+60min. Este
# tope (nunca alejarse de lag_1 más de un 50% en ninguna dirección) es un
# freno de emergencia independiente del amortiguamiento: no cambia nada en
# la mayoría de los ciclos (la pendiente amortiguada rara vez llega a ese
# punto), pero corta los peores sobresaltos. Barrido real de tope en
# {ninguno, 1.0, 0.7, 0.5, 0.35} sobre las 6 estaciones con quiebre: 0.5 fue
# empate o mejora en LOS 24 pares estación×horizonte (nunca empeoró
# ninguno), con la mayor ganancia justo en 03000/05100 +45/+60min (+0.48
# promedio) — el foco de esta sesión.


def trend_extrapolated_signal_smoothed(lag_1, lag_1h, horizon_steps, phi=TREND_DAMPING_PHI,
                                        max_relative_deviation=TREND_MAX_RELATIVE_DEVIATION):
    """Variante de trend_extrapolated_signal con pendiente SUAVIZADA sobre
    1h (4 pasos) en vez de un solo par de lags — 2026-09-30: en estaciones
    con caída sostenida pero muy volátil paso a paso (05100: sube y baja
    30-50 unidades de un ciclo a otro incluso en plena caída), lag_1-lag_2
    a veces da pendiente positiva por puro ruido de un solo punto, y el
    boost reacciona mal justo cuando más se necesita. (lag_1-lag_1h)/3
    promedia 3 diferencias consecutivas de 15min (telescopado), mucho más
    estable, sin perder capacidad de reaccionar dentro de la misma hora.

    La extrapolación se AMORTIGUA geométricamente (ver TREND_DAMPING_PHI):
    en vez de sumar slope*horizon_steps (line recta, sobreestima en
    horizontes largos durante picos/valles del ciclo diario), se suma
    slope*phi*(1-phi**horizon_steps)/(1-phi) — cada paso adicional pesa
    phi veces el anterior, así que la extrapolación se aplana en vez de
    seguir creciendo sin límite. Con phi=1.0 es exactamente la versión
    sin amortiguar (factor=horizon_steps).

    Además se topa a `max_relative_deviation` de lag_1 (ver
    TREND_MAX_RELATIVE_DEVIATION) — freno adicional para los casos donde
    ni el amortiguamiento alcanza a evitar el sobresalto."""
    lag_1 = np.asarray(lag_1, dtype=float)
    lag_1h = np.asarray(lag_1h, dtype=float)
    slope = (lag_1 - lag_1h) / 3.0
    if phi >= 1.0:
        factor = horizon_steps
    else:
        factor = phi * (1 - phi ** horizon_steps) / (1 - phi)
    value = lag_1 + slope * factor
    if max_relative_deviation is not None:
        value = np.clip(value, lag_1 * (1 - max_relative_deviation), lag_1 * (1 + max_relative_deviation))
    return np.clip(value, 0, None)


BOOST_BLEND_WEIGHT_TREND = 0.5  # peso del componente de tendencia vs. el
# promedio simple de los últimos 2 puntos — 2026-10-01: el profe subió el
# drift de verdad y cambió el RITMO mismo de la demanda en estaciones que
# antes eran estables (07105/10009/06111/09122 pasaron de un único pico
# diario a oscilar entre casi 0 y su pico varias veces en pocas horas).
# Durante una oscilación así, extrapolar la pendiente (trend_extrapolated_
# signal_smoothed) sobrecorrige sistemáticamente — no hay tendencia
# sostenida que extrapolar, es un vaivén — mientras que el promedio de los
# últimos 2 puntos (sin proyectar nada) sigue el vaivén de cerca. Barrido
# real con datos crudos de ambos grupos (las 4 estaciones con oscilación
# nueva Y las 6 con quiebre sostenido ya conocidas): w_trend=0.5 ganó en
# promedio en AMBOS grupos frente a los dos extremos (71.98 vs 70.23 de la
# señal pura de tendencia en las 6 conocidas; 60.42 vs 58.39 en las 4
# nuevas) — mezclar no es una concesión, es estrictamente mejor que
# cualquiera de las dos señales solas.


def blended_boost_signal(lag_1, lag_2, lag_1h, horizon_steps, w_trend=BOOST_BLEND_WEIGHT_TREND):
    """Mezcla trend_extrapolated_signal_smoothed (sigue tendencias
    sostenidas) con el promedio simple de lag_1/lag_2 (sigue oscilaciones
    rápidas sin sobrecorregir) — ver BOOST_BLEND_WEIGHT_TREND para el
    porqué y la validación. Es la señal de boost por defecto desde
    2026-10-01."""
    lag_1 = np.asarray(lag_1, dtype=float)
    lag_2 = np.asarray(lag_2, dtype=float)
    trend_signal = trend_extrapolated_signal_smoothed(lag_1, lag_1h, horizon_steps)
    short_avg = (lag_1 + lag_2) / 2.0
    return np.clip(w_trend * trend_signal + (1 - w_trend) * short_avg, 0, None)


FAST_BOOST_HORIZONS = {15, 30, 45, 60}  # los 4 horizontes: con lag_1 como
# señal del boost (ver más abajo) deja de haber horizontes "malos" — a
# diferencia del intento anterior (2026-09-29) con rolling_mean_4h, que
# solo funcionaba limpio en +15min. Backtest real 2026-09-30
# (05100/05000/07111 vs. control 02300/03000): ganancia positiva en los
# 4 horizontes para las 3 estaciones con quiebre sostenido (+4.3 a
# +10.3 puntos).
FAST_BOOST_THRESHOLD = 0.10  # desviación mínima (10%) de boost_fast vs.
# naive para activar el boost.
FAST_BOOST_K = 0.5  # valor por defecto si no hay entrada en
# FAST_BOOST_K_BY_HORIZON para el horizonte pedido.
FAST_BOOST_K_BY_HORIZON = {15: 0.5, 30: 0.3, 45: 0.1, 60: 0.05}  # k por
# horizonte en vez de un único valor global — encontrado por sweep real
# sobre estaciones de "ráfaga" (02300/06000/07111/05000/09122, TEST_DAYS=7
# completo, 2026-10-02): el boost de extrapolación de tendencia
# sobrecorrige sistemáticamente justo cuando la demanda revierte, y ese
# sobrecorrección empeora cuanto más largo es el horizonte (más tiempo
# para que la tendencia extrapolada se aleje de la realidad). k=0.5 (el
# valor global anterior) seguía siendo ~óptimo en +15min, pero en
# +30min perdía ~0.2-1.7pts vs. el pico en k≈0.3, en +45min perdía
# ~1.4pts vs. el pico en k≈0.1, y en +60min perdía ~4.5pts vs. el pico en
# k≈0.05 (el PEOR valor de todo el sweep para ese horizonte).


def compute_fast_boost(naive_pred, boost_fast_pred, is_drifted, horizon_min, threshold=None, k=None):
    """Boost adicional a w_fast, SOLO para filas de estaciones con quiebre
    de demanda ya confirmado (`is_drifted`, ver detect_drifted_stations).
    Reemplaza el boost dinámico descartado el 2026-09-28 (comparaba contra
    rolling_mean_24h y confundía cualquier hora pico con una emergencia,
    -27 puntos en 03000 estable) — este compara contra naive_pred, que ya
    es el nivel esperado PARA ESA hora/día (no un promedio plano), y además
    solo se activa en estaciones ya confirmadas con quiebre, así que nunca
    dispara por el ruido normal de una estación estable.

    `boost_fast_pred` es la ÚLTIMA observación real (lag_1), no
    rolling_mean_4h — cambiado el 2026-09-30: un promedio de 4h reacciona
    demasiado lento cuando la demanda sigue moviéndose (sube/baja de un
    ciclo al otro), sobre todo en +30/+45/+60min; lag_1 seguía siendo la
    lectura MÁS reciente posible, sin promediar, así que reacciona de
    inmediato — validado en backtest real: mejora en los 4 horizontes
    para las 3 estaciones con quiebre sostenido conocidas, y de paso
    arregla dos casos nuevos (un pico de un día y una caída pronunciada)
    donde rolling_mean_4h como señal del boost empeoraba."""
    if horizon_min not in FAST_BOOST_HORIZONS:
        return np.zeros(len(np.asarray(naive_pred)))
    threshold = FAST_BOOST_THRESHOLD if threshold is None else threshold
    k = FAST_BOOST_K_BY_HORIZON.get(horizon_min, FAST_BOOST_K) if k is None else k
    naive_pred = np.asarray(naive_pred, dtype=float)
    boost_fast_pred = np.asarray(boost_fast_pred, dtype=float)
    naive_safe = np.where(naive_pred == 0, np.nan, naive_pred)
    dev = np.nan_to_num((boost_fast_pred - naive_pred) / naive_safe, nan=0.0)
    boost = np.clip(np.abs(dev) - threshold, 0, None) * k
    return np.where(np.asarray(is_drifted), np.clip(boost, 0, 1.0), 0.0)


def hybrid_predict(test_df, naive_pred, fast_pred, gbm_pred, mix_weights_by_station,
                    drifted_stations=None, horizon_min=None, boost_fast_pred=None,
                    boost_threshold=None, boost_k=None):
    """Mezcla los 3 candidatos por estación según mix_weights_by_station
    ({"gbm": w_gbm, "fast": w_fast}, peso naive implícito = 1-w_gbm-w_fast)
    — ver blend_weights_3way.

    `drifted_stations`/`horizon_min` son opcionales: si se pasan, se suma
    el boost reactivo de compute_fast_boost por encima del peso base —
    ver esa función para el porqué y las salvaguardas. `boost_fast_pred`
    (por defecto, la extrapolación de tendencia SUAVIZADA de
    `trend_extrapolated_signal_smoothed(lag_1, lag_1h, horizon_steps)` si
    hay horizon_min y ambos lags disponibles — 2026-09-30, cambiado de la
    versión de 2 puntos tras backtest real: la suavizada ganó en LOS 24
    pares estación×horizonte con quiebre, a veces por >15 puntos (un solo
    par de lags es demasiado ruidoso en estaciones volátiles como 05100);
    si no, cae a lag_1 puro, y si tampoco hay lag_1 cae a `fast_pred`) es
    la señal que usa SOLO el boost — separada de `fast_pred`
    (rolling_mean_4h, sigue igual que siempre en la mezcla base) para no
    tocar el comportamiento ya validado de las estaciones estables, que
    nunca pasan por el boost."""
    naive_pred, fast_pred, gbm_pred = np.asarray(naive_pred), np.asarray(fast_pred), np.asarray(gbm_pred)
    if boost_fast_pred is None:
        if "lag_1" in test_df.columns and "lag_2" in test_df.columns and "lag_1h" in test_df.columns and horizon_min is not None:
            boost_fast_pred = blended_boost_signal(test_df["lag_1"], test_df["lag_2"], test_df["lag_1h"], horizon_min // 15)
        elif "lag_1" in test_df.columns and "lag_1h" in test_df.columns and horizon_min is not None:
            boost_fast_pred = trend_extrapolated_signal_smoothed(test_df["lag_1"], test_df["lag_1h"], horizon_min // 15)
        elif "lag_1" in test_df.columns and "lag_2" in test_df.columns and horizon_min is not None:
            boost_fast_pred = trend_extrapolated_signal(test_df["lag_1"], test_df["lag_2"], horizon_min // 15)
        elif "lag_1" in test_df.columns:
            boost_fast_pred = test_df["lag_1"].to_numpy()
        else:
            boost_fast_pred = fast_pred
    else:
        boost_fast_pred = np.asarray(boost_fast_pred, dtype=float)
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
    # w_boost es un peso APARTE del w_fast base: multiplica a boost_fast_pred
    # (lag_1/tendencia), nunca a fast_pred (rolling_mean_4h) — así el boost
    # no cambia el peso de rolling_mean_4h que ya eligió blend_weights_3way,
    # solo agrega una porción nueva encima.
    w_boost = np.zeros(len(sids))
    w_gbm_eff = w_gbm
    if drifted_stations and horizon_min is not None:
        is_drifted = sids.isin(drifted_stations).to_numpy()
        w_boost = compute_fast_boost(naive_pred, boost_fast_pred, is_drifted, horizon_min,
                                      threshold=boost_threshold, k=boost_k)
        # El boost puede comerle espacio a w_gbm (no solo al sobrante de
        # 1-w_gbm-w_fast) — 2026-09-30, diagnosticado con datos reales:
        # blend_weights_3way elige w_gbm en validación, ANTES de que el
        # quiebre grande ocurriera (ej. 05000 +73%: w_gbm=0.80, apenas
        # 0.20 de espacio para el boost aunque la desviación real fuera
        # mucho mayor que eso — el boost quedaba topado muy por debajo de
        # lo que la señal pedía). w_naive y w_fast NUNCA se tocan, solo
        # w_gbm cede su espacio — preferimos confiar más en la señal
        # reactiva que en un GBM entrenado mayormente con el régimen
        # viejo, pero sin tocar los componentes ya validados aparte.
        # Sigue pasando por la MISMA autovalidación por estación/horizonte
        # que cualquier otro cambio al boost (ver run()).
        w_boost = np.clip(w_boost, 0, np.clip(1.0 - w_fast, 0, None))
        room_original = np.clip(1.0 - w_gbm - w_fast, 0, None)
        excess = np.clip(w_boost - room_original, 0, None)
        w_gbm_eff = np.clip(w_gbm - excess, 0, None)
    w_naive = 1.0 - w_gbm_eff - w_fast - w_boost
    return w_naive * naive_pred + w_fast * fast_pred + w_boost * boost_fast_pred + w_gbm_eff * gbm_pred


def run(observations: pd.DataFrame, context: pd.DataFrame):
    base = build_feature_frame(observations, context)
    max_date = base["observed_at"].max()
    test_start = max_date - pd.Timedelta(days=TEST_DAYS)
    val_start = test_start - pd.Timedelta(days=VALIDATION_DAYS)
    station_categories = sorted(base["station_id"].unique().tolist())

    drifted_stations = detect_drifted_stations(observations, max_date)
    if drifted_stations:
        print(f"Quiebre de demanda detectado (auto, últimos {DRIFT_RECENT_DAYS}d vs. los "
              f"{DRIFT_LOOKBACK_DAYS - DRIFT_RECENT_DAYS}d previos): {drifted_stations}")
        print("  -> solo diagnóstico por ahora (ver NOTA en run(): acortar la ventana de "
              "entrenamiento para estas estaciones se probó y empeoró el resultado).")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    summary_rows = []

    for horizon_min in HORIZONS_MIN:
        horizon_steps = horizon_min // 15
        df_h = shift_target_for_horizon(base, horizon_steps).dropna(
            subset=[
                "lag_1", "lag_2", "lag_1h", "lag_4_96", "lag_672",
                "rolling_mean_24h", "rolling_std_24h", "rolling_mean_4h", "rolling_std_4h",
                "momentum_vs_ayer", "drift_4h_vs_24h",
            ]
        )

        fit_train_df = df_h[df_h["observed_at"] < val_start]
        val_df = df_h[(df_h["observed_at"] >= val_start) & (df_h["observed_at"] < test_start)]
        full_train_df = df_h[df_h["observed_at"] < test_start]  # train + validation
        test_df = df_h[df_h["observed_at"] >= test_start]

        # NOTA (2026-09-29): se probó recortar fit_train_df/full_train_df a
        # una ventana corta (14 días) SOLO para las estaciones con quiebre
        # detectado, dejando val/test intactos — descartado tras backtest
        # real (07111/05000/05100 vs. control 02300/03000): empeoró a las
        # 5 estaciones, incluidas las que se quería arreglar (05100 +60min:
        # 58.58 -> 54.52; 07111: 77.34 -> 75.21). El GBM ya captura el
        # nivel reciente vía lag_1/rolling_mean_4h — quitarle historial
        # solo le resta filas para aprender el patrón hora/día-de-semana,
        # sin compensar con nada. Se mantiene detect_drifted_stations
        # (diagnóstico útil, no dañino) pero no se aplica ningún recorte.

        # Paso 1 — elegir los pesos de mezcla por estación SOLO con
        # validación (fit_train_df -> predice val_df), nunca se toca
        # test_df aquí.
        val_naive_pred = naive_baseline(fit_train_df, val_df)
        val_fast_pred = naive_fast_baseline(val_df)
        _, val_gbm_pred = gbm_candidate(fit_train_df, val_df, station_categories)

        # Selección por estación entre 3 candidatos de GBM (2026-10-01): se
        # decide ACÁ, en validación, nunca asumido — mismo patrón que
        # winner_by_station (gbm-vs-naive) más abajo. CatBoost solo gana en
        # un puñado de estaciones (05100 sobre todo); sklearn-MAE (pérdida
        # absoluta, alineada con WAPE en vez de Poisson) gana en CASI todas
        # MENOS 05100 (donde pierde fuerte, -7.6 a -15.6 — media vs mediana
        # en una estación de swings extremos). Ver
        # gbm_candidate_catboost/gbm_candidate_mae/PerStationGBM.
        _, val_catboost_pred = gbm_candidate_catboost(fit_train_df, val_df, station_categories)
        _, val_mae_pred = gbm_candidate_mae(fit_train_df, val_df, station_categories)
        _, val_delta_pred = gbm_candidate_delta(fit_train_df, val_df, station_categories)
        val_gbm_acc_by_type = {
            "sklearn": evaluate_by_station(val_df, val_gbm_pred).set_index("station_id")["accuracy"],
            "catboost": evaluate_by_station(val_df, val_catboost_pred).set_index("station_id")["accuracy"],
            "sklearn_mae": evaluate_by_station(val_df, val_mae_pred).set_index("station_id")["accuracy"],
            "delta": evaluate_by_station(val_df, val_delta_pred).set_index("station_id")["accuracy"],
        }
        gbm_model_type_by_station = {
            sid: max(val_gbm_acc_by_type, key=lambda t: val_gbm_acc_by_type[t].get(sid, -1))
            for sid in station_categories
        }
        val_preds_by_type = {
            "sklearn": val_gbm_pred, "catboost": val_catboost_pred,
            "sklearn_mae": val_mae_pred, "delta": val_delta_pred,
        }
        val_station_ids = val_df["station_id"].astype(str).to_numpy()
        val_type_arr = np.array([gbm_model_type_by_station.get(sid, DEFAULT_GBM_TYPE) for sid in val_station_ids])
        val_gbm_pred = np.select(
            [val_type_arr == t for t in val_preds_by_type],
            [val_preds_by_type[t] for t in val_preds_by_type],
            default=val_gbm_pred,
        )

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

        model_sklearn, gbm_pred_sklearn = gbm_candidate(full_train_df, test_df, station_categories)
        model_catboost, gbm_pred_catboost = gbm_candidate_catboost(full_train_df, test_df, station_categories)
        model_mae, gbm_pred_mae = gbm_candidate_mae(full_train_df, test_df, station_categories)
        model_delta, gbm_pred_delta = gbm_candidate_delta(full_train_df, test_df, station_categories)
        test_preds_by_type = {
            "sklearn": gbm_pred_sklearn, "catboost": gbm_pred_catboost,
            "sklearn_mae": gbm_pred_mae, "delta": gbm_pred_delta,
        }
        test_station_ids = test_df["station_id"].astype(str).to_numpy()
        test_type_arr = np.array([gbm_model_type_by_station.get(sid, DEFAULT_GBM_TYPE) for sid in test_station_ids])
        gbm_pred = np.select(
            [test_type_arr == t for t in test_preds_by_type],
            [test_preds_by_type[t] for t in test_preds_by_type],
            default=gbm_pred_sklearn,
        )
        gbm_by_station = evaluate_by_station(test_df, gbm_pred)
        gbm_overall_wape, gbm_overall_acc = wape_accuracy(test_df["target_demand"], gbm_pred)
        for type_name in ("catboost", "sklearn_mae", "delta"):
            chosen = [sid for sid, t in gbm_model_type_by_station.items() if t == type_name]
            if chosen:
                print(f"  -> {type_name} elegido (sobre sklearn GBM Poisson) en estaciones: {chosen}")

        # Reentreno final para DESPLIEGUE (2026-10-02): todo lo de arriba
        # entrena con full_train_df (excluye TEST_DAYS=7 días) para poder
        # medir accuracy honesto en test_df nunca visto — correcto para
        # DECIDIR si promover, pero significa que el artefacto que se
        # guarda y sirve en vivo nunca había visto ni un ejemplo de los
        # últimos 7 días. Validado con backtest real (holdout de 24h que
        # NINGÚN modelo vio, sin fuga): entrenar incluyendo la semana más
        # reciente sube el accuracy +20 a +32 puntos en ese holdout, justo
        # durante el quiebre de régimen que empezó 2026-09-18 — el
        # champion nunca había visto ni un ejemplo del régimen actual
        # porque TEST_DAYS lo dejaba siempre afuera. Por eso se reentrena
        # UNA VEZ MÁS acá, ahora sí con df_h completo (full_train_df +
        # test_df), SOLO para el artefacto que se guarda — la comparación
        # candidato-vs-champion de arriba y la de promote.py siguen
        # usando exclusivamente el modelo entrenado con full_train_df,
        # nunca este, así que la decisión de promoción sigue siendo
        # honesta (sin fuga de test_df hacia la métrica de decisión).
        # Solo se reentrena (y se guarda) el tipo de candidato que de verdad
        # gana en al menos una estación (gbm_model_type_by_station, decidido
        # en el Paso 1) — nunca los 4 completos. Con los 4 siempre incluidos,
        # el .joblib pasó el límite de tamaño de Supabase Storage del
        # proyecto (confirmado 2026-10-02: subida real con 400/EntityTooLarge,
        # mientras que archivos de prueba de 50MB sí pasaban y de 80MB no) —
        # y entrenar tipos que ninguna estación usa es trabajo tirado de
        # todos modos, nunca se sirven en predict().
        dummy_test = df_h.iloc[:1]
        candidate_fn_by_type = {
            "sklearn": gbm_candidate, "catboost": gbm_candidate_catboost,
            "sklearn_mae": gbm_candidate_mae, "delta": gbm_candidate_delta,
        }
        used_types = set(gbm_model_type_by_station.values())
        models_by_type_deploy = {t: candidate_fn_by_type[t](df_h, dummy_test, station_categories)[0] for t in used_types}
        model = PerStationGBM(models_by_type_deploy, gbm_model_type_by_station)

        # Autovalidación del boost (2026-09-30): antes de confiar en
        # `drifted_stations` a ciegas, se mide en ESTE MISMO test si el
        # boost de verdad mejora cada estación detectada — nunca se
        # asume. Reemplaza el paso manual que se hizo hoy a mano para
        # 02300 (ayuda) y 03000 (empeora, se descartó): con esto, una
        # estación nueva con quiebre en el futuro pasa por la misma
        # prueba automáticamente, sin necesitar intervención manual.
        # `boost_validated_stations` (no `drifted_stations` crudo) es lo
        # que se guarda en el champion y lo que usa la inferencia en
        # vivo — ver register_candidate/load_champion_bundle en promote.py
        # y predict_targets en submit_current_cycle.py.
        hybrid_pred_sin_boost = hybrid_predict(test_df, naive_pred, fast_pred, gbm_pred, mix_weights_by_station)
        hybrid_acc_sin_boost = evaluate_by_station(test_df, hybrid_pred_sin_boost).set_index("station_id")["accuracy"]
        # boost_fast_pred usa por defecto la pendiente SUAVIZADA (1h, 4 pasos,
        # ver trend_extrapolated_signal_smoothed) — ganó en los 24 pares
        # estación×horizonte con quiebre en el backtest real de 2026-09-30,
        # a veces por >15 puntos, frente a la pendiente de un solo par de lags
        # (demasiado ruidosa para estaciones como 05100 en plena caída).
        #
        # boost_candidates (2026-10-01): se prueba el boost en TODAS las
        # estaciones, no solo en `drifted_stations` — el detector de
        # drift compara PROMEDIOS DIARIOS (ventanas de 1 y 5 días), así
        # que un quiebre breve dentro de un solo día (ej. 03000 cayendo
        # de 369 a 36 en ~2h la noche del 2026-09-18, sin que el
        # promedio del día se mueva lo suficiente) nunca entraba a la
        # autovalidación — se descartaba sin probarse. Esto NO relaja
        # ningún criterio: el boost sigue sin aplicarse salvo que el
        # backtest real de ESTE MISMO test lo confirme explícitamente
        # (misma comparación hybrid_acc_con_boost > hybrid_acc_sin_boost
        # de siempre), así que una estación sin quiebre real simplemente
        # sale de la prueba sin boost, igual que antes. `drifted_stations`
        # (el detector por promedio diario) sigue siendo lo único que usa
        # blend_weights_3way para topar w_gbm — no se toca ese uso.
        boost_candidates = {sid: drifted_stations.get(sid, 0.0) for sid in station_categories}
        hybrid_pred_con_boost = hybrid_predict(test_df, naive_pred, fast_pred, gbm_pred, mix_weights_by_station,
                                                drifted_stations=boost_candidates, horizon_min=horizon_min)
        hybrid_acc_con_boost = evaluate_by_station(test_df, hybrid_pred_con_boost).set_index("station_id")["accuracy"]
        boost_validated_stations = {
            sid: boost_candidates[sid] for sid in boost_candidates
            if sid in hybrid_acc_con_boost.index and hybrid_acc_con_boost[sid] > hybrid_acc_sin_boost[sid]
        }
        print(f"  -> boost validado con backtest real en este mismo test (probado en las {len(boost_candidates)} "
              f"estaciones, no solo las {len(drifted_stations)} con drift de promedio diario): "
              f"{list(boost_validated_stations)} SÍ mejoran, "
              f"{[s for s in boost_candidates if s not in boost_validated_stations]} NO (se dejan sin boost)")
        for sid in boost_candidates:
            if sid in hybrid_acc_sin_boost.index and sid in hybrid_acc_con_boost.index:
                w = mix_weights_by_station.get(sid, {"gbm": 0.0, "fast": 0.0})
                room = max(0.0, 1.0 - w["gbm"] - w["fast"])
                delta = hybrid_acc_con_boost[sid] - hybrid_acc_sin_boost[sid]
                if abs(delta) >= 0.5:  # solo ruido suprimido del log, no del resultado
                    print(f"     {sid}: sin_boost={hybrid_acc_sin_boost[sid]:.2f}  "
                          f"con_boost={hybrid_acc_con_boost[sid]:.2f} ({delta:+.2f})  "
                          f"w_gbm={w['gbm']:.2f} w_fast={w['fast']:.2f} espacio_boost={room:.2f}"
                          f"{'  [drift de promedio diario]' if sid in drifted_stations else '  [solo por autovalidación]'}")

        hybrid_pred = hybrid_predict(test_df, naive_pred, fast_pred, gbm_pred, mix_weights_by_station,
                                      drifted_stations=boost_validated_stations, horizon_min=horizon_min)
        hybrid_by_station = evaluate_by_station(test_df, hybrid_pred)
        hybrid_overall_wape, hybrid_overall_acc = wape_accuracy(test_df["target_demand"], hybrid_pred)

        # Se guarda el modelo JUNTO con el orden de categorías de
        # station_id que vio en entrenamiento: HistGradientBoostingRegressor
        # codifica la categórica por posición, no por el string, así que
        # predict.py debe reconstruir exactamente el mismo orden o las
        # predicciones quedarían mal asignadas sin ningún error visible.
        # compress=3 (2026-10-02): el proyecto de Supabase Storage tiene un
        # límite de tamaño de archivo entre 50-80MB — podar los tipos de GBM
        # no usados (ver arriba) no alcanzó para +45min, que sigue usando
        # varios tipos a la vez (confirmado: EntityTooLarge real en
        # producción). La compresión de joblib es transparente (sin cambiar
        # el modelo ni requerir tocar predict.py — joblib.load detecta el
        # formato solo) y los árboles de GBM comprimen muy bien por su
        # redundancia estructural; 3 es un balance razonable entre tamaño y
        # velocidad de guardado/carga (no se necesita el máximo de 9 acá).
        model_path = ARTIFACTS_DIR / f"gbm_h{horizon_min}.joblib"
        joblib.dump({"model": model, "station_categories": station_categories}, model_path, compress=3)

        summary_rows.append({
            "horizon_min": horizon_min,
            "n_train": len(full_train_df),
            "n_test": len(test_df),
            "drifted_stations": drifted_stations,
            "boost_validated_stations": list(boost_validated_stations),
            "winner_by_station": winner_by_station,
            "mix_weights_by_station": mix_weights_by_station,
            "gbm_model_type_by_station": gbm_model_type_by_station,
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
