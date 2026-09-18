"""Ingeniería de features — capa 2 del esquema (supabase/schema.sql).

Entra: observaciones (demanda cruda) + contexto (clima/eventos), ya
cargadas por ingest.py. Sale: un DataFrame con una fila por
(station_id, observed_at) y las columnas de `feature_vector`.

Decisiones de diseño (para que Jorge pueda defenderlas):

- `lag_1`: demanda 15 min atrás (el paso inmediatamente anterior).
- `lag_4_96`: demanda 96 pasos atrás = mismo cuarto de hora, un día
  antes. Se eligió esta ventana (en vez de, p. ej., 4 pasos = 1 hora)
  porque el EDA mostró que la estacionalidad hora×día-de-semana es la
  señal más fuerte del dataset: "qué pasó ayer a esta hora" es más
  informativo que "qué pasó hace una hora" para este patrón.
- `rolling_mean_24h` / `rolling_std_24h`: promedio y desviación móvil de
  las 96 observaciones previas (24h), EXCLUYENDO la fila actual (shift
  antes de la ventana) para no filtrar el propio valor objetivo hacia
  la feature. El std captura qué tan errática es una estación a esa
  hora — dos estaciones con la misma media pueden tener volatilidad
  muy distinta.
- `lag_672`: demanda 672 pasos atrás = mismo cuarto de hora, mismo día
  de la semana, una semana antes. Complementa a `lag_4_96` (ayer)
  agregando persistencia semana a semana (ej. si la demanda viene
  subiendo semana tras semana, `lag_4_96` no lo captura pero
  `lag_672` sí, al comparar contra el mismo día de la semana). Cuesta
  los primeros 7 días de la serie (quedan sin este valor), igual que
  `lag_4_96` cuesta el primer día.
- `lag_672` y `rolling_std_24h` NO están en `supabase/schema.sql`
  todavía (esa tabla documenta el feature_vector "oficial" v1, con
  `lag_1`, `lag_4_96` y `rolling_mean_24h` nada más). Se calculan aquí
  porque el paso que persiste feature_vector en Supabase no existe
  todavía — cuando se construya, hay que decidir si se amplía el
  esquema o se deja esta ingeniería solo en el pipeline de
  entrenamiento.
- `rain_mm`, `temperature_c`, `event_intensity`: se toman tal cual del
  contexto en el mismo `observed_at` de la fila — son observaciones
  pasadas/presentes, nunca `*_forecast` (eso sería la variable a usar
  en inferencia real para el futuro, no aquí).

Riesgo de fuga de datos: el punto más delicado es `rolling_mean_24h`.
Si no se excluye la fila actual antes de promediar, el modelo vería
información que técnicamente no tendría en el momento de predecir.
Por eso el `shift(1)` explícito antes del `rolling`.

Este módulo NO decide el horizonte de predicción (eso lo hace
train.py, desplazando el target). Aquí cada fila describe "lo que se
sabe hasta este `observed_at`", reutilizable para cualquier horizonte.
"""

import pandas as pd

STEP_MINUTES = 15
LAG_1_STEPS = 1
LAG_DAY_STEPS = 96  # 24h / 15min
LAG_WEEK_STEPS = 672  # 7 días / 15min
ROLLING_STEPS = 96  # 24h


def build_feature_frame(observations: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Construye el feature_vector a partir de observaciones y contexto.

    `observations`: columnas station_id, observed_at, demand.
    `context`: columnas observed_at, rain_mm, rain_forecast,
    temperature_c, temperature_forecast, event_intensity.
    """
    obs = observations.copy()
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)
    ctx = context.copy()
    ctx["observed_at"] = pd.to_datetime(ctx["observed_at"], utc=True)

    df = obs.merge(
        ctx[["observed_at", "rain_mm", "temperature_c", "event_intensity"]],
        on="observed_at",
        how="left",
    )
    df = df.sort_values(["station_id", "observed_at"]).reset_index(drop=True)

    local_time = df["observed_at"].dt.tz_convert("America/Bogota")
    df["hour"] = local_time.dt.hour.astype("int16")
    df["day_of_week"] = local_time.dt.dayofweek.astype("int16")  # 0=lunes
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    g = df.groupby("station_id")["demand"]
    df["lag_1"] = g.shift(LAG_1_STEPS)
    df["lag_4_96"] = g.shift(LAG_DAY_STEPS)
    df["lag_672"] = g.shift(LAG_WEEK_STEPS)
    shifted = g.shift(1)
    df["rolling_mean_24h"] = shifted.rolling(ROLLING_STEPS, min_periods=ROLLING_STEPS).mean().reset_index(level=0, drop=True)
    df["rolling_std_24h"] = shifted.rolling(ROLLING_STEPS, min_periods=ROLLING_STEPS).std().reset_index(level=0, drop=True)

    df["target_demand"] = df["demand"].astype(float)

    cols = [
        "station_id", "observed_at", "hour", "day_of_week", "is_weekend",
        "lag_1", "lag_4_96", "lag_672", "rolling_mean_24h", "rolling_std_24h",
        "rain_mm", "temperature_c", "event_intensity",
        "target_demand",
    ]
    return df[cols]


def shift_target_for_horizon(df: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    """Reasigna target_demand a la demanda `horizon_steps` adelante.

    Las columnas de features (hour, lags, clima, etc.) describen lo que
    se sabe en `observed_at`; el target pasa a ser la demanda futura en
    `observed_at + horizon_steps*15min`. Filas sin ese futuro disponible
    (cola de la serie) quedan con target NaN y se descartan.
    """
    out = df.copy()
    out["target_demand"] = out.groupby("station_id")["target_demand"].shift(-horizon_steps)
    return out.dropna(subset=["target_demand"])
