"""Backtest puntual de solo lectura — prueba si una pérdida de cuantil
sesgada hacia abajo (loss="quantile", q<0.5) corrige la sobre-
predicción sistemática que se vio en producción (07107/02300/05100:
siempre predicen de más, nunca de menos — ver diagnóstico 2026-10-03).
También prueba un suavizado post-hoc simple (blend con rolling_mean_4h)
como hipótesis alternativa. Usa el mismo split de validación que
train.py Paso 1 (fit_train_df -> val_df, nunca toca test_df) — ninguna
fuga. No escribe nada, no sube modelos. Se corre desde
.github/workflows/backtest-quantile.yml."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import supabase_client as sb
from src import train as train_mod
from src.train import FEATURE_COLS, HORIZONS_MIN, N_ENSEMBLE, TEST_DAYS, VALIDATION_DAYS
from src.features import build_feature_frame, shift_target_for_horizon

# "gamma" se descarta: requiere y > 0 estricto y la demanda real puede
# ser 0 en horas de muy baja actividad. "quantile" por debajo de 0.5
# apunta deliberadamente más abajo que la mediana — hipótesis directa
# contra el patrón de sobre-predicción observado (nunca sub-predicción)
# en las estaciones que más arrastran el promedio hacia abajo.
QUANTILES = [0.35, 0.40, 0.45]
EMA_ALPHAS = [0.7, 0.85]  # peso del GBM crudo; resto va a rolling_mean_4h


def _train_quantile(train_df, test_df, station_categories, q):
    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()
    X_train["station_id"] = pd.Categorical(X_train["station_id"], categories=station_categories)
    X_test["station_id"] = pd.Categorical(X_test["station_id"], categories=station_categories)

    preds = []
    for i in range(N_ENSEMBLE):
        m = HistGradientBoostingRegressor(
            categorical_features=["station_id"],
            loss="quantile",
            quantile=q,
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
        preds.append(m.predict(X_test))
    return np.clip(np.mean(preds, axis=0), 0, None)


def main():
    observations = sb.select_all("observacion", select="station_id,observed_at,demand", order="observed_at.asc,station_id.asc")
    context = sb.select_all("contexto", order="observed_at.asc")
    obs_df = pd.DataFrame(observations)
    ctx_df = pd.DataFrame(context)

    base = build_feature_frame(obs_df, ctx_df)
    max_date = base["observed_at"].max()
    test_start = max_date - pd.Timedelta(days=TEST_DAYS)
    val_start = test_start - pd.Timedelta(days=VALIDATION_DAYS)
    station_categories = sorted(base["station_id"].unique().tolist())
    print(f"max_date={max_date}  val_start={val_start}  test_start={test_start}")

    wins = {}

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

        print(f"\n=== +{horizon_min}min — fit_train={len(fit_train_df)} val={len(val_df)} ===")

        _, baseline_pred = train_mod.gbm_candidate(fit_train_df, val_df, station_categories)
        baseline_acc = train_mod.evaluate_by_station(val_df, baseline_pred).set_index("station_id")["accuracy"]
        bias = (baseline_pred - val_df["target_demand"].to_numpy())
        print(f"  baseline (poisson, max_depth=8): promedio={baseline_acc.mean():.2f}  "
              f"sesgo_medio={bias.mean():+.2f} (positivo=sobre-predice)")

        results = {"baseline": baseline_acc}
        preds_by_name = {"baseline": baseline_pred}

        for q in QUANTILES:
            name = f"quantile_{int(q*100)}"
            pred = _train_quantile(fit_train_df, val_df, station_categories, q)
            acc = train_mod.evaluate_by_station(val_df, pred).set_index("station_id")["accuracy"]
            results[name] = acc
            preds_by_name[name] = pred
            b = (pred - val_df["target_demand"].to_numpy()).mean()
            print(f"  {name}: promedio={acc.mean():.2f}  sesgo_medio={b:+.2f}")

        rolling_4h = val_df["rolling_mean_4h"].to_numpy()
        for alpha in EMA_ALPHAS:
            name = f"ema_a{int(alpha*100)}"
            pred = np.clip(alpha * baseline_pred + (1 - alpha) * rolling_4h, 0, None)
            acc = train_mod.evaluate_by_station(val_df, pred).set_index("station_id")["accuracy"]
            results[name] = acc
            b = (pred - val_df["target_demand"].to_numpy()).mean()
            print(f"  {name}: promedio={acc.mean():.2f}  sesgo_medio={b:+.2f}")

        print(f"  --- por estación (quién gana, margen vs. baseline) ---")
        for sid in station_categories:
            row = {name: results[name].get(sid, float("nan")) for name in results}
            best_name = max(row, key=lambda k: row[k])
            margin = row[best_name] - row["baseline"]
            flag = "  <-- GANA" if best_name != "baseline" and margin >= 1.0 else ""
            print(f"    {sid}: " + "  ".join(f"{k}={v:.2f}" for k, v in row.items()) + f"  -> mejor={best_name} (margen={margin:+.2f}){flag}")
            if best_name != "baseline" and margin >= 1.0:
                wins[best_name] = wins.get(best_name, 0) + 1

    print(f"\n=== resumen: cuántas veces (estación x horizonte, de {len(station_categories) * len(HORIZONS_MIN)}) gana cada variante por >=1pt ===")
    for name in list(QUANTILES and [f"quantile_{int(q*100)}" for q in QUANTILES]) + [f"ema_a{int(a*100)}" for a in EMA_ALPHAS]:
        print(f"  {name}: {wins.get(name, 0)}")


if __name__ == "__main__":
    main()
