"""Backtest puntual de solo lectura — prueba si variantes de
hiperparámetros del GBM (más regularizadas, para estaciones volátiles
como 05100) ganan en validación para alguna estación, ANTES de tocar
train.py en producción. Usa exactamente el mismo split train.py
Paso 1 (fit_train_df -> val_df, nunca toca test_df) — ninguna fuga.

No escribe nada, no sube ningún modelo. Se corre desde
.github/workflows/backtest-hparams.yml."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import supabase_client as sb
from src import train as train_mod
from src.train import FEATURE_COLS, HORIZONS_MIN, N_ENSEMBLE, TEST_DAYS, VALIDATION_DAYS
from src.features import build_feature_frame, shift_target_for_horizon

# Variantes a probar contra el gbm_candidate actual (max_depth=8,
# min_samples_leaf=30, learning_rate=0.05 — ver gbm_candidate en
# train.py). Hipótesis: una estación con swings violentos (05100:
# 30->400 en una hora) puede sobreajustarse al ruido con árboles tan
# profundos — más regularización podría generalizar mejor ahí, aunque
# pierda algo en estaciones tranquilas (por eso se mide por estación,
# no en promedio — la selección sería por estación, igual que ya pasa
# con el tipo de candidato).
VARIANTS = {
    "shallow": dict(max_depth=4, min_samples_leaf=60, learning_rate=0.05),
    "shallow_slow": dict(max_depth=4, min_samples_leaf=50, learning_rate=0.03),
    "medium": dict(max_depth=6, min_samples_leaf=40, learning_rate=0.05),
}


def _train_variant(train_df, test_df, station_categories, hparams):
    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()
    X_train["station_id"] = pd.Categorical(X_train["station_id"], categories=station_categories)
    X_test["station_id"] = pd.Categorical(X_test["station_id"], categories=station_categories)

    preds = []
    for i in range(N_ENSEMBLE):
        m = HistGradientBoostingRegressor(
            categorical_features=["station_id"],
            loss="poisson",
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42 + i,
            **hparams,
        )
        m.fit(X_train, train_df["target_demand"])
        preds.append(m.predict(X_test))
    import numpy as np
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

    overall_wins = {name: 0 for name in VARIANTS}

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
        print(f"  baseline (actual, max_depth=8/leaf=30/lr=0.05): promedio={baseline_acc.mean():.2f}")

        results = {"baseline": baseline_acc}
        for name, hparams in VARIANTS.items():
            pred = _train_variant(fit_train_df, val_df, station_categories, hparams)
            acc = train_mod.evaluate_by_station(val_df, pred).set_index("station_id")["accuracy"]
            results[name] = acc
            print(f"  {name} ({hparams}): promedio={acc.mean():.2f}")

        print(f"  --- por estación (quién gana, margen vs. baseline) ---")
        for sid in station_categories:
            row = {name: results[name].get(sid, float("nan")) for name in results}
            best_name = max(row, key=lambda k: row[k])
            margin = row[best_name] - row["baseline"]
            flag = "  <-- GANA VARIANTE" if best_name != "baseline" and margin >= 1.0 else ""
            print(f"    {sid}: " + "  ".join(f"{k}={v:.2f}" for k, v in row.items()) + f"  -> mejor={best_name} (margen={margin:+.2f}){flag}")
            if best_name != "baseline" and margin >= 1.0:
                overall_wins[best_name] = overall_wins.get(best_name, 0) + 1

    print(f"\n=== resumen: cuántas veces (estación x horizonte, de {len(station_categories) * len(HORIZONS_MIN)}) gana cada variante por >=1pt ===")
    for name, wins in overall_wins.items():
        print(f"  {name}: {wins}")


if __name__ == "__main__":
    main()
