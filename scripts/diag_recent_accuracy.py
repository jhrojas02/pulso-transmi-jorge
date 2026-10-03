"""Diagnóstico puntual de solo lectura — desglosa la accuracy real
(prediccion vs. observacion) de los últimos ciclos resueltos, por
horizonte y por estación, para encontrar qué frena la recuperación
que se ve en el portal. Se corre desde
.github/workflows/diag-recent-accuracy.yml. No escribe nada."""

import pandas as pd

from src import supabase_client as sb
from src.train import wape_accuracy

N_CICLOS = 6  # mismo que "últimos 6 ciclos" del portal


def main():
    preds = sb.select_all(
        "prediccion",
        select="station_id,target_timestamp,horizonte,demanda_predicha,model_id,generated_at",
        order="prediction_id.asc",
    )
    obs = sb.select_all("observacion", select="station_id,observed_at,demand", order="observed_at.asc,station_id.asc")
    pred_df = pd.DataFrame(preds)
    obs_df = pd.DataFrame(obs)
    merged = pred_df.merge(
        obs_df, left_on=["station_id", "target_timestamp"], right_on=["station_id", "observed_at"], how="inner"
    )
    if merged.empty:
        print("Sin predicciones con realidad ya observada todavía.")
        return

    print(f"total predicciones con realidad: {len(merged)}")
    print(f"model_id usados: {sorted(merged['model_id'].unique())}")

    # "ciclo" = un target_timestamp distinto (todas las estaciones se
    # predicen juntas para el mismo timestamp objetivo).
    cycles_by_horizon = {}
    for horizon_idx, g in merged.groupby("horizonte"):
        horizon_min = int(horizon_idx) * 15
        cycles = sorted(g["target_timestamp"].unique())[-N_CICLOS:]
        cycles_by_horizon[horizon_min] = cycles
        g_recent = g[g["target_timestamp"].isin(cycles)]
        wape, acc = wape_accuracy(g_recent["demand"], g_recent["demanda_predicha"])
        print(f"\n=== +{horizon_min}min — últimos {len(cycles)} ciclos (n={len(g_recent)}) — accuracy={acc:.2f} wape={wape:.2f} ===")
        print(f"  model_id(s) en esta ventana: {sorted(g_recent['model_id'].unique())}")
        for station_id, gs in g_recent.groupby("station_id"):
            s_wape, s_acc = wape_accuracy(gs["demand"], gs["demanda_predicha"])
            flag = "  <-- BAJO" if s_acc < 50 else ""
            print(f"  {station_id}: accuracy={s_acc:6.2f}  wape={s_wape:6.2f}  n={len(gs)}  "
                  f"pred_media={gs['demanda_predicha'].mean():.1f}  real_media={gs['demand'].mean():.1f}{flag}")

    print("\n=== champion vigente por horizonte ===")
    champ_rows = sb.select_all("champion", order="horizon_min.asc")
    for row in champ_rows:
        modelo = sb.select_one("modelo", filters={"model_id": f"eq.{row['model_id']}"})
        acc = None
        if modelo:
            metrics = sb.select_all("metrica_validacion", filters={"model_id": f"eq.{row['model_id']}", "station_id": "is.null"})
            if metrics:
                acc = metrics[0]["accuracy"]
        print(f"+{row['horizon_min']}min: model_id={row['model_id']}  promoted_at={row['promoted_at']}  "
              f"accuracy_validacion={acc}")


if __name__ == "__main__":
    main()
