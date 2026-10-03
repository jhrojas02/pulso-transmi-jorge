"""Diagnóstico puntual de solo lectura — ¿07107/02300/05100 (las que
sobre-predicen en producción) están cubiertas por el boost reactivo
del champion vigente, o se les está quedando corto? Se corre desde
.github/workflows/diag-boost-status.yml. No escribe nada."""

import pandas as pd

from src import supabase_client as sb
from src.train import detect_drifted_stations

STATIONS_TO_CHECK = ["07107", "02300", "05100"]


def main():
    champ_rows = sb.select_all("champion", order="horizon_min.asc")
    for row in champ_rows:
        modelo = sb.select_one("modelo", filters={"model_id": f"eq.{row['model_id']}"})
        if not modelo:
            continue
        fl = modelo["feature_list"]
        boost_validated = fl.get("boost_validated_stations", [])
        gbm_type_by_station = fl.get("gbm_model_type_by_station", {})
        mix_weights = fl.get("mix_weights_by_station", {})
        print(f"\n=== +{row['horizon_min']}min — model_id={row['model_id']} ===")
        print(f"  boost_validated_stations (todas): {boost_validated}")
        for sid in STATIONS_TO_CHECK:
            in_boost = sid in boost_validated
            gtype = gbm_type_by_station.get(sid, "?")
            w = mix_weights.get(sid, {})
            print(f"  {sid}: boost_validado={in_boost}  tipo_gbm={gtype}  mix_weights={w}")

    print("\n=== detect_drifted_stations (quiebre por PROMEDIO DIARIO, el detector que decide si se prueba el boost) ===")
    observations = sb.select_all("observacion", select="station_id,observed_at,demand", order="observed_at.asc,station_id.asc")
    obs_df = pd.DataFrame(observations)
    obs_df["observed_at"] = pd.to_datetime(obs_df["observed_at"], utc=True)
    max_date = obs_df["observed_at"].max()
    drifted = detect_drifted_stations(obs_df, max_date)
    print(f"max_date usado: {max_date}")
    print(f"estaciones marcadas con quiebre: {drifted}")
    for sid in STATIONS_TO_CHECK:
        print(f"  {sid}: {'SÍ marcada' if sid in drifted else 'NO marcada'} (factor={drifted.get(sid, 'n/a')})")


if __name__ == "__main__":
    main()
