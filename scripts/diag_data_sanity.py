"""Diagnóstico puntual de solo lectura — sanidad de los datos crudos:
¿observacion está fresco y completo?, ¿hay duplicados?, ¿qué pinta
tiene la serie reciente de 05100 (la estación que reventó el último
candidato)? Se corre desde .github/workflows/diag-data-sanity.yml.
No escribe nada."""

import pandas as pd

from src import supabase_client as sb

STATION_TO_INSPECT = "05100"


def main():
    obs = sb.select_all("observacion", select="station_id,observed_at,demand", order="observed_at.asc,station_id.asc")
    df = pd.DataFrame(obs)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)

    print(f"total filas observacion: {len(df)}")
    print(f"rango: {df['observed_at'].min()} -> {df['observed_at'].max()}")

    print("\n=== filas por estación (debería ser ~igual entre todas) ===")
    print(df.groupby("station_id").size().to_string())

    dup = df.duplicated(subset=["station_id", "observed_at"], keep=False)
    print(f"\n=== duplicados exactos (station_id, observed_at): {dup.sum()} filas ===")
    if dup.sum() > 0:
        print(df[dup].sort_values(["station_id", "observed_at"]).head(20).to_string())

    print(f"\n=== últimos 5 días de {STATION_TO_INSPECT} (la que reventó el candidato) ===")
    cutoff = df["observed_at"].max() - pd.Timedelta(days=5)
    st = df[(df["station_id"] == STATION_TO_INSPECT) & (df["observed_at"] >= cutoff)].sort_values("observed_at")
    print(f"n={len(st)}  media={st['demand'].mean():.1f}  min={st['demand'].min()}  max={st['demand'].max()}")

    # huecos: con datos cada 15min, un salto > 20min es un hueco real.
    gaps = st["observed_at"].diff()
    real_gaps = gaps[gaps > pd.Timedelta(minutes=20)]
    print(f"huecos (> 20min entre filas consecutivas): {len(real_gaps)}")
    if len(real_gaps) > 0:
        for idx in real_gaps.index:
            print(f"  hueco de {gaps[idx]} antes de {st.loc[idx, 'observed_at']}")

    # resumen diario para ver el nivel general día a día (sin inundar el log con cada fila de 15min)
    print(f"\n=== {STATION_TO_INSPECT}: demanda promedio por día (últimos 5 días) ===")
    daily = st.set_index("observed_at")["demand"].resample("1D").agg(["mean", "min", "max", "count"])
    print(daily.to_string())

    # últimas 24h fila por fila, para ver el swing fino que reventó al candidato
    print(f"\n=== {STATION_TO_INSPECT}: últimas 24h, fila por fila ===")
    last_24h = st[st["observed_at"] >= df["observed_at"].max() - pd.Timedelta(hours=24)]
    print(last_24h[["observed_at", "demand"]].to_string(index=False))


if __name__ == "__main__":
    main()
