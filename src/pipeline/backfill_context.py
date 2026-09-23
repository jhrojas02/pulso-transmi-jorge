"""Corrección única: reemplaza en Supabase las filas de `contexto`
posteriores a REAL_CONTEXT_CUTOFF (que hasta ahora se habían llenado
repitiendo la última lectura real, literal, para siempre) por el
estimado climatológico correspondiente.

Por qué hace falta: antes de este cambio, sync.py llenaba los huecos
de contexto copiando la última fila real conocida sin variarla nunca.
Con el tiempo eso deja cientos de filas con el mismo rain_mm/
temperature_c exacto repetido, que además queda mezclado en el
histórico de entrenamiento — reentrenar sin corregir esto perpetúa el
sesgo. Se corre una sola vez (no es parte del ciclo operativo normal).
"""

import pandas as pd

from src import supabase_client as sb
from src.features import climatological_context, estimate_context_row
from src.pipeline.sync import REAL_CONTEXT_CUTOFF


def main():
    all_context = sb.select_all("contexto", order="observed_at.asc")
    df = pd.DataFrame(all_context)
    df["_ts"] = pd.to_datetime(df["observed_at"], utc=True)

    real = df[df["_ts"] <= REAL_CONTEXT_CUTOFF]
    filled = df[df["_ts"] > REAL_CONTEXT_CUTOFF]
    print(f"contexto real: {len(real)} filas, contexto a corregir (post-cutoff): {len(filled)} filas")
    if filled.empty:
        print("Nada que corregir.")
        return

    climatology = climatological_context(real.drop(columns=["_ts"]))
    corrected = [estimate_context_row(row["observed_at"], climatology) for _, row in filled.iterrows()]

    sb.write("contexto", corrected, on_conflict="observed_at", merge=True)
    print(f"Corregidas {len(corrected)} filas con estimado climatológico (merge-duplicates, sobrescribe).")


if __name__ == "__main__":
    main()
