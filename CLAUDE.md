# Contexto crítico de la competencia — leer antes de tocar accuracy/ranking

## La métrica que de verdad cuenta NO es la que trae la API con PULSO_API_KEY

El portal web (`https://pulso-transmi.72-60-245-2.sslip.io`, login del
usuario en su navegador) muestra dos rankings:

- **"Clasificación acumulada"**: acumulada desde el **Corte 1: 25 de
  septiembre, 00:00 hora de Bogotá**. "Los ciclos anteriores no cuentan."
- **"Clasificación · últimos 6 ciclos"**: WAPE de los últimos 6 ciclos
  resueltos desde el Corte 1 — esta es la vista que el usuario mira para
  comparar posiciones, y la que de verdad importa competitivamente.

Estas vistas vienen de `/v1/portal/leaderboard` y `/v1/portal/dashboard`,
que requieren **cookie de sesión del portal** (`ptm_session`, el login del
usuario en su propio navegador) — confirmado contra `openapi.json` del
servicio. **No se pueden consultar con `PULSO_API_KEY`** (el secret que
usa `diag.yml`/`scripts/diag_check.py`).

Lo que `scripts/diag_check.py` SÍ puede consultar con la API key es
`/v1/leaderboard?window=cumulative|rolling_24h` — y ese endpoint solo
acepta esos dos valores (confirmado contra el schema). Es una métrica
DISTINTA a la del portal: no respeta el Corte 1 y es una ventana de 24h
en vez de "últimos 6 ciclos". **No usar `rolling_24h`/`cumulative` de la
API como proxy del ranking real** — hablar con el usuario para que
comparta un screenshot del portal cuando se necesite el número real.

## Snapshot real más reciente (pedir uno nuevo si ha pasado tiempo)

Screenshot del usuario, **"últimos 6 ciclos"**, ~2026-10-01 14:10 hora
del teléfono del usuario (NO es el reloj simulado de los datos, que va
por 2026-09-19 en esa misma fecha):

| # | Estudiante | Accuracy |
|---|---|---|
| 1 | John Alejandro Bernal Guaman | 92.5% |
| 2 | Mateo Hoyos Cárdenas (usa CatBoost) | 90.5% |
| 3 | Isaias Cespedes Novoa | 84.4% |
| 18 | **Jorge Horacio Rojas Criollo (nosotros)** | **32.4%** |

Nota: el `/v1/leaderboard` vía API key, en ese mismo momento aproximado,
daba `cumulative=68.83` y `rolling_24h=34.93` para nosotros — el
`rolling_24h` (34.93) por coincidencia quedó cerca del "últimos 6 ciclos"
real del portal (32.4%), pero esto NO está garantizado en general (son
cálculos distintos) — siempre preferir el número del portal si el usuario
lo comparte.

## Otros hallazgos de esta sesión que valen la pena recordar

- El dataset es **histórico/simulado**, no tiempo real: `observacion`
  llega hasta ~2026-09-19 (avanza con cada sync, pero muy por detrás del
  reloj real). No usar `now()` de Postgres para filtrar ventanas de
  tiempo contra estos datos — usar `max(observed_at)` como referencia de
  "ahora".
- El régimen de demanda actual es **volátil, no periódico** — se probó
  una hipótesis de ciclo de ~4h (05100/09000 en antifase) y SE DESCARTÓ:
  se veía bien en una ventana de 30h pero perdía por 13-74 puntos en
  ventanas de 60h/72h. Cualquier señal nueva debe validarse contra
  ventanas de varios días, nunca solo las últimas 24-30h (ya pasó una vez
  con un GBM ponderado por recencia que hubo que revertir en producción).
- CatBoost (vía `gbm_candidate_catboost` + `PerStationGBM` en
  `src/train.py`) gana fuerte solo en 05100 (y algo en 03000/09000),
  pierde en el resto — selección por estación, nunca reemplazo total.
- El boost reactivo (`boost_candidates` en `run()`) ahora se autovalida
  en las 12 estaciones, no solo en las marcadas por
  `detect_drifted_stations` (ese detector usa promedios DIARIOS y se le
  escapan quiebres de 1-2h dentro de un mismo día, como el de 03000 el
  2026-09-18).
- `src/http_retry.py` reintenta (3 intentos, 2s/5s) las llamadas a la API
  del profesor — un timeout transitorio de su servidor no debe tirar el
  ciclo completo de `predict.yml`.
