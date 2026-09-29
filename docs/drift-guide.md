# Guía: cómo afrontar drift de demanda en pulso-transmi-jorge

Contexto completo, no un resumen — pensado para que otra persona (o una
IA) retome el trabajo sin haber visto la conversación donde se hizo.
Repo: `jhrojas02/pulso-transmi-jorge`, rama `main`.

## 0. Qué es este proyecto

Reto estudiantil de forecasting de demanda de TransMilenio: 12 estaciones,
4 horizontes de predicción (+15/+30/+45/+60 min), leaderboard oficial
contra otros estudiantes. Datos **sintéticos** con seed conocido
(`generator_seed: 20260916`, `demand_is_synthetic: true` en `/v1/meta`).

**El dataset corre en un reloj virtual propio, no en tiempo real.**
Confirmado contra `GET /v1/clock`: devuelve `virtual_now` (el tiempo del
dataset) y `server_time` (tiempo real) por separado — `virtual_now` suele
estar varios días detrás del tiempo real. Esto es NORMAL y no significa
que los datos estén desactualizados — compara siempre contra `/v1/clock`,
nunca contra la fecha de hoy.

API: `https://pulso-transmi.72-60-245-2.sslip.io`. Endpoints útiles:
`/v1/clock`, `/v1/forecast-cycles/current`, `/v1/leaderboard?window=cumulative|rolling_24h`
(requiere `Authorization: Bearer $PULSO_API_KEY`), `/v1/me`, `/v1/stations`
(metadata: nombre, corredor, lat/lon).

`.github/workflows/diag.yml` (workflow_dispatch only, nunca escribe nada)
ejecuta `scripts/diag_check.py`, que se REESCRIBE cada vez según qué se
necesite consultar. Patrón para consultar producción desde un sandbox sin
salida de red directa a la DB: editar `diag_check.py` → commit → push a
`main` → disparar `diag.yml` vía `mcp__github__actions_run_trigger` → leer
logs vía `mcp__github__get_job_logs` (usar `job_id`, no `run_id`).

`mcp__Supabase__execute_sql` (tool de Claude) da acceso de lectura directo
a la base de datos sin pasar por GitHub Actions, si esa tool está
disponible.

## 1. El problema de fondo: quiebres de demanda reales, no ruido

De tiempo en tiempo, una o más estaciones tienen un quiebre de nivel de
demanda real y abrupto (cambio sostenido de >20-40% en pocos días) —
esto se confirma mirando los datos crudos de `observacion` directamente
(demanda diaria promedio por estación), no asumiendo por el accuracy bajo.

**Cómo detectarlo:** `src/train.py:detect_drifted_stations()` lo hace
automático en cada corrida de `train.yml` — compara la demanda media de
los últimos 5 días contra los 14 anteriores, por estación, y marca las
que cambiaron más de 20%. Queda visible en los logs y en
`training_summary.json` (campo `drifted_stations`).

**Pista útil para diagnosticar la CAUSA (no solo el síntoma):** si dos
estaciones adyacentes en el mismo corredor (ver `/v1/stations`, campo
`corridor`) se mueven en direcciones opuestas al mismo tiempo, es probable
que sea redistribución de demanda (desvío de ruta, cierre parcial) más
que dos eventos independientes — vale la pena cruzar el timing exacto
antes de asumir causas distintas.

### Por qué es estructuralmente difícil de corregir rápido

`train.py` usa un split temporal train(31d)/val(7d)/test(7d)
(`TEST_DAYS=7, VALIDATION_DAYS=7`): `val_start = max_date - 14d`,
`test_start = max_date - 7d`. Cuando un quiebre empieza pocos días antes
del último dato disponible, cae DENTRO de la ventana de test, nunca antes.
Esto tiene dos consecuencias:

1. Cualquier mecanismo que dependa de calibración en `val_df` (que por
   diseño termina ANTES de que empezara el quiebre) es ciego a él — no
   importa qué tan bueno sea el candidato, se calibra sin haberlo visto.
2. La ventana de test mezcla régimen viejo y nuevo — cualquier modelo se
   mide contra un test que no es homogéneo, así que "ganarle" al régimen
   nuevo casi siempre cuesta en el régimen viejo que sigue presente en el
   mismo test.

Esto no es una limitación de una técnica en particular — es una
limitación de **cuánta evidencia post-quiebre existe**, que es poca
(pocos días) mientras el quiebre es reciente. Antes de invertir tiempo en
una corrección de código, vale la pena verificar cuántos días de evidencia
post-quiebre hay ya disponibles — con más días acumulados, técnicas que
hoy no generalizan pueden empezar a funcionar solas.

## 2. Qué mejora el accuracy — vigente hoy

- **`MAX_STATION_REGRESSION = 4.0`** + **`CHAMPION_FLOOR_ACCURACY_STATION
  = 60.0`** en `promote.py`. Con 12 estaciones, el ruido normal de
  reentrenar (semilla distinta, unos días más de datos) ya produce caídas
  de 3-4pts en la estación más volátil aunque el candidato mejore en
  promedio — un umbral más estricto bloquea prácticamente todo
  reentrenamiento. El floor evita que una estación ya rota (<60% en vivo)
  siga bloqueando candidatos que mejoran todo lo demás.
- **Blend de 3 vías** (`naive_fast_baseline` + `blend_weights_3way` +
  `hybrid_predict`, en `train.py`, propagado a `promote.py` y
  `submit_current_cycle.py`): agrega persistencia de corto plazo
  (`rolling_mean_4h` tal cual) como tercer candidato junto al naive
  estacional y al GBM, con peso elegido por validación POR ESTACIÓN
  (converge a ~0 en estaciones estables, así que nunca empeora ahí).
- **`min_samples_leaf = 30`** en `gbm_candidate()` (HistGradientBoostingRegressor)
  — validado con barrido real sobre varias estaciones y horizontes.
- **Boost reactivo en +15min para estaciones con quiebre confirmado**
  (`compute_fast_boost()` en `train.py`): suma peso extra a la
  persistencia de 4h SOLO cuando (a) `detect_drifted_stations` ya
  confirmó un quiebre sostenido en esa estación (nunca un pico puntual) y
  (b) el horizonte es +15min. Compara contra `naive_baseline` (el nivel
  esperado PARA ESA hora/día específica, no un promedio plano de 24h) —
  eso es clave: comparar contra un promedio plano confunde cualquier hora
  pico normal con una emergencia. El gate por estación hace que las
  estaciones sin quiebre queden en 0.0 de cambio siempre, por
  construcción — no solo "poco riesgo", cero.
  Conectado en los 3 puntos del pipeline que comparten la misma lógica
  (nunca duplicada):
  - `train.py`: evaluación final del candidato en test.
  - `promote.py`: evaluación EN VIVO del champion (mismo boost en ambos
    lados, si no la comparación candidato-vs-champion queda injusta).
  - `submit_current_cycle.py`: `predict_targets()`, recalculando
    `drifted_stations` en cada ciclo con las observaciones recién
    sincronizadas — reacciona sin esperar al próximo reentrenamiento.
  Vigente solo en +15min: en horizontes más largos la señal de 4h ya no
  predice bien tan lejos mientras el quiebre sigue en movimiento.

## 3. Cómo validar cualquier cambio nuevo (antes de tocar producción)

Antes de aplicar CUALQUIER cambio a `train.py`/`promote.py`: correr un
backtest real con `src.train.run()` sobre datos reales descargados de la
base de datos (no una reimplementación aparte, la función real de
producción), comparando accuracy por estación entre "antes" y "después" —
nunca solo el promedio general, y siempre incluyendo estaciones de
control que NO deberían verse afectadas por el cambio. Un cambio que
mejora el promedio pero empeora una estación estable escondida en el
promedio es exactamente el tipo de error más fácil de cometer en este
pipeline (12 estaciones, cada una con su propio comportamiento).

Reglas prácticas que ayudan a no sobreajustar:
- Cualquier técnica que dependa de calibración (`blend_weights_3way`,
  elección de ventana, etc.) hay que evaluarla sobre una ventana de test
  fija y del mismo tamaño para todas las estaciones — nunca acortar la
  ventana de validación/test para "ver" un quiebre más rápido; eso hace
  que la decisión de promoción se vuelva ruidosa y puede dejar pasar
  candidatos peores por casualidad estadística.
- Un mecanismo reactivo (que reacciona a una señal en el momento, no a
  algo calibrado de antemano) siempre necesita un gate explícito que lo
  limite a los casos donde ya hay evidencia confirmada (ej.
  `detect_drifted_stations`) — sin eso, termina reaccionando también al
  ruido normal de estaciones sin ningún problema.
- Si una idea depende de muy pocos días de evidencia (una estación, un
  evento reciente), el riesgo de que "funcione" solo por casualidad
  estadística es alto — conviene ser más exigente con la validación en
  esos casos, no menos.

## 4. Arquitectura del pipeline

- `src/features.py`: feature engineering (`build_feature_frame`,
  `shift_target_for_horizon`). Lags: 1, 2, 96 (1 día), 672 (1 semana).
  Rolling: 4h y 24h (mean+std). `momentum_vs_ayer`, `drift_4h_vs_24h`.
  Clima tomado del `observed_at` actual (nunca `*_forecast`).
- `src/train.py`: `run()` orquesta entrenamiento por horizonte, split
  temporal train/val/test (NUNCA aleatorio), 3 candidatos (naive, fast,
  GBM Poisson) + blend por validación, `N_ENSEMBLE=3` (bagging).
  `detect_drifted_stations()` y `compute_fast_boost()` viven acá (ver
  sección 2).
- `src/pipeline/promote.py`: reentrena candidato con TODO el histórico,
  compara contra champion vigente evaluado EN VIVO (nunca contra su
  métrica vieja), decide promoción con `MIN_IMPROVEMENT` +
  `MAX_STATION_REGRESSION` + `CHAMPION_FLOOR_ACCURACY_STATION`. Corre en
  `train.yml` (cron + trigger por drift desde `monitor.py`).
- `src/pipeline/submit_current_cycle.py`: sync incremental + inferencia
  del ciclo abierto + envío a `/v1/submissions`. Corre en `predict.yml`
  (cron cada 10min).
- `src/pipeline/monitor.py`: evalúa accuracy real (`operational_metric`,
  ventanas `cumulative`/`rolling_24h`), dispara `train.yml` antes de
  tiempo si detecta drift fuerte.

## 5. Ideas pendientes / no exploradas

- Verificar cada tanto si las estaciones con quiebre mejoran solas con
  más ciclos de reentrenamiento (la ventana de val/test se desliza hacia
  adelante y va incluyendo más días del régimen nuevo con el tiempo).
- Explorar si hay ganancia adicional en las estaciones SIN quiebre —
  tienen suficiente historial para experimentar con mucho menos riesgo de
  sobreajuste.
- Relajar el candado de promoción (`MIN_IMPROVEMENT`) para reaccionar más
  rápido a quiebres futuros es una palanca real, pero a costa de más
  riesgo — requiere backtest explícito antes de aplicarlo (ver sección 3).
- No se exploró a fondo que la demanda es sintética con seed conocido —
  podría haber patrón determinístico reverse-engenieerable.

## 6. Cómo retomar

1. Leer este archivo completo primero.
2. `git log --oneline -20` en `main` para ver el estado exacto de commits.
3. Confirmar el reloj virtual actual: `curl .../v1/clock` — todo se ubica
   en el tiempo relativo a eso, no al calendario real.
4. Si hace falta consultar producción: reescribir `scripts/diag_check.py`
   (o usar `mcp__Supabase__execute_sql` si está disponible), commit+push,
   disparar `diag.yml`, leer logs con `job_id` (no `run_id`).
5. Antes de cualquier cambio a `train.py`/`promote.py`: backtest real
   primero (ver sección 3) — nunca aplicar a producción sin validar
   contra estaciones de control estables, aunque la idea "debería
   funcionar" en teoría.
