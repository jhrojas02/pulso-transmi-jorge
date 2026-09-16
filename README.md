# Pulso TransMi — proyecto de Jorge

Repositorio de trabajo para el reto **Pulso TransMi** (MLOps / Visualización 2,
2026-II, Universidad Externado de Colombia). Se basa en el
[starter kit oficial](https://github.com/uexternadojz/pulso-transmi-sdk) del curso,
que provee el cliente Python (`pulso_transmi`) para consultar la API de datos.

## El reto

Pronosticar la demanda sintética cada 15 minutos en 12 estaciones reales de
TransMilenio, con un pipeline capaz de medir desempeño, decidir y reentrenar a
medida que se liberan observaciones nuevas durante la competencia.

## Estado actual

- [x] Entorno configurado (Python 3.11, venv, SDK instalado)
- [x] Acceso al API confirmado (`/health`, `/v1/meta`)
- [x] Datos iniciales descargados y validados (sin nulos, sin duplicados,
      continuidad de 15 min completa en las 12 estaciones)
- [x] EDA inicial — ver [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)
- [ ] Definir la pregunta predictiva formal
- [ ] Baselines (naive + uno más elaborado) con validación temporal
- [ ] Pipeline de ingesta/features/entrenamiento/predicción/monitoreo (`src/`)
- [ ] Automatización con GitHub Actions

## Hallazgos del EDA

- Dataset sintético limpio: 12 estaciones × 4.320 periodos = 51.840 observaciones,
  del 26 de julio al 8 de septiembre de 2026, sin huecos ni duplicados.
- Estacionalidad hora×día-de-semana muy marcada (doble pico entre semana, caída
  en fines de semana) — la señal más fuerte del dataset.
- La estación **NQS** domina la demanda promedio; hay que evaluar el modelo por
  estación (no solo en agregado), porque el WAPE se promedia por estación.
- Clima y eventos muestran relación no lineal con la demanda, posiblemente
  confundida con la hora del día — requiere control explícito antes de usarlos
  como features.

## Cómo correr esto

Requiere Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # si se usa el SDK directamente para descargar datos
```

## Estructura esperada (en construcción)

```text
pulso-transmi-jorge/
├── notebooks/       ← exploración (EDA, experimentos)
├── src/             ← ingest.py, features.py, train.py, predict.py, monitor.py
├── tests/
├── artifacts/       ← modelos empaquetados (joblib) — no versionados
└── .github/workflows/pipeline.yml
```
