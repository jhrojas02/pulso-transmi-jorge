# Dashboard (bono)

Next.js (App Router). Página server-rendered que lee, en el servidor:

- `champion` + `modelo` en Supabase → versión y fecha de entrenamiento del
  modelo vigente por horizonte.
- `sync_state` → retraso del collector.
- `operational_metric` → accuracy/cobertura por estación y horizonte
  (acumulado) y la comparación acumulado vs. rolling 24h como señal de drift.
- `ejecucion_pipeline` → historial de corridas y decisiones de
  reentrenamiento (las que escribe `promote.py`).
- `GET /v1/leaderboard?window=cumulative` y `GET /v1/me` de la API oficial →
  posición, accuracy y cobertura propias.

Todas esas llamadas viven en `lib/data.ts`, que solo se importa desde
Server Components — `SUPABASE_SERVICE_KEY` y `PULSO_API_KEY` nunca se envían
al navegador. No hay ninguna variable `NEXT_PUBLIC_*` en este proyecto.

## Desarrollo local

```bash
cd dashboard
npm install
cp .env.example .env.local   # completar las llaves
npm run dev
```

## Variables de entorno (Vercel → Settings → Environment Variables)

| Variable | Tipo |
|---|---|
| `SUPABASE_URL` | Plain |
| `SUPABASE_SERVICE_KEY` | Encrypted |
| `PULSO_API_KEY` | Encrypted |
| `PULSO_API_BASE` | Plain (opcional, tiene default) |
