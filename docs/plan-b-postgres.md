# Plan B: migrar de Supabase a Postgres genérico (Neon, Railway, etc.)

Preparado por si Supabase restringe el proyecto (402 / read-only) por la
política de Fair Use de Cached Egress, antes de que se resuelva el ticket
de soporte o llegue el próximo reseteo de ciclo. **No está activado** —
el pipeline sigue usando Supabase normalmente hasta que se siga esta guía.

## Qué cambia

- La base de datos deja de ser Supabase y pasa a ser cualquier Postgres
  estándar (Neon tiene un free tier generoso y sin este tipo de política
  agresiva de egress cacheado — es la opción recomendada, pero cualquier
  Postgres con connection string sirve).
- Los modelos entrenados (`.joblib`, 7-9MB c/u) se guardan como `bytea`
  directo en una tabla (`model_blob`) en vez de un bucket de Storage —
  Neon/Railway no ofrecen storage de objetos, así que esto evita depender
  de un segundo servicio.
- **Ningún script del pipeline cambia.** `supabase_client.py` detecta
  automáticamente si `DATABASE_URL` está configurada y, si es así,
  redirige todo (`select_all`, `write`, `storage_download`, etc.) hacia
  `pg_client.py` — que habla el mismo "protocolo" interno pero contra
  Postgres directo. Ver el bloque al final de `src/supabase_client.py`.

## Pasos para activar

### 1. Crear la base de datos

- Ir a [neon.tech](https://neon.tech), crear cuenta gratis, crear un
  proyecto nuevo (Postgres 16+).
- Copiar el **connection string** (algo como
  `postgresql://usuario:password@ep-xxx.neon.tech/neondb?sslmode=require`).

### 2. Crear el schema

Desde tu máquina (con `psql` instalado) o desde el SQL Editor de Neon:

```bash
psql "$DATABASE_URL" -f supabase/schema.sql
psql "$DATABASE_URL" -f supabase/model_blob.sql
```

`supabase/schema.sql` ya es Postgres estándar (no tiene nada específico
de Supabase salvo `enable row level security` sin políticas, que en un
Postgres normal simplemente no bloquea nada mientras te conectes con el
rol dueño de las tablas — que es lo normal al usar el connection string
que te da Neon).

### 3. Copiar los datos existentes desde Supabase

Mientras Supabase **todavía responda** (aunque esté en grace period, las
lecturas deberían seguir funcionando — solo un 402 total las bloquearía):

```bash
export SUPABASE_URL=...            # el mismo de siempre
export SUPABASE_SERVICE_KEY=...    # el mismo de siempre
export DATABASE_URL=postgresql://...neon.tech/...

python -m scripts.migrate_to_postgres
```

Esto copia todas las tablas (estaciones, observaciones, modelos,
predicciones, histórico completo) y los 4 modelos champion actuales
desde Storage hacia `model_blob`. Es seguro correrlo más de una vez —
todo usa upsert por llave primaria, nunca duplica ni borra nada en el
origen (Supabase).

### 4. Apuntar GitHub Actions al nuevo Postgres

En `Settings → Secrets and variables → Actions` del repo:

- Agregar el secret **`DATABASE_URL`** con el connection string de Neon.
- Dejar `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` como están (no hace falta
  borrarlos — con `DATABASE_URL` presente, el pipeline los ignora).

En cuanto `DATABASE_URL` exista como secret, la **siguiente corrida** de
`predict.yml`/`train.yml` ya usa Postgres en vez de Supabase — sin tocar
ni un archivo `.yml`, sin otro cambio de código.

### 5. Verificar

- Disparar `predict.yml` manualmente (`workflow_dispatch`) y revisar los
  logs: debe sincronizar, predecir y enviar sin errores.
- Confirmar en el Postgres nuevo (`psql "$DATABASE_URL" -c "select count(*) from prediccion;"`)
  que las predicciones nuevas están llegando ahí.

## Para volver a Supabase después

Cuando Supabase deje de estar restringido: borrar (o dejar vacío) el
secret `DATABASE_URL` en GitHub Actions. La siguiente corrida vuelve a
usar Supabase automáticamente — no hay que sincronizar nada de vuelta
salvo que se haya generado actividad nueva en el Postgres de respaldo
mientras tanto (en ese caso, correr `migrate_to_postgres.py` al revés,
o simplemente aceptar el pequeño hueco de historial si el plan B se usó
solo unos días).

## Limitaciones conocidas de este plan B

- El dashboard de Vercel (`dashboard/lib/data.ts`) sigue apuntando a
  Supabase directamente (usa su propia REST API, no pasa por
  `supabase_client.py`) — mientras Supabase esté restringido, el
  dashboard no va a poder leer datos nuevos aunque el pipeline sí
  funcione contra Neon. No se migró porque el dashboard es de solo
  lectura para vos, no afecta el reto ni el leaderboard.
- No se migran las políticas de RLS con granularidad — en Neon, con el
  rol que te da el connection string (dueño de las tablas), RLS sin
  políticas no bloquea nada, igual que hoy con la `service_role` key de
  Supabase.
