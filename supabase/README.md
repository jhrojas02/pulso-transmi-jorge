# Supabase (opcional / bono)

`schema.sql` contiene el DDL de PostgreSQL derivado de `docs/data-model.md`:
8 tablas (`estacion`, `contexto`, `observacion`, `feature_vector`, `modelo`,
`metrica_validacion`, `ejecucion_pipeline`, `prediccion`) con FKs, checks e
índices, más Row Level Security habilitado sin políticas.

## Estado actual

Ya aplicado en un proyecto real de Supabase, vía el conector MCP conectado
a esta cuenta de Claude:

- **Proyecto:** `pulso-transmi` (región `sa-east-1`, plan gratuito)
- **URL del API:** `https://giahdocqjnpscgbskkyf.supabase.co`
- **Las 8 tablas están creadas** y con RLS **habilitado sin políticas** —
  bloqueado para `anon`/`authenticated`, solo accesible con la
  `service_role key` (que siempre salta RLS).

`schema.sql` queda como la fuente de verdad versionada del esquema — si se
necesita recrear el proyecto o levantar uno nuevo, se vuelve a aplicar tal
cual (es idempotente: usa `if not exists`).

## Cómo conectar el pipeline (`src/ingest.py`, etc.) a esta base

1. En el dashboard de Supabase → *Project Settings → API*, copia la
   **`service_role key`** (no la `anon`/`publishable`).
2. Guárdala como **secret de GitHub Actions** en tu repo
   (`SUPABASE_SERVICE_ROLE_KEY`), nunca en el código ni en `.env` versionado.
3. El pipeline se conecta con esa key + la URL de arriba; como usa
   `service_role`, ignora RLS y puede leer/escribir todas las tablas.

## Modelos promovidos (Supabase Storage)

Bucket **privado** `models`, creado vía Storage API con la `secret key`
(nunca con la `publishable key`, que quedaría bloqueada por RLS/policies
de Storage igual que las tablas). Convención de ruta, para que cada
versión quede identificable y nunca se sobrescriba silenciosamente:

```
models/<model_id>/gbm.joblib
```

`model_id` coincide con la fila de la tabla `modelo` (ej.
`model_2026-09-18_hybrid_h15`), y `modelo.artifact_uri` guarda la
referencia como `supabase-storage://models/<model_id>/gbm.joblib` — el
componente que sirve predicciones (`predict.py`, aún por construir)
debe leer esa columna, no asumir la ruta.

Nota: el modelo "champion" real por horizonte es un híbrido (naive +
GBM, elegido por estación en validación); lo único que se serializa en
Storage es la parte de gradient boosting — el baseline naive se
recalcula en tiempo de predicción a partir de `observacion`/`contexto`
(no tiene estado entrenable que valga la pena guardar como artefacto).
`modelo.feature_list` incluye qué estación usa cuál de los dos.

## Si en el futuro se construye el dashboard (Vercel)

El dashboard NO debe usar la `service_role key` en el navegador. Dos
opciones:
- Agregar políticas de **`SELECT`** explícitas por tabla para el rol
  `anon` (solo lectura) y usar la `anon`/`publishable key` en el frontend.
- O, más seguro, un backend intermedio (API route de Vercel) que consulte
  Supabase con la `service_role key` del lado del servidor y nunca la
  exponga al navegador.

Cualquiera de las dos formas se agrega como una migración nueva en este
archivo (o uno adicional), nunca deshabilitando RLS.

## Cómo volver a aplicar el esquema manualmente (si hiciera falta)

**SQL Editor (más simple):** en el panel del proyecto, *SQL Editor → New
query*, pega el contenido de `schema.sql` y ejecuta.

**psql / CLI de Supabase**, desde tu propia terminal:

```bash
psql "$SUPABASE_DB_URL" -f supabase/schema.sql
# o, con la CLI de Supabase instalada:
supabase db execute -f supabase/schema.sql
```

`SUPABASE_DB_URL` es la cadena de conexión de *Project Settings → Database
→ Connection string*.

## Recordatorio de seguridad

**Nunca** expongas la `service_role key` ni una cadena de conexión con
contraseña en código de frontend/navegador ni en el repo. Va como secret
de GitHub Actions o variable de entorno del servidor, igual que la API key
de Pulso TransMi.
