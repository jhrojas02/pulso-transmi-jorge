# Supabase (opcional / bono)

`schema.sql` contiene el DDL de PostgreSQL derivado de `docs/data-model.md`,
listo para crear las 7 tablas del proyecto (`estacion`, `contexto`,
`observacion`, `feature_vector`, `modelo`, `metrica_validacion`,
`ejecucion_pipeline`, `prediccion`) en un proyecto de Supabase.

## Cómo aplicarlo

**Opción A — SQL Editor (más simple):** en el panel de tu proyecto de
Supabase, abre *SQL Editor → New query*, pega el contenido de `schema.sql`
y ejecuta.

**Opción B — psql / CLI de Supabase**, desde tu propia terminal (no desde
esta sesión, que no tiene credenciales ni acceso configurado a tu proyecto):

```bash
psql "$SUPABASE_DB_URL" -f supabase/schema.sql
# o, con la CLI de Supabase instalada:
supabase db execute -f supabase/schema.sql
```

`SUPABASE_DB_URL` es la cadena de conexión de *Project Settings → Database
→ Connection string* (usa la de "connection pooling" si tu red bloquea
conexiones directas por IPv6).

## Por qué no lo apliqué yo directamente

Esta sesión no tiene un conector de Supabase ni tus credenciales de
proyecto, así que no puedo ejecutar el DDL contra tu base real. Si quieres
que lo haga desde aquí, la forma más segura es que tú mismo corras el
comando de la Opción B en tu terminal — evita pegar la cadena de conexión
(que incluye la contraseña de la base) en el chat.

## Recordatorio de seguridad

**Nunca** expongas la `service_role key` ni la cadena de conexión con
contraseña en código de frontend/navegador. Para el dashboard opcional
(Vercel), usa únicamente la `anon key` pública con Row Level Security
habilitado, o mejor, un backend intermedio que consulte Supabase con la
`service_role key` del lado del servidor.
