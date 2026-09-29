-- Plan B (docs/plan-b-postgres.md): equivalente al bucket "models" de
-- Supabase Storage, pero como tabla de Postgres genérico (Neon, Railway,
-- etc. no ofrecen storage de objetos como Supabase). A 7-9MB por modelo,
-- guardarlos como bytea es perfectamente razonable.
--
-- Uso: `psql "$DATABASE_URL" -f supabase/schema.sql -f supabase/model_blob.sql`
-- (además del schema.sql normal, que ya es Postgres estándar y no
-- necesita cambios para correr fuera de Supabase).

create table if not exists model_blob (
    path        text primary key,
    data        bytea not null,
    updated_at  timestamptz not null default now()
);
