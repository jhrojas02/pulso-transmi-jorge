-- Esquema PostgreSQL para Supabase — Pulso TransMi
-- Traduce el diagrama entidad-relación de docs/data-model.md a DDL real.
-- Uso: pegar en el SQL Editor de Supabase, o `psql "$SUPABASE_DB_URL" -f supabase/schema.sql`
--
-- Notas de diseño respecto al diagrama Mermaid original:
--   - station_id es TEXT, no INT: el SDK usa códigos como "07107" (cero a la
--     izquierda), que un entero destruiría.
--   - feature_list en MODELO es JSONB (lista de nombres de features), en vez
--     de un string plano, para poder consultarlo sin parsear.
--   - Ningún ID tiene DEFAULT gen_random_uuid(): los IDs (model_id, run_id,
--     metric_id, prediction_id) se generan en el pipeline con nombres
--     legibles y trazables (ej. "model_2026-09-08_gbm_v3"), no UUIDs opacos.

-- ============================================================
-- Capa 1: datos fuente (tal cual los entrega el API del starter kit)
-- ============================================================

create table if not exists estacion (
    station_id      text primary key,
    station_name    text not null,
    corridor        text not null,
    latitude        double precision not null,
    longitude       double precision not null
);

create table if not exists contexto (
    observed_at             timestamptz primary key,
    rain_mm                 double precision,
    rain_forecast           double precision,
    temperature_c           double precision,
    temperature_forecast    double precision,
    event_intensity         double precision
);

create table if not exists observacion (
    station_id      text not null references estacion (station_id),
    observed_at     timestamptz not null references contexto (observed_at),
    demand          integer not null check (demand >= 0),
    primary key (station_id, observed_at)
);

create index if not exists idx_observacion_observed_at on observacion (observed_at);

-- ============================================================
-- Capa 2: Machine Learning (construida sobre la capa anterior)
-- ============================================================

create table if not exists feature_vector (
    station_id          text not null,
    observed_at         timestamptz not null,
    hour                smallint not null check (hour between 0 and 23),
    day_of_week         smallint not null check (day_of_week between 0 and 6),
    is_weekend          boolean not null,
    lag_1               double precision,
    lag_4_96            double precision,
    rolling_mean_24h    double precision,
    rain_mm             double precision,
    temperature_c       double precision,
    event_intensity     double precision,
    target_demand       double precision not null,
    primary key (station_id, observed_at),
    foreign key (station_id, observed_at) references observacion (station_id, observed_at),
    foreign key (observed_at) references contexto (observed_at)
);

create table if not exists modelo (
    model_id                text primary key,
    version                 text not null,
    algoritmo               text not null,
    trained_at              timestamptz not null,
    cutoff_train_inicio     date not null,
    cutoff_train_fin        date not null,
    code_commit             text not null,
    feature_list            jsonb not null,
    artifact_uri            text not null,
    check (cutoff_train_inicio <= cutoff_train_fin)
);

create table if not exists metrica_validacion (
    metric_id       text primary key,
    model_id        text not null references modelo (model_id),
    station_id      text references estacion (station_id), -- NULL = agregado
    split           text not null,
    wape            double precision not null check (wape >= 0),
    accuracy        double precision not null,
    evaluated_at    timestamptz not null
);

create index if not exists idx_metrica_model on metrica_validacion (model_id);
create index if not exists idx_metrica_station on metrica_validacion (station_id);

create table if not exists ejecucion_pipeline (
    run_id                  text primary key,
    run_at                  timestamptz not null,
    cursor_hasta            timestamptz not null,
    status                  text not null check (status in ('ok', 'error', 'parcial')),
    drift_metric            double precision,
    decision_reentrenar     boolean not null,
    motivo_decision         text not null,
    model_id                text references modelo (model_id)
);

create table if not exists prediccion (
    prediction_id       text primary key,
    run_id               text not null references ejecucion_pipeline (run_id),
    model_id             text not null references modelo (model_id),
    station_id           text not null references estacion (station_id),
    target_timestamp     timestamptz not null,
    horizonte             smallint not null check (horizonte between 1 and 4),
    demanda_predicha      double precision not null,
    generated_at          timestamptz not null
);

create index if not exists idx_prediccion_run on prediccion (run_id);
create index if not exists idx_prediccion_station_target on prediccion (station_id, target_timestamp);
