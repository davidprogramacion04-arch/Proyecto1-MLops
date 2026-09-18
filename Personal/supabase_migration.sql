-- Ejecutar en Supabase SQL Editor
-- Esquema real de la API de Pulso TransMi

create table if not exists public.stations (
  station_id text primary key,
  station_name text not null,
  corridor text,
  latitude double precision,
  longitude double precision,
  created_at timestamptz not null default now()
);

create table if not exists public.observations (
  observed_at timestamptz not null,
  station_id text not null references public.stations(station_id) on delete cascade,
  demand integer not null,
  created_at timestamptz not null default now(),
  primary key (observed_at, station_id)
);

create table if not exists public.context (
  observed_at timestamptz primary key,
  rain_mm double precision,
  rain_forecast double precision,
  temperature_c double precision,
  temperature_forecast double precision,
  event_intensity double precision,
  created_at timestamptz not null default now()
);

create index if not exists idx_observations_station_id on public.observations(station_id);
create index if not exists idx_observations_observed_at on public.observations(observed_at);
create index if not exists idx_context_observed_at on public.context(observed_at);

-- Validación rápida del esquema
select table_schema, table_name
from information_schema.tables
where table_schema = 'public'
  and table_name in ('stations', 'observations', 'context')
order by table_name;

-- Consultas de prueba
select * from public.stations limit 5;
select * from public.observations limit 5;
select * from public.context limit 5;
