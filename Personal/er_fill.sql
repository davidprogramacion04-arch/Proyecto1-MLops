-- Capa derivada para el modelo de entidad-relación
-- Se toma la fuente real de la API (stations, observations, context)
-- y se transforma hacia el modelo conceptual de rutas, paradas y viajes.

-- 1) Rutas: una ruta por corredor
insert into public.route (route_code, name, route_type, active)
select distinct
    lower(replace(corridor, ' ', '_')) as route_code,
    corridor as name,
    'corridor' as route_type,
    true as active
from public.stations
where corridor is not null
on conflict (route_code) do nothing;

-- 2) Paradas: una parada por estación
-- La fuente original usa station_id como texto (ej: '03000');
-- por eso se convierte a bigint para casar con la clave relacional.
insert into public.stop (stop_id, code, name, latitude, longitude, locality, zone)
select
    cast(station_id as bigint) as stop_id,
    station_id as code,
    station_name as name,
    latitude,
    longitude,
    corridor as locality,
    corridor as zone
from public.stations
on conflict (stop_id) do nothing;

-- 3) Relación ruta-parada: cada estación pertenece al corredor correspondiente
with station_rank as (
    select
        s.station_id,
        s.corridor,
        row_number() over (
            partition by s.corridor
            order by s.station_name
        ) as stop_sequence
    from public.stations s
    where s.corridor is not null
)
insert into public.route_stop (route_id, stop_id, stop_sequence, direction)
select
    r.route_id,
    cast(sr.station_id as bigint) as stop_id,
    sr.stop_sequence,
    'inbound' as direction
from station_rank sr
join public.route r
  on r.route_code = lower(replace(sr.corridor, ' ', '_'))
on conflict do nothing;

-- 4) Vehículos: un vehículo sintético por corredor
insert into public.vehicle (vehicle_code, vehicle_type, capacity, operator, active)
select distinct
    lower(replace(corridor, ' ', '_')) || '_veh' as vehicle_code,
    'corridor_vehicle' as vehicle_type,
    0 as capacity,
    corridor as operator,
    true as active
from public.stations
where corridor is not null
on conflict (vehicle_code) do nothing;

-- 5) Viajes: un viaje sintético por (ruta, vehículo, fecha, hora)
insert into public.trip (route_id, vehicle_id, service_date, start_time, end_time, direction, status)
select distinct
    r.route_id,
    v.vehicle_id,
    cast(o.observed_at as date) as service_date,
    cast(o.observed_at as time) as start_time,
    cast(o.observed_at as time) as end_time,
    'inbound' as direction,
    'completed' as status
from public.observations o
join public.stations s
  on s.station_id = o.station_id
join public.route r
  on r.route_code = lower(replace(s.corridor, ' ', '_'))
join public.vehicle v
  on v.vehicle_code = lower(replace(s.corridor, ' ', '_')) || '_veh'
on conflict (route_id, vehicle_id, service_date, start_time) do nothing;

-- 6) Eventos de parada: cada observación se convierte en un stop_event
insert into public.stop_event (trip_id, stop_id, arrival_time, departure_time, dwell_seconds, sequence)
select
    t.trip_id,
    cast(s.station_id as bigint) as stop_id,
    o.observed_at as arrival_time,
    o.observed_at as departure_time,
    900 as dwell_seconds,
    row_number() over (
        partition by t.trip_id
        order by o.observed_at
    ) as sequence
from public.observations o
join public.stations s
  on s.station_id = o.station_id
join public.route r
  on r.route_code = lower(replace(s.corridor, ' ', '_'))
join public.trip t
  on t.route_id = r.route_id
 and t.service_date = cast(o.observed_at as date)
 and t.start_time = cast(o.observed_at as time)
on conflict do nothing;

-- 7) Posiciones de vehículos: sintéticas, derivadas de la estación y el timestamp
insert into public.vehicle_position (vehicle_id, latitude, longitude, speed, heading, timestamp)
select
    v.vehicle_id,
    s.latitude,
    s.longitude,
    0 as speed,
    0 as heading,
    o.observed_at as timestamp
from public.observations o
join public.stations s
  on s.station_id = o.station_id
join public.vehicle v
  on v.vehicle_code = lower(replace(s.corridor, ' ', '_')) || '_veh';

-- Consulta de validación
select 'route' as tabla, count(*) as total from public.route
union all
select 'stop', count(*) from public.stop
union all
select 'route_stop', count(*) from public.route_stop
union all
select 'vehicle', count(*) from public.vehicle
union all
select 'trip', count(*) from public.trip
union all
select 'stop_event', count(*) from public.stop_event
union all
select 'vehicle_position', count(*) from public.vehicle_position;
