import os
from typing import Iterable

from supabase import create_client

from pulso_transmi.client import PulsoTransmiClient


def load_stations(supabase) -> None:
    client = PulsoTransmiClient()
    rows = client.stations().to_dict(orient='records')
    if not rows:
        return
    supabase.table('stations').upsert(rows, on_conflict='station_id').execute()
    print(f"stations: {len(rows)} rows upserted")


def load_observations(supabase) -> None:
    client = PulsoTransmiClient()
    cursor = None
    total = 0
    while True:
        page = client.observations_page(cursor=cursor, limit=5000)
        rows = page.get('data', [])
        if not rows:
            break
        supabase.table('observations').upsert(rows, on_conflict='observed_at,station_id').execute()
        total += len(rows)
        cursor = page.get('next_cursor')
        if cursor is None:
            break
    print(f"observations: {total} rows upserted")


def load_context(supabase) -> None:
    client = PulsoTransmiClient()
    cursor = None
    total = 0
    while True:
        page = client.context_page(cursor=cursor, limit=5000)
        rows = page.get('data', [])
        if not rows:
            break
        supabase.table('context').upsert(rows, on_conflict='observed_at').execute()
        total += len(rows)
        cursor = page.get('next_cursor')
        if cursor is None:
            break
    print(f"context: {total} rows upserted")


if __name__ == '__main__':
    url = os.getenv('SUPABASE_URL')
    key = (
        os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        or os.getenv('SUPABASE_ANON_KEY')
        or os.getenv('SUPABASE_KEY')
    )
    if not url or not key:
        raise RuntimeError(
            'Faltan SUPABASE_URL y una clave válida: SUPABASE_SERVICE_ROLE_KEY, SUPABASE_ANON_KEY o SUPABASE_KEY'
        )

    if os.getenv('SUPABASE_SERVICE_ROLE_KEY'):
        print('Usando SUPABASE_SERVICE_ROLE_KEY para evitar RLS en la migración inicial.')
    else:
        print('Advertencia: se está usando una clave anónima; puede fallar por RLS. Usa service role para migración inicial.')

    supabase = create_client(url, key)
    load_stations(supabase)
    load_observations(supabase)
    load_context(supabase)
