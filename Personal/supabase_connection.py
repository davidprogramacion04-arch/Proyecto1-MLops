import os
from supabase import create_client

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")

if not url or not key:
    raise RuntimeError(
        "Faltan las variables de entorno SUPABASE_URL y SUPABASE_ANON_KEY (o SUPABASE_KEY)."
    )

supabase = create_client(url, key)

for table in ["route", "route_stop", "stop", "vehicle", "vehicle_position", "trip", "stop_event"]:
    try:
        result = supabase.table(table).select("*").limit(5).execute()
        print(f"\n--- {table} ---")
        print(result.data)
    except Exception as exc:
        print(f"\n--- {table} ---")
        print(f"ERROR: {exc}")