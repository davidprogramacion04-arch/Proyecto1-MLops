from __future__ import annotations

import os
from pathlib import Path

from supabase import create_client

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_pulso_transmi.joblib"


def upload_model() -> str:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            "Faltan las variables de entorno SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY. "
            "También puedes usar SUPABASE_ANON_KEY o SUPABASE_KEY si la configuración lo permite."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo serializado: {MODEL_PATH}")

    supabase = create_client(url, key)
    bucket_name = "models"

    with MODEL_PATH.open("rb") as f:
        result = supabase.storage.from_(bucket_name).upload(
            path="random_forest_pulso_transmi.joblib",
            file=f,
            file_options={"content-type": "application/octet-stream", "upsert": "true"},
        )

    print("Modelo subido con éxito a Supabase Storage.")
    print(result)
    return f"{bucket_name}/random_forest_pulso_transmi.joblib"


if __name__ == "__main__":
    upload_model()
