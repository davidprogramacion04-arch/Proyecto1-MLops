from __future__ import annotations

import io
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from supabase import create_client

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "random_forest_pulso_transmi.joblib",
    BASE_DIR / "models" / "random_forest_pulso_transmi.joblib",
    PROJECT_ROOT / "models" / "xgboost_pulso_transmi.joblib",
    BASE_DIR / "models" / "xgboost_pulso_transmi.joblib",
]
MODEL_LOCAL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])

app = FastAPI(title="Pulso TransMi Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    station_id: str
    corridor: str = Field(..., description="Nombre del corredor o estación")
    observed_at: str = Field(..., description="Timestamp ISO en UTC o local")
    rain_mm: float | None = 0.0
    rain_forecast: float | None = 0.0
    temperature_c: float | None = 25.0
    temperature_forecast: float | None = 25.0
    event_intensity: float | None = 0.0
    hour: int | None = None
    day_of_week: int | None = None
    month: int | None = None
    is_weekend: int | None = 0
    lag_1: float | None = None
    lag_2: float | None = None
    lag_3: float | None = None
    lag_6: float | None = None
    lag_12: float | None = None
    lag_24: float | None = None
    rolling_mean_3: float | None = None
    rolling_mean_6: float | None = None
    rolling_mean_12: float | None = None
    rolling_mean_24: float | None = None

    def to_row(self) -> pd.DataFrame:
        timestamp = pd.to_datetime(self.observed_at)
        row = {
            "station_id": self.station_id,
            "corridor": self.corridor,
            "rain_mm": self.rain_mm,
            "rain_forecast": self.rain_forecast,
            "temperature_c": self.temperature_c,
            "temperature_forecast": self.temperature_forecast,
            "event_intensity": self.event_intensity,
            "hour": self.hour if self.hour is not None else timestamp.hour,
            "day_of_week": self.day_of_week if self.day_of_week is not None else timestamp.dayofweek,
            "month": self.month if self.month is not None else timestamp.month,
            "is_weekend": self.is_weekend if self.is_weekend is not None else int(timestamp.dayofweek >= 5),
            "lag_1": self.lag_1,
            "lag_2": self.lag_2,
            "lag_3": self.lag_3,
            "lag_6": self.lag_6,
            "lag_12": self.lag_12,
            "lag_24": self.lag_24,
            "rolling_mean_3": self.rolling_mean_3,
            "rolling_mean_6": self.rolling_mean_6,
            "rolling_mean_12": self.rolling_mean_12,
            "rolling_mean_24": self.rolling_mean_24,
        }
        return pd.DataFrame([row])


def _load_model_from_storage():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")

    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return joblib.load(candidate)

    if url and key:
        supabase = create_client(url, key)
        for model_name in ["random_forest_pulso_transmi.joblib", "xgboost_pulso_transmi.joblib"]:
            try:
                blob = supabase.storage.from_("models").download(model_name)
                return joblib.load(io.BytesIO(blob))
            except Exception:
                continue

    raise FileNotFoundError(
        "No se encontró el modelo local ni en Supabase Storage. "
        "Guárdalo primero con train_and_save_model.py o sube el archivo con upload_model_to_supabase.py."
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "random_forest_pulso_transmi"}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    model = _load_model_from_storage()
    record = payload.to_row()
    pred = float(model.predict(record)[0])
    return {
        "station_id": payload.station_id,
        "observed_at": payload.observed_at,
        "demand_prediction": max(0.0, pred),
        "demand_units": "personas_por_hora",
    }


@app.post("/predict_batch")
def predict_batch(payloads: list[PredictionRequest]) -> list[dict]:
    model = _load_model_from_storage()
    records = pd.concat([item.to_row() for item in payloads], ignore_index=True)
    preds = model.predict(records)
    return [
        {
            "station_id": row.station_id,
            "observed_at": row.observed_at,
            "demand_prediction": max(0.0, float(value)),
        }
        for row, value in zip(payloads, preds)
    ]
