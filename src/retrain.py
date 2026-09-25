"""Reentrenamiento del modelo Random Forest incorporando histórico + stream reciente.

Uso:
    Personal/.venv/bin/python -m src.retrain
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.predict_cycle import (
    DEFAULT_ENV_FILE,
    FEATURES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    combine_observations,
    historical_data,
    load_env_file,
    stream_observations,
)

ROOT = Path(__file__).resolve().parents[1]


def build_training_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Descarga histórico y stream, une observaciones y genera la matriz de entrenamiento."""
    stations, historical, context = historical_data()
    streamed = stream_observations()
    observations = combine_observations(historical, streamed)
    
    # Mapear estaciones y contexto
    station_corridor = stations.set_index("station_id")["corridor"].to_dict()
    
    # Unir contexto por observed_at
    context_clean = context.copy()
    context_clean["observed_at"] = pd.to_datetime(context_clean["observed_at"], utc=True)
    
    merged = pd.merge_asof(
        observations.sort_values("observed_at"),
        context_clean.sort_values("observed_at"),
        on="observed_at",
        direction="backward",
    )
    
    # Imputar contexto si hubiera nulos
    merged["rain_mm"] = merged["rain_mm"].fillna(0.0)
    merged["rain_forecast"] = merged["rain_forecast"].fillna(0.0)
    merged["temperature_c"] = merged["temperature_c"].fillna(22.0)
    merged["temperature_forecast"] = merged["temperature_forecast"].fillna(22.0)
    merged["event_intensity"] = merged["event_intensity"].fillna(0.0)
    
    merged["corridor"] = merged["station_id"].map(station_corridor).fillna("Unknown")
    
    # Features temporales en hora Colombia
    bogota_time = merged["observed_at"].dt.tz_convert("America/Bogota")
    merged["hour"] = bogota_time.dt.hour
    merged["day_of_week"] = bogota_time.dt.dayofweek
    merged["month"] = bogota_time.dt.month
    merged["is_weekend"] = (merged["day_of_week"] >= 5).astype(int)
    
    # Lags y Rolling means por estación
    merged = merged.sort_values(["station_id", "observed_at"]).reset_index(drop=True)
    for lag in (1, 2, 3, 6, 12, 24):
        merged[f"lag_{lag}"] = merged.groupby("station_id")["demand"].shift(lag)
    for window in (3, 6, 12, 24):
        merged[f"rolling_mean_{window}"] = merged.groupby("station_id")["demand"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        
    # Eliminar primeras filas con nulos causados por lags máximos
    clean_df = merged.dropna(subset=[f"lag_{lag}" for lag in (1, 2, 3, 6, 12, 24)]).reset_index(drop=True)
    return clean_df, stations, historical, streamed


def train_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    categorical_features = ["station_id", "corridor"]
    numeric_features = [
        "rain_mm", "rain_forecast", "temperature_c", "temperature_forecast",
        "event_intensity", "hour", "day_of_week", "month", "is_weekend",
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_24",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "rolling_mean_24"
    ]
    
    target = "demand"
    
    # Split temporal (85% train, 15% validación temporal)
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    X_train = train_df[FEATURES]
    y_train = train_df[target]
    X_val = val_df[FEATURES]
    y_val = val_df[target]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ],
        remainder="drop",
    )
    
    # Hiperparámetros optimizados y ligeros
    best_params = {
        "n_estimators": 120,
        "max_depth": 14,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(**best_params)),
        ]
    )
    
    print("Entrenando en conjunto de validación temporal...")
    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict(X_val)
    
    # Métricas
    wape = float(np.sum(np.abs(y_val.to_numpy() - val_preds)) / np.sum(np.abs(y_val.to_numpy())))
    accuracy = float(max(0.0, 100.0 * (1.0 - wape)))
    rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    mae = float(mean_absolute_error(y_val, val_preds))
    r2 = float(r2_score(y_val, val_preds))
    
    print(f"Resultados Validación Temporal -> WAPE: {wape:.4f} | Accuracy: {accuracy:.2f}% | R2: {r2:.4f} | RMSE: {rmse:.2f}")
    
    # Reentrenar sobre todo el dataset (100% de datos disponibles hasta ahora)
    print("Reentrenando pipeline con el 100% de los datos históricos + stream...")
    pipeline.fit(df[FEATURES], df[target])
    
    now = datetime.now(timezone.utc)
    metadata = {
        "model_name": "random_forest_pulso_transmi",
        "trained_at": now.isoformat(),
        "training_data_end": df["observed_at"].max().isoformat(),
        "training_samples": len(df),
        "params": best_params,
        "validation_wape": round(wape, 4),
        "validation_accuracy": round(accuracy, 2),
        "validation_r2": round(r2, 4),
        "validation_rmse": round(rmse, 2),
        "validation_mae": round(mae, 2),
    }
    return pipeline, metadata


def main() -> int:
    load_env_file(DEFAULT_ENV_FILE)
    print("Cargando datos para reentrenamiento...")
    df, stations, historical, streamed = build_training_dataset()
    print(f"Dataset total preparado: {len(df)} filas (Histórico: {len(historical)} + Stream: {len(streamed)}).")
    
    model, metadata = train_and_evaluate(df)
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    print("REENTRENAMIENTO_COMPLETADO=ok")
    print("METADATA=" + json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
