from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pulso_transmi import PulsoTransmiClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "random_forest_pulso_transmi.joblib"
METADATA_PATH = MODEL_DIR / "random_forest_pulso_transmi_metadata.json"


def build_feature_frame() -> pd.DataFrame:
    client = PulsoTransmiClient()
    stations = client.stations()
    observations = client.observations_dataframe(page_size=5000)
    context = client.context_dataframe(page_size=5000)

    df = observations.merge(context, on="observed_at", how="left").merge(stations, on="station_id", how="left")
    df["observed_at"] = pd.to_datetime(df["observed_at"]).dt.tz_localize(None)
    df = df.sort_values("observed_at").reset_index(drop=True)

    df["hour"] = df["observed_at"].dt.hour
    df["day_of_week"] = df["observed_at"].dt.dayofweek
    df["month"] = df["observed_at"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"lag_{lag}"] = df.groupby("station_id")["demand"].shift(lag)

    for window in [3, 6, 12, 24]:
        df[f"rolling_mean_{window}"] = (
            df.groupby("station_id")["demand"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    for col in [c for c in df.columns if c.startswith("lag_") or c.startswith("rolling_mean_")]:
        df[col] = df[col].fillna(df[col].median())

    return df


def train_model() -> tuple[Pipeline, dict]:
    df = build_feature_frame()

    target = "demand"
    categorical_features = ["station_id", "corridor"]
    numeric_features = [
        "rain_mm",
        "rain_forecast",
        "temperature_c",
        "temperature_forecast",
        "event_intensity",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    ] + [f"lag_{lag}" for lag in [1, 2, 3, 6, 12, 24]] + [
        f"rolling_mean_{window}" for window in [3, 6, 12, 24]
    ]
    features = categorical_features + numeric_features

    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    X_train = train_df[features]
    y_train = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ],
        remainder="drop",
    )

    rf_grid = {
        "n_estimators": [300, 500, 700],
        "max_depth": [8, 12, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.8],
    }

    best_result = None
    for params in ParameterGrid(rf_grid):
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(random_state=42, n_jobs=-1, **params)),
            ]
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        wape = np.sum(np.abs(y_val.to_numpy() - pred)) / np.sum(np.abs(y_val.to_numpy()))
        accuracy = max(0.0, 100.0 * (1.0 - wape))
        metrics = {
            "r2": float(r2_score(y_val, pred)),
            "rmse": float(np.sqrt(np.mean((y_val.to_numpy() - pred) ** 2))),
            "mae": float(np.mean(np.abs(y_val.to_numpy() - pred))),
            "wape": float(wape),
            "accuracy": float(accuracy),
        }
        candidate = {"params": params, **metrics}
        if best_result is None or candidate["r2"] > best_result["r2"]:
            best_result = candidate

    if best_result is None:
        raise RuntimeError("No se pudo seleccionar un mejor RandomForest.")

    final_params = best_result["params"]
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1, **final_params)),
        ]
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    wape = np.sum(np.abs(y_val.to_numpy() - pred)) / np.sum(np.abs(y_val.to_numpy()))
    accuracy = max(0.0, 100.0 * (1.0 - wape))
    metadata = {
        "model_name": "random_forest_pulso_transmi",
        "selected_model": "RandomForest",
        "target": target,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "params": final_params,
        "validation_wape": float(wape),
        "validation_accuracy": float(accuracy),
    }
    return model, metadata


if __name__ == "__main__":
    model, metadata = train_model()
    joblib.dump(model, MODEL_PATH)
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Metadata guardada en: {METADATA_PATH}")
    print({k: v for k, v in metadata.items() if k not in {"categorical_features", "numeric_features"}})
