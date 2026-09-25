"""Predice el ciclo vigente y monitorea drift sin filtrar información futura.

Uso seguro (no envía nada):
    Personal/.venv/bin/python -m src.predict_cycle --dry-run

Envío real (solo cuando exista un ciclo abierto):
    Personal/.venv/bin/python -m src.predict_cycle --submit
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd

from pulso_transmi import PulsoTransmiClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT / "Personal" / ".venv" / "env.txt"
MODEL_PATH = ROOT / "models" / "random_forest_pulso_transmi.joblib"
MODEL_METADATA_PATH = ROOT / "models" / "random_forest_pulso_transmi.json"
STATE_PATH = ROOT / "artifacts" / "prediction_state.json"
API_URL = "https://pulso-transmi.72-60-245-2.sslip.io"
FEATURES = [
    "station_id", "corridor", "rain_mm", "rain_forecast", "temperature_c",
    "temperature_forecast", "event_intensity", "hour", "day_of_week", "month",
    "is_weekend", "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_24",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "rolling_mean_24",
]
NUMERIC_DRIFT_FEATURES = ["demand", "lag_1", "lag_6", "lag_24", "rolling_mean_24"]
PSI_THRESHOLD = 0.20
WAPE_DEGRADATION_FACTOR = 1.20


def load_env_file(path: Path) -> None:
    """Carga pares KEY=VALUE locales; nunca imprime secretos."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def api_headers() -> dict[str, str]:
    api_key = os.getenv("PULSO_API_KEY")
    if not api_key:
        raise RuntimeError("Falta PULSO_API_KEY. Cárgala desde los secretos de GitHub o archivo .env.")
    return {"Authorization": f"Bearer {api_key}"}


def get_current_cycle() -> dict[str, Any] | None:
    response = httpx.get(f"{API_URL}/v1/forecast-cycles/current", headers=api_headers(), timeout=30)
    if response.status_code == 404:
        detail = response.json().get("detail", {})
        if detail.get("code") == "no_open_cycle":
            return None
    response.raise_for_status()
    return response.json()


def stream_observations() -> pd.DataFrame:
    """Lee todo el stream publicado; es idempotente al deduplicar después."""
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    with httpx.Client(base_url=API_URL, headers=api_headers(), timeout=30) as client:
        while True:
            params: dict[str, Any] = {"limit": 5000}
            if cursor:
                params["cursor"] = cursor
            response = client.get("/v1/stream/observations", params=params)
            response.raise_for_status()
            page = response.json()
            rows.extend(page.get("data", []))
            cursor = page.get("next_cursor")
            if cursor is None:
                break
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["station_id", "observed_at", "demand", "released_at"])
    frame["station_id"] = frame["station_id"].astype("string")
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["released_at"] = pd.to_datetime(frame["released_at"], utc=True)
    return frame


def historical_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with PulsoTransmiClient(api_key=os.getenv("PULSO_API_KEY")) as client:
        stations = client.stations()
        observations = client.observations_dataframe(page_size=5000)
        context = client.context_dataframe(page_size=5000)
    return stations, observations, context


def sync_to_supabase(stations: pd.DataFrame, observations: pd.DataFrame, context: pd.DataFrame) -> None:
    """Sincroniza datos a Supabase de forma segura si las credenciales están configuradas."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return
    try:
        from supabase import create_client
        supabase = create_client(url, key)
        
        # 1. Stations
        station_rows = stations.to_dict(orient="records")
        if station_rows:
            supabase.table("stations").upsert(station_rows, on_conflict="station_id").execute()
            
        # 2. Context
        if not context.empty:
            ctx_df = context.copy()
            ctx_df["observed_at"] = pd.to_datetime(ctx_df["observed_at"]).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            ctx_rows = ctx_df.to_dict(orient="records")
            # Batch upsert
            for i in range(0, len(ctx_rows), 500):
                supabase.table("context").upsert(ctx_rows[i:i + 500], on_conflict="observed_at").execute()
                
        # 3. Observations (últimos 5000 para sincronización rápida)
        if not observations.empty:
            obs_df = observations.copy().tail(5000)
            obs_df["observed_at"] = pd.to_datetime(obs_df["observed_at"]).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            obs_records = [
                {"observed_at": row["observed_at"], "station_id": str(row["station_id"]).zfill(5), "demand": int(row["demand"])}
                for _, row in obs_df.iterrows()
            ]
            for i in range(0, len(obs_records), 500):
                supabase.table("observations").upsert(obs_records[i:i + 500], on_conflict="observed_at,station_id").execute()
        print("SUPABASE_SYNC=ok")
    except Exception as exc:
        print(f"SUPABASE_SYNC_WARNING: No se pudo sincronizar con Supabase: {exc}", file=sys.stderr)


def combine_observations(historical: pd.DataFrame, streamed: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(
        [historical[["station_id", "observed_at", "demand"]], streamed[["station_id", "observed_at", "demand"]]],
        ignore_index=True,
    )
    combined["station_id"] = combined["station_id"].astype("string")
    combined["observed_at"] = pd.to_datetime(combined["observed_at"], utc=True)
    return (
        combined.drop_duplicates(["station_id", "observed_at"], keep="last")
        .sort_values(["station_id", "observed_at"])
        .reset_index(drop=True)
    )


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().sort_values(["station_id", "observed_at"])
    for lag in (1, 2, 3, 6, 12, 24):
        result[f"lag_{lag}"] = result.groupby("station_id")["demand"].shift(lag)
    for window in (3, 6, 12, 24):
        result[f"rolling_mean_{window}"] = result.groupby("station_id")["demand"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
    return result


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI robusto para variables numéricas; ignora valores faltantes."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) < 20 or len(cur) < 20:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(ref.mean(), cur.mean()) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_share = np.histogram(cur, bins=edges)[0] / len(cur)
    epsilon = 1e-6
    ref_share = np.clip(ref_share, epsilon, None)
    cur_share = np.clip(cur_share, epsilon, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def drift_report(historical: pd.DataFrame, streamed: pd.DataFrame, validation_wape: float | None) -> dict[str, Any]:
    """Compara el stream con la referencia histórica y evalúa predicciones previas."""
    reference = add_lag_features(historical)
    current = add_lag_features(combine_observations(historical, streamed))
    current = current[current["observed_at"] > historical["observed_at"].max()]
    psi: dict[str, float | None] = {}
    for column in NUMERIC_DRIFT_FEATURES:
        value = population_stability_index(reference[column], current[column])
        psi[column] = None if np.isnan(value) else round(value, 4)
    valid_psi = [value for value in psi.values() if value is not None]
    max_psi = max(valid_psi, default=0.0)
    state = read_state()
    history = state.get("predictions", [])
    actual = combine_observations(historical, streamed).rename(columns={"observed_at": "target_at", "demand": "actual"})
    predicted = pd.DataFrame(history)
    rolling_wape: float | None = None
    evaluated = 0
    if not predicted.empty:
        predicted["target_at"] = pd.to_datetime(predicted["target_at"], utc=True)
        predicted["station_id"] = predicted["station_id"].astype("string")
        joined = predicted.merge(actual, on=["station_id", "target_at"], how="inner")
        if not joined.empty and joined["actual"].abs().sum() > 0:
            joined = joined.sort_values("target_at").tail(12 * 24)
            rolling_wape = float((joined["actual"] - joined["value"]).abs().sum() / joined["actual"].abs().sum())
            evaluated = len(joined)
    performance_drift = (
        rolling_wape is not None
        and validation_wape is not None
        and rolling_wape > validation_wape * WAPE_DEGRADATION_FACTOR
    )
    return {
        "stream_rows": int(len(streamed)),
        "psi": psi,
        "max_psi": round(max_psi, 4),
        "data_drift": bool(max_psi >= PSI_THRESHOLD),
        "evaluated_predictions": evaluated,
        "rolling_wape": None if rolling_wape is None else round(rolling_wape, 4),
        "validation_wape": validation_wape,
        "performance_drift": performance_drift,
        "retrain_recommended": bool(max_psi >= PSI_THRESHOLD or performance_drift),
    }


def send_drift_notification(report: dict[str, Any]) -> None:
    """Envía una alerta vía Webhook (Discord/Slack/Teams) si está configurado."""
    webhook_url = os.getenv("DRIFT_WEBHOOK_URL")
    if not webhook_url:
        return
    try:
        content = (
            f"🚨 **ALERTA DE DRIFT DETECTADA - Pulso TransMilenio** 🚨\n\n"
            f"- **Reentrenamiento recomendado:** {report.get('retrain_recommended')}\n"
            f"- **Max PSI:** {report.get('max_psi')} (Umbral: {PSI_THRESHOLD})\n"
            f"- **Data Drift:** {report.get('data_drift')}\n"
            f"- **Performance Drift:** {report.get('performance_drift')}\n"
            f"- **Rolling WAPE:** {report.get('rolling_wape')}\n"
            f"- **Detalle PSI:** ```json\n{json.dumps(report.get('psi', {}), indent=2)}\n```"
        )
        payload = {"content": content, "text": content}
        httpx.post(webhook_url, json=payload, timeout=10)
        print("DRIFT_NOTIFICATION=sent")
    except Exception as exc:
        print(f"NOTIFICATION_WARNING: No se pudo enviar webhook: {exc}", file=sys.stderr)


def write_github_summary(report: dict[str, Any], cycle: dict[str, Any] | None, submission: dict[str, Any] | None) -> None:
    """Escribe resumen para la interfaz de GitHub Actions."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    try:
        status_badge = "🚨 **REENTRENAMIENTO RECOMENDADO**" if report.get("retrain_recommended") else "✅ **MODELO ESTABLE**"
        cycle_info = f"Ciclo: `{cycle.get('cycle_id')}` | Estado Envío: `{submission.get('status')}`" if cycle and submission else "Sin ciclo abierto actualmente."
        markdown = f"""
### 📊 Reporte de Monitoreo - Pulso TransMilenio
* **Estado:** {status_badge}
* **Max PSI:** `{report.get('max_psi')}` (Umbral: `{PSI_THRESHOLD}`)
* **Data Drift:** `{'Sí' if report.get('data_drift') else 'No'}`
* **Performance Drift:** `{'Sí' if report.get('performance_drift') else 'No'}`
* **Rolling WAPE:** `{report.get('rolling_wape')}`

#### ⏱ Estado del Ciclo
{cycle_info}
"""
        with open(summary_path, "a", encoding="utf-8") as file:
            file.write(markdown)
    except Exception:
        pass


def read_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"predictions": []}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_state(predictions: list[dict[str, Any]], drift: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = read_state().get("predictions", [])
    known = {(item["station_id"], item["target_at"], item.get("cycle_id")) for item in existing}
    existing.extend(
        item for item in predictions if (item["station_id"], item["target_at"], item.get("cycle_id")) not in known
    )
    STATE_PATH.write_text(
        json.dumps({"predictions": existing[-10000:], "last_drift": drift}, indent=2), encoding="utf-8"
    )


def model_metadata() -> dict[str, Any]:
    if not MODEL_METADATA_PATH.exists():
        return {}
    return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def last_context_values(context: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float]:
    available = context.copy()
    available["observed_at"] = pd.to_datetime(available["observed_at"], utc=True)
    available = available[available["observed_at"] <= cutoff].sort_values("observed_at")
    fallback = {"rain_mm": 0.0, "rain_forecast": 0.0, "temperature_c": 22.0, "temperature_forecast": 22.0, "event_intensity": 0.0}
    if available.empty:
        return fallback
    last = available.iloc[-1]
    return {key: float(last.get(key, fallback[key])) if pd.notna(last.get(key)) else fallback[key] for key in fallback}


def make_prediction_frame(
    cycle: dict[str, Any], observations: pd.DataFrame, stations: pd.DataFrame, context: pd.DataFrame
) -> pd.DataFrame:
    cutoff = pd.Timestamp(cycle["data_cutoff"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    history = observations[observations["observed_at"] <= cutoff].copy()
    if history.empty:
        raise RuntimeError("No hay observaciones disponibles antes del data_cutoff del ciclo.")
    station_corridor = stations.set_index("station_id")["corridor"].to_dict()
    context_values = last_context_values(context, cutoff)
    targets = sorted(cycle.get("targets", []), key=lambda item: (item["station_id"], item["target_at"]))
    if not targets:
        raise RuntimeError("El ciclo no contiene targets.")
    rows: list[dict[str, Any]] = []
    values: dict[str, dict[pd.Timestamp, float]] = {}
    for station_id, station_frame in history.groupby("station_id"):
        values[str(station_id)] = dict(zip(station_frame["observed_at"], station_frame["demand"].astype(float)))
    for target in targets:
        station_id = str(target["station_id"])
        target_at = pd.Timestamp(target["target_at"])
        if target_at.tzinfo is None:
            target_at = target_at.tz_localize("UTC")
        series = values.get(station_id, {})
        if not series:
            raise RuntimeError(f"No hay historial para la estación {station_id}.")
        ordered = sorted(series)
        fallback = float(series[ordered[-1]])
        lag_values = {lag: float(series.get(target_at - pd.Timedelta(minutes=int(15 * lag)), fallback)) for lag in (1, 2, 3, 6, 12, 24)}
        rolling = {
            window: float(np.mean([series.get(target_at - pd.Timedelta(minutes=int(15 * l)), fallback) for l in range(1, window + 1)]))
            for window in (3, 6, 12, 24)
        }
        row = {
            "station_id": station_id,
            "corridor": station_corridor.get(station_id),
            "target_at": target_at,
            **context_values,
            "hour": target_at.tz_convert("America/Bogota").hour,
            "day_of_week": target_at.tz_convert("America/Bogota").dayofweek,
            "month": target_at.tz_convert("America/Bogota").month,
            "is_weekend": int(target_at.tz_convert("America/Bogota").dayofweek >= 5),
            **{f"lag_{lag}": value for lag, value in lag_values.items()},
            **{f"rolling_mean_{window}": value for window, value in rolling.items()},
        }
        rows.append(row)
    return pd.DataFrame(rows)


def submit(cycle: dict[str, Any], records: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    client_run_id = f"random-forest-{cycle['cycle_id']}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": client_run_id,
        "data_cutoff": cycle["data_cutoff"],
        "model": {
            "version": "random_forest_pulso_transmi",
            "trained_at": metadata.get("trained_at", cycle["data_cutoff"]),
            "training_data_end": metadata.get("training_data_end", cycle["data_cutoff"]),
            "git_commit": git_commit(),
        },
        "predictions": records,
    }
    response = httpx.post(
        f"{API_URL}/v1/submissions",
        headers={**api_headers(), "Idempotency-Key": client_run_id},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json() if response.content else {"status": response.status_code}


def set_github_output(name: str, value: Any) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as file:
            file.write(f"{name}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Envía el payload; sin esta opción solo hace dry-run.")
    parser.add_argument("--dry-run", action="store_true", help="Compatibilidad explícita; es el comportamiento predeterminado.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    args = parser.parse_args()
    load_env_file(args.env_file)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo: {MODEL_PATH}")
    metadata = model_metadata()
    validation_wape = metadata.get("validation_wape")
    stations, historical, context = historical_data()
    streamed = stream_observations()
    
    # Sincronizar con Supabase si está disponible
    sync_to_supabase(stations, combine_observations(historical, streamed), context)
    
    # Calcular Drift
    report = drift_report(historical, streamed, validation_wape)
    print("DRIFT=" + json.dumps(report, ensure_ascii=False))
    set_github_output("retrain_recommended", str(report.get("retrain_recommended", False)).lower())
    set_github_output("max_psi", str(report.get("max_psi", 0.0)))
    
    if report.get("retrain_recommended"):
        send_drift_notification(report)
        
    cycle = get_current_cycle()
    if cycle is None:
        print("No hay ciclo abierto: se calculó drift y se sincronizó, pero no se generó submission.")
        write_github_summary(report, None, None)
        return 0
        
    observations = combine_observations(historical, streamed)
    frame = make_prediction_frame(cycle, observations, stations, context)
    model = joblib.load(MODEL_PATH)
    values = model.predict(frame[FEATURES])
    records = [
        {"station_id": str(row.station_id).zfill(5), "target_at": row.target_at.isoformat(), "value": max(0.0, float(value))}
        for row, value in zip(frame.itertuples(index=False), values)
    ]
    print("PREDICTIONS=" + json.dumps(records, ensure_ascii=False))
    if not args.submit:
        print("Dry-run completado. Usa --submit únicamente después de revisar PREDICTIONS.")
        write_github_summary(report, cycle, {"status": "dry-run"})
        return 0
        
    response = submit(cycle, records, metadata)
    for record in records:
        record["cycle_id"] = cycle["cycle_id"]
    write_state(records, report)
    print("SUBMISSION=" + json.dumps(response, ensure_ascii=False))
    write_github_summary(report, cycle, response)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
