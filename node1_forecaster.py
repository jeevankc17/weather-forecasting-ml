# ─────────────────────────────────────────────────────────────
# EnvirologApp — ML Forecaster for node-1
# Model: XGBoost  |  Targets: Temperature, Humidity, Air Quality
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import json
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Database URL from backend .env
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_9YoWFjEGt2Ug@ep-long-unit-a11pph7p-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)


   
def retrain():
    """
    Retrain model using latest data from sensor_data table.
    Fetches temperature, humidity, airQuality from PostgreSQL database.
    """
    # ── STEP 1: Load data from database ───────────────────────────
    engine = create_engine(DATABASE_URL)
    
    query = text("""
    SELECT 
        "createdAt" as "Timestamp",
        temperature as "Temperature (°C)",
        humidity as "Humidity (%)",
        "airQuality" as "Air Quality"
    FROM sensor_data
    WHERE temperature IS NOT NULL
      AND humidity IS NOT NULL
      AND "airQuality" IS NOT NULL
    ORDER BY "createdAt" ASC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    engine.dispose()  # Close connection
    
    if len(df) < 10:
        print(f"Not enough data to train: {len(df)} rows. Need at least 10.")
        return
    
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    print(f"Loaded {len(df)} rows from database")


    # ── STEP 2: Create lag features ───────────────────────────────
    # The model learns from the previous 3 readings to predict the next.
    targets = ["Temperature (°C)", "Humidity (%)", "Air Quality"]

    for col in targets:
        df[f"{col}_lag1"] = df[col].shift(1)   # value 1 step ago
        df[f"{col}_lag2"] = df[col].shift(2)   # value 2 steps ago
        df[f"{col}_lag3"] = df[col].shift(3)   # value 3 steps ago
        df[f"{col}_lag4"] = df[col].shift(4)   # value 4 steps ago
        df[f"{col}_lag5"] = df[col].shift(5)   # value 5 steps ago

    df = df.dropna().reset_index(drop=True)    # remove rows with missing lags


    # ── STEP 3: Split features (X) and targets (y) ───────────────
    feature_cols = [c for c in df.columns if "lag" in c]
    X = df[feature_cols]
    y = df[targets]

    print(f"Features: {feature_cols}")


    # ── STEP 4: Train / test split (last 20% = test) ─────────────

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )


    # ── STEP 5: Train XGBoost model ───────────────────────────────
    # MultiOutputRegressor trains one XGBoost per target column.
    model = MultiOutputRegressor(
        xgb.XGBRegressor(n_estimators=400, max_depth=4,
                        learning_rate=0.1, random_state=42)
    )
    model.fit(X_train, y_train)
    print("Training done!")


    # ── STEP 6: Evaluate on test set ──────────────────────────────
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds, multioutput="raw_values")

    print("\n── Test MAE ──────────────────────────────────")

    for i in range(len(targets)):
        print(f"{targets[i]} MAE = {mae[i]:.3f}")

    # ── STEP 7: Forecast next readings with timestamps ─────────
    # Generate exact points needed for each duration option (independent forecasts)

    avg_gap_seconds = df["Timestamp"].diff().dropna().dt.total_seconds().mean()
    last_time       = df["Timestamp"].iloc[-1]   # last known timestamp
    last_reading    = df.iloc[-1]
    base_history    = df[targets].values.tolist()[-5:]  # Last 5 readings as base

    # Define forecast configurations: (duration_name, interval_minutes, steps)
    forecast_configs = [
        ('30min', 1, 30),      # 30min: 30 points at 1 min intervals
        ('1hr', 1, 60),        # 1hr: 60 points at 1 min intervals
        ('6hr', 5, 72),        # 6hr: 72 points at 5 min intervals
        ('12hr', 10, 72),      # 12hr: 72 points at 10 min intervals
        ('24hr', 60, 24),      # 24hr: 24 points at 1 hour intervals
        ('1week', 1440, 7),    # 1week: 7 points at 1 day intervals
    ]

    future_preds = []
    for duration_name, interval_mins, steps in forecast_configs:
        interval_seconds = interval_mins * 60
        history = base_history.copy()  # Start fresh for each duration
        
        for step in range(steps):
            # Build one feature row from last 5 readings
            row = []
            for col_idx in range(len(targets)):
                row += [history[-1][col_idx],   # lag1
                        history[-2][col_idx],   # lag2
                        history[-3][col_idx],   # lag3
                        history[-4][col_idx],   # lag4
                        history[-5][col_idx]]   # lag5

            pred = model.predict([row])[0]
            pred_time = last_time + pd.Timedelta(seconds=interval_seconds * (step + 1))

            history.append(pred.tolist())
            future_preds.append({
                "step":             len(future_preds) + 1,
                "predictedTime":    pred_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
                "temperature":      float(round(pred[0], 2)),
                "humidity":         float(round(pred[1], 2)),
                "airQuality":       float(round(pred[2], 1)),
                "duration":         duration_name,
            })

    print(f"\n── Generated {len(future_preds)} forecast steps ────────────────────────────")

    # ── STEP 8: Save model ───────────────────────────────────────
    joblib.dump(model, "node1_model.pkl")
    joblib.dump(feature_cols, "node1_features.pkl")
    print("Model saved as node1_model.pkl")

    # ── STEP 9: Save forecast data as JSON for backend API ─────────────
    forecast_data = {
        "lastUpdated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        "avgGapSeconds": float(round(avg_gap_seconds, 2)),
        "rowCount": int(len(df)),
        "lastReading": {
            "timestamp": last_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
            "temperature": float(last_reading["Temperature (°C)"]),
            "humidity": float(last_reading["Humidity (%)"]),
            "airQuality": float(last_reading["Air Quality"]),
        },
        "mae": {
            "temperature": float(round(mae[0], 3)),
            "humidity": float(round(mae[1], 3)),
            "airQuality": float(round(mae[2], 3)),
        },
        "forecast": future_preds,
    }

    with open("forecast_data.json", "w") as f:
        json.dump(forecast_data, f, indent=2)
    
    print("Forecast data saved as forecast_data.json")
    print(f"Last reading: {last_time} | Avg gap: {avg_gap_seconds:.1f}s")