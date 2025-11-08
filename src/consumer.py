# src/consumer.py
from __future__ import annotations
import os, json, time
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaConsumer
from sqlalchemy import create_engine

# -------------------------------
# Cargar variables de entorno
# -------------------------------
load_dotenv()

BASE          = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = BASE / "models" / "model.pkl"

BROKERS       = os.getenv("KAFKA_BROKERS", "localhost:9092")
TOPIC         = os.getenv("KAFKA_TOPIC", "workshop3.happiness")
MODEL_BUNDLE  = os.getenv("MODEL_BUNDLE", str(DEFAULT_MODEL))
SQLITE_PATH   = os.getenv("SQLITE_PATH", "data/preds.db")
SQLITE_TABLE  = os.getenv("SQLITE_TABLE", "predictions")

# -------------------------------
# Utilidades
# -------------------------------
def ensure_sqlite():
    """Crea la BD/tabla si no existen (con flags is_train/is_test)."""
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    eng = create_engine(f"sqlite:///{SQLITE_PATH}")
    with eng.begin() as conn:
        conn.exec_driver_sql(f"""
            CREATE TABLE IF NOT EXISTS {SQLITE_TABLE} (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          DATETIME DEFAULT CURRENT_TIMESTAMP,
                country     TEXT,
                year        INTEGER,
                is_train    INTEGER,
                is_test     INTEGER,
                actual      REAL,
                prediction  REAL,
                error_abs   REAL
            );
        """)
    return eng

def to_float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None

def to_int01(x):
    try:
        v = int(x)
        return 1 if v == 1 else 0
    except Exception:
        # aceptar "true"/"false" como texto
        if isinstance(x, str) and x.lower() in {"true","1","yes"}:
            return 1
        return 0

# -------------------------------
# Main
# -------------------------------
def main():
    # Cargar modelo + metadatos
    bundle   = joblib.load(MODEL_BUNDLE)          # {"model","features","means",...}
    model    = bundle["model"]
    features = bundle["features"]
    means    = bundle.get("means", {})

    # DB lista
    eng = ensure_sqlite()

    # Consumidor Kafka
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="wk3-consumer-group",
    )

    print(f"[CONSUMER] Escuchando {TOPIC} en {BROKERS} ...")

    for msg in consumer:
        rec = msg.value  # dict que envía el producer

        country = rec.get("country")
        year    = rec.get("year")

        # Vector de features con imputación por medias del TRAIN
        row_feats = {}
        for f in features:
            v = to_float_or_none(rec.get(f))
            if v is None:
                v = means.get(f, 0.0)
            row_feats[f] = v
        X = pd.DataFrame([row_feats], columns=features)

        # Predicción
        y_pred  = float(model.predict(X)[0])
        actualf = to_float_or_none(rec.get("actual"))

        # Flags 0/1
        is_train = to_int01(rec.get("is_train"))
        is_test  = to_int01(rec.get("is_test"))

        # Error absoluto (si hay ground truth)
        err = abs(y_pred - actualf) if actualf is not None else None

        # Guardar en SQLite
        out = pd.DataFrame([{
            "country": country,
            "year": year,
            "is_train": is_train,
            "is_test": is_test,
            "actual": actualf,
            "prediction": y_pred,
            "error_abs": err
        }])
        out.to_sql(SQLITE_TABLE, con=eng, if_exists="append", index=False)

        print(f"← recv country={country} year={year} "
              f"train={is_train} test={is_test} actual={actualf} pred={y_pred:.3f} err={err}")

        # pausa mínima para que el log sea legible
        time.sleep(0.003)

if __name__ == "__main__":
    main()
