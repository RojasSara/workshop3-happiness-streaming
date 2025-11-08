# src/producer.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Any
import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaProducer
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------
# Cargar .env
# ---------------------------
load_dotenv()

BASE       = Path(__file__).resolve().parents[1]
ART        = BASE / "artifacts"
MODELS     = BASE / "models"

# Kafka / archivos / flags desde .env (con defaults seguros)
BROKERS        = os.getenv("KAFKA_BROKERS", "localhost:9092")
TOPIC          = os.getenv("KAFKA_TOPIC", "workshop3.happiness")
DATASET_CSV    = os.getenv("DATASET_CSV", str(ART / "dataset_unified.csv"))
MODEL_BUNDLE   = os.getenv("MODEL_BUNDLE", str(MODELS / "model.pkl"))

# Control de envío
SEND_ONLY_TEST = os.getenv("SEND_ONLY_TEST", "false").lower() in {"1", "true", "yes"}

# Debe coincidir con tu pipeline offline
RANDOM_STATE   = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE      = float(os.getenv("TEST_SIZE", "0.30"))
TARGET         = "happiness_score"   # objetivo original usado en el offline

# ---------------------------
# Utilidades
# ---------------------------
def to_jsonable(x: Any):
    """Convierte valores de pandas/numpy a tipos JSON serializables."""
    if pd.isna(x):
        return None
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return x

def ensure_flags(df: pd.DataFrame, target_col: str = TARGET) -> pd.DataFrame:
    """
    Si el dataset no trae is_train/is_test, recrea el split 70/30 usando
    la misma semilla del pipeline offline y añade ambas columnas.
    """
    if {"is_train", "is_test"}.issubset(df.columns):
        return df

    if target_col not in df.columns:
        raise ValueError(
            f"No se encontró la columna '{target_col}' para recrear el split 70/30."
        )

    df2 = df.copy()
    # Split solo sobre filas con target no nulo
    valid_idx = df2.dropna(subset=[target_col]).index
    tr_idx, te_idx = train_test_split(
        valid_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    df2["is_train"] = 0
    df2["is_test"]  = 0
    df2.loc[tr_idx, "is_train"] = 1
    df2.loc[te_idx, "is_test"]  = 1

    # Persistimos una copia enriquecida (útil para trazabilidad)
    try:
        out = ART / "dataset_unified_with_flags.csv"
        df2.to_csv(out, index=False)
        print(f"[PRODUCER] Flags creados y guardados en {out}")
    except Exception:
        pass

    return df2

def ensure_actual(df: pd.DataFrame, target_col: str = TARGET) -> pd.DataFrame:
    """Crea columna 'actual' si no existe, copiando del target."""
    if "actual" not in df.columns:
        if target_col not in df.columns:
            raise ValueError(
                f"No existe 'actual' ni '{target_col}' en el dataset para derivarlo."
            )
        df = df.copy()
        df["actual"] = df[target_col]
    return df

# ---------------------------
# Main
# ---------------------------
def main():
    # Cargamos el bundle solo para conocer la lista de features (no se entrena aquí)
    bundle   = joblib.load(MODEL_BUNDLE)     # {"model","features","means",...}
    features = bundle["features"]

    # Dataset unificado del offline
    df = pd.read_csv(DATASET_CSV)

    # Asegurar columnas meta necesarias
    df = ensure_flags(df, target_col=TARGET)
    df = ensure_actual(df, target_col=TARGET)

    # Metadatos útiles para la BD / BI
    meta_cols = [c for c in ["country", "year", "actual", "is_train", "is_test"] if c in df.columns]

    # Filtro según configuración
    to_send = df.copy()
    if SEND_ONLY_TEST and "is_test" in to_send.columns:
        to_send = to_send[to_send["is_test"] == 1].copy()

    # Columnas a serializar (solo las existentes)
    send_cols = meta_cols + [c for c in features if c in to_send.columns]

    # Productor Kafka
    producer = KafkaProducer(
        bootstrap_servers=BROKERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        linger_ms=10,
    )

    modo = "solo TEST" if SEND_ONLY_TEST else "train + test"
    print(f"[PRODUCER] Enviando {len(to_send)} mensajes a {TOPIC} ({modo}) ...")

    sent = 0
    for idx, row in to_send.iterrows():
        msg = {c: to_jsonable(row.get(c)) for c in send_cols}
        producer.send(TOPIC, msg)
        sent += 1

        # Log legible
        print(
            f"→ sent row_id={idx} country={msg.get('country')} year={msg.get('year')} "
            f"train={msg.get('is_train')} test={msg.get('is_test')} actual={msg.get('actual')}"
        )

    producer.flush()
    print(f"[PRODUCER] Sent {sent} messages -> {TOPIC}")

if __name__ == "__main__":
    main()
