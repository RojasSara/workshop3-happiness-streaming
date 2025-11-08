# scripts/export_for_bi.py
import os, json, sqlite3
import pandas as pd

# 1) Exportar la tabla predictions de SQLite a CSV
os.makedirs("data", exist_ok=True)
con = sqlite3.connect("data/preds.db")
df = pd.read_sql_query("SELECT * FROM predictions", con)

# Tipos recomendados para Power BI
for c in ("year","is_train","is_test"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
for c in ("actual","prediction","error_abs"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

out_csv = "data/predictions.csv"
df.to_csv(out_csv, index=False)
print(f"OK -> {out_csv} ({len(df)} filas)")

# 2) Exportar métricas del modelo a CSV
with open("artifacts/metrics.json","r") as f:
    metrics = json.load(f)

mdf = pd.DataFrame([metrics])
out_metrics = "artifacts/metrics_for_bi.csv"
mdf.to_csv(out_metrics, index=False)
print(f"OK -> {out_metrics}")

con.close()
