# Happiness Score Streaming Pipeline

This project builds a complete data workflow to predict the **World Happiness Score** using Python, scikit-learn, Kafka, and SQLite.  
It includes ETL processing, model training, real-time streaming, and a small BI dashboard for evaluation.

## Overview
- Load and unify five happiness datasets (2015–2019)
- Clean, transform and generate EDA outputs
- Train a Linear Regression model and save it as `model.pkl`
- Kafka Producer streams processed records
- Kafka Consumer predicts and stores results in `preds.db`
- Export predictions to CSV for Power BI dashboard

## Key Files
/artifacts – metrics, features and unified dataset
/data – raw, processed data and SQLite database
/models – trained model (model.pkl)
/src – producer, consumer, and offline pipeline
/scripts – export_for_bi.py
/dashboard – Power BI report (PBIX)

markdown
Copiar código

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
Run offline ETL + training:

bash
Copiar código
python src/run_offline_pipeline.py
Start Kafka:

bash
Copiar código
docker-compose up -d
Start Producer:

bash
Copiar código
python src/producer.py
Start Consumer:

bash
Copiar código
python src/consumer.py
Export data for Power BI:

bash
Copiar código
python scripts/export_for_bi.py
Dashboard
Power BI dashboard included in:

bash
Copiar código
/dashboard/dashboard_happiness.pbix