# src/run_offline_pipeline.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Compatibilidad para calcular RMSE con cualquier versión de scikit-learn
try:
    # scikit-learn >= 1.3
    from sklearn.metrics import root_mean_squared_error as sk_rmse
    _HAS_SK_RMSE = True
except Exception:
    _HAS_SK_RMSE = False

# -------------------------------------------------
# Rutas y configuración
# -------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "data_raw"
EDA_OUT = BASE / "data" / "eda_out"
CHARTS = EDA_OUT / "charts"
ART = BASE / "artifacts"
MODELS = BASE / "models"

for d in [EDA_OUT, CHARTS, ART, MODELS]:
    d.mkdir(parents=True, exist_ok=True)

YEARS = [2015, 2016, 2017, 2018, 2019]
FILES = [RAW_DIR / f"{y}.csv" for y in YEARS]

RANDOM_STATE = 42
TEST_SIZE = 0.30
TARGET = "happiness_score"

# -------------------------------------------------
# Utilidades
# -------------------------------------------------
def normalize_cols(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        c2 = str(c).strip().lower()
        c2 = re.sub(r"[/\-.(),]", " ", c2)
        c2 = re.sub(r"\s+", " ", c2).strip().replace(" ", "_")
        out.append(c2)
    return out

def canonicalize(df: pd.DataFrame, year_hint: int | None) -> pd.DataFrame:
    """Unifica nombres/formatos distintos entre años y tipa numéricos relevantes."""
    df = df.copy()
    df.columns = normalize_cols(df.columns)

    # country
    if "country" not in df.columns:
        for alt in ["country_or_region", "country_name", "name"]:
            if alt in df.columns:
                df["country"] = df[alt]
                break

    # target
    if "happiness_score" not in df.columns:
        for alt in ["life_ladder", "ladder_score", "score"]:
            if alt in df.columns:
                df["happiness_score"] = df[alt]
                break

    # GDP per capita
    if "gdp_per_capita" not in df.columns:
        for alt in ["logged_gdp_per_capita", "gdp_per_capita_ppp", "economy_gdp_per_capita", "gdp"]:
            if alt in df.columns:
                df["gdp_per_capita"] = df[alt]
                break

    # Healthy life expectancy
    if "healthy_life_expectancy" not in df.columns:
        for alt in ["health_life_expectancy", "life_expectancy", "healthy_life_expectancy_at_birth"]:
            if alt in df.columns:
                df["healthy_life_expectancy"] = df[alt]
                break

    # year desde nombre del archivo si falta
    if "year" not in df.columns and year_hint is not None:
        df["year"] = year_hint

    # Castear numéricos (suave)
    numeric_like = [
        "happiness_score", "gdp_per_capita", "social_support",
        "healthy_life_expectancy", "freedom", "generosity",
        "perceptions_of_corruption", "dystopia_residual"
    ]
    for c in numeric_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = [
        "country","year","region","happiness_score","gdp_per_capita","social_support",
        "healthy_life_expectancy","freedom","generosity","perceptions_of_corruption",
        "dystopia_residual"
    ]
    present = [c for c in keep if c in df.columns]
    return df[present]

# -------------------------------------------------
# EDA básico (CSV en data/eda_out/)
# -------------------------------------------------
def run_eda(full: pd.DataFrame, per_year_cols: Dict[int, List[str]]):
    # 1) shapes
    shapes = [{"year": y, "rows": int((full["year"] == y).sum()), "cols": len(per_year_cols[y])}
              for y in sorted(per_year_cols)]
    pd.DataFrame(shapes).to_csv(EDA_OUT / "per_year_shapes.csv", index=False)

    # 2) presencia de columnas
    all_cols = sorted({c for cols in per_year_cols.values() for c in cols})
    rows = []
    for y in sorted(per_year_cols):
        row = {"year": y}
        yset = set(per_year_cols[y])
        for c in all_cols:
            row[c] = int(c in yset)
        rows.append(row)
    pd.DataFrame(rows).to_csv(EDA_OUT / "column_presence_by_year.csv", index=False)

    # 3) nulos por año
    miss_rows = []
    for y in sorted(per_year_cols):
        sub = full[full["year"] == y]
        m = sub.isna().mean(numeric_only=False).to_dict()
        m["year"] = y
        miss_rows.append(m)
    pd.DataFrame(miss_rows).sort_values("year").to_csv(EDA_OUT / "missingness_by_year.csv", index=False)

    # 4) stats del target + top/bottom
    stats = []
    if TARGET in full.columns:
        for y in sorted(per_year_cols):
            s = full.loc[full["year"] == y, TARGET].dropna()
            if len(s):
                stats.append({
                    "year": y, "n": int(len(s)), "mean": float(s.mean()),
                    "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                    "min": float(s.min()), "q1": float(s.quantile(0.25)),
                    "median": float(s.median()), "q3": float(s.quantile(0.75)), "max": float(s.max())
                })
            sub = full[full["year"] == y]
            if {"country", TARGET}.issubset(sub.columns):
                ssub = sub[["country", TARGET]].dropna()
                ssub.sort_values(TARGET, ascending=False).head(10).to_csv(EDA_OUT / f"top10_{y}.csv", index=False)
                ssub.sort_values(TARGET, ascending=True).head(10).to_csv(EDA_OUT / f"bottom10_{y}.csv", index=False)
    pd.DataFrame(stats).to_csv(EDA_OUT / "happiness_score_stats_by_year.csv", index=False)

    # 5) correlación numérica global
    num_cols = full.select_dtypes(include="number").columns.tolist()
    if len(num_cols) > 1:
        corr = full[num_cols].corr(numeric_only=True)
        corr.to_csv(EDA_OUT / "correlation_numeric_all_years.csv", index=True)

    # 6) tendencia promedio anual
    if TARGET in full.columns:
        trend = full.groupby("year")[TARGET].mean().reset_index(name="mean_happiness_score")
        trend.to_csv(EDA_OUT / "trend_mean_score_by_year.csv", index=False)
    
        # === NUEVAS GRÁFICAS (se guardan en data/eda_out/charts) ===
    try:
        # 1) Heatmap de correlación global (numéricas)
        num_cols = full.select_dtypes(include="number").columns.tolist()
        if len(num_cols) > 1:
            corr = full[num_cols].corr(numeric_only=True)
            # ya guardaste el CSV arriba; ahora guardamos la imagen:
            fig = plt.figure(figsize=(10, 8))
            plt.imshow(corr.values, aspect="auto")
            plt.title("Correlation Heatmap (All Years, numeric columns)")
            plt.xticks(ticks=np.arange(len(num_cols)), labels=num_cols, rotation=90)
            plt.yticks(ticks=np.arange(len(num_cols)), labels=num_cols)
            plt.colorbar()
            plt.tight_layout()
            fig.savefig(CHARTS / "correlation_heatmap_all_years.png", dpi=160)
            plt.close(fig)
    except Exception as e:
        print(f"[EDA] Heatmap correlación: {e}")

    try:
        # 2) Boxplot de happiness_score por año (comparativa)
        if TARGET in full.columns and "year" in full.columns:
            sub = full[[TARGET, "year"]].dropna()
            if len(sub):
                # aseguramos orden por año
                years_sorted = sorted(sub["year"].dropna().unique().tolist())
                data_by_year = [sub.loc[sub["year"] == y, TARGET].values for y in years_sorted]
                fig = plt.figure(figsize=(10, 6))
                plt.boxplot(data_by_year, labels=years_sorted, showfliers=False)
                plt.title("Happiness Score by Year (Boxplot)")
                plt.xlabel("Year")
                plt.ylabel("Happiness Score")
                plt.tight_layout()
                fig.savefig(CHARTS / "boxplot_happiness_by_year.png", dpi=160)
                plt.close(fig)
    except Exception as e:
        print(f"[EDA] Boxplot por año: {e}")

    try:
        # 3) Tendencia del promedio anual (gráfico)
        if TARGET in full.columns and "year" in full.columns:
            trend = full.groupby("year")[TARGET].mean().reset_index(name="mean_happiness_score")
            if len(trend):
                fig = plt.figure(figsize=(9, 5))
                plt.plot(trend["year"], trend["mean_happiness_score"], marker="o")
                plt.title("Mean Happiness Score by Year (2015–2019)")
                plt.xlabel("Year")
                plt.ylabel("Mean Happiness Score")
                plt.grid(True)
                plt.tight_layout()
                fig.savefig(CHARTS / "trend_mean_score_by_year.png", dpi=160)
                plt.close(fig)
    except Exception as e:
        print(f"[EDA] Tendencia anual (png): {e}")

    try:
        # 4) Bottom-10 por año (grilla comparativa)
        if {"country", TARGET, "year"}.issubset(full.columns):
            years_sorted = sorted(per_year_cols.keys())
            n = len(years_sorted)
            # grilla 2x3 (queda un hueco vacío si solo son 5)
            rows, cols = 2, 3
            fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
            axes = axes.flatten()

            for i, y in enumerate(years_sorted):
                ax = axes[i]
                suby = full[full["year"] == y][["country", TARGET]].dropna()
                if len(suby):
                    bot = suby.sort_values(TARGET, ascending=True).head(10)
                    ax.barh(bot["country"], bot[TARGET])
                    ax.set_title(f"Bottom 10 – {y}")
                    ax.set_xlabel("Happiness Score")
                    ax.invert_yaxis()  # el más bajo arriba
                else:
                    ax.set_title(f"Bottom 10 – {y} (no data)")
                    ax.axis("off")

            # ocultar subplot sobrante si aplica
            for j in range(i + 1, rows * cols):
                axes[j].axis("off")

            plt.tight_layout()
            fig.savefig(CHARTS / "bottom10_grid.png", dpi=160)
            plt.close(fig)
    except Exception as e:
        print(f"[EDA] Bottom10 grid: {e}")


# -------------------------------------------------
# Selección simple de features basada en reglas
# -------------------------------------------------
def select_features(full: pd.DataFrame, per_year_cols: Dict[int, List[str]]) -> List[str]:
    if TARGET not in full.columns:
        return []

    presence = pd.DataFrame(
        [{"year": y, **{c: 1 for c in per_year_cols[y]}} for y in per_year_cols]
    ).fillna(0).set_index("year")
    present_in_years = presence.sum(axis=0)

    # --- FIX: evitar colisión de 'year' al hacer reset_index()
    cols_wo_year = [c for c in full.columns if c != "year"]
    miss_long = (
        full[cols_wo_year]
        .isna()
        .groupby(full["year"])
        .mean()
        .reset_index()  # ahora crea 'year' sin duplicarlo
    )
    miss_long = miss_long.melt(id_vars="year", var_name="column", value_name="pct_null")
    miss_med = miss_long.groupby("column")["pct_null"].median()

    corr_abs = {}
    for c in full.select_dtypes(include="number").columns:
        if c == TARGET:
            continue
        avail = full[[TARGET, c]].dropna()
        if len(avail) > 10:
            corr_abs[c] = abs(avail.corr().loc[TARGET, c])
    corr_abs = pd.Series(corr_abs, name="corr_with_target")

    rules = pd.DataFrame({
        "present_in_years": present_in_years,
        "median_missing": miss_med,
        "abs_corr_with_target": corr_abs
    }).fillna({"median_missing": 1.0, "abs_corr_with_target": 0.0})

    selected = rules[
        (rules["present_in_years"] >= 4) &
        (rules["median_missing"] <= 0.20) &
        (rules["abs_corr_with_target"] >= 0.20)
    ].sort_values(["abs_corr_with_target","present_in_years"], ascending=[False, False])

    drop_cols = {"year"}
    return [c for c in selected.index.tolist() if c not in drop_cols]

# -------------------------------------------------
# Entrenamiento con Ridge + CV y guardado de artefactos
# -------------------------------------------------
def train_model(df: pd.DataFrame, features: List[str]) -> dict:
    """
    Split 70/30 con flags, imputación por media (aprendida en TRAIN),
    búsqueda de alpha con RidgeCV (MAE), evaluación en TEST y guardado
    de dataset/feature list/modelo/métricas.
    """
    # 1) Split 70/30 + flags + label
    df = df.dropna(subset=[TARGET]).copy()
    tr_idx, te_idx = train_test_split(df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    df["is_train"] = 0; df.loc[tr_idx, "is_train"] = 1
    df["is_test"]  = 1 - df["is_train"]
    df["actual"]   = df[TARGET]

    train = df[df["is_train"] == 1].copy()
    test  = df[df["is_test"]  == 1].copy()

    # 2) Imputación por media aprendida en TRAIN
    means = train[features].mean().to_dict()
    for c in features:
        train[c] = pd.to_numeric(train[c], errors="coerce").fillna(means[c])
        test[c]  = pd.to_numeric(test[c],  errors="coerce").fillna(means[c])

    Xtr, ytr = train[features], train[TARGET]
    Xte, yte = test[features].values,  test[TARGET].values

    # 3) Ridge con selección de alpha vía CV (scoring MAE)
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    alpha_grid = np.arange(0.01, 1.00, 0.05)
    ridge_cv = RidgeCV(alphas=alpha_grid, cv=cv, scoring='neg_mean_absolute_error')
    ridge_cv.fit(Xtr, ytr)
    best_alpha = float(ridge_cv.alpha_)

    mdl = Ridge(alpha=best_alpha)
    mdl.fit(Xtr, ytr)

    # 4) Métricas en TEST
    y_pred = mdl.predict(Xte)
    mae = mean_absolute_error(yte, y_pred)

    # RMSE compatible con cualquier versión
    if _HAS_SK_RMSE:
        rmse = float(sk_rmse(yte, y_pred))
    else:
        rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))

    r2 = r2_score(yte, y_pred)


    print(f"[RIDGE] alpha*={best_alpha:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f} | R²={r2:.3f} | n_test={len(yte)}")

    # 5) Guardados estándar (compatibles con producer/consumer)
    df.to_csv(ART / "dataset_unified.csv", index=False)
    pd.Series(features, name="feature").to_csv(ART / "features_used.csv", index=False)
    joblib.dump({"model": mdl, "features": features, "means": means, "alpha": best_alpha}, MODELS / "model.pkl")
    with open(ART / "metrics.json", "w") as f:
        json.dump({
            "alpha": best_alpha,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "n_test": int(len(yte)),
            "features": features
        }, f, indent=2)

    # (Opcional) Diagnóstico y_pred vs y_true
    try:
        diag = ART / "diag_y_pred_vs_y_true.png"
        plt.figure()
        plt.scatter(yte, y_pred, s=12)
        lo, hi = float(min(yte.min(), y_pred.min())), float(max(yte.max(), y_pred.max()))
        plt.plot([lo, hi], [lo, hi], linestyle='--')
        plt.title('y_pred vs y_true (TEST)')
        plt.xlabel('y_true')
        plt.ylabel('y_pred')
        plt.tight_layout()
        plt.savefig(diag, dpi=140)
        plt.close()
    except Exception:
        pass

    return {"alpha": best_alpha, "mae": mae, "rmse": rmse, "r2": r2, "features": features}

# -------------------------------------------------
# MAIN: ETL + EDA + SPLIT + MODEL
# -------------------------------------------------
def main():
    # ETL: cargar y unificar
    dfs = []
    per_year_cols: Dict[int, List[str]] = {}
    for f in FILES:
        if not f.exists():
            raise FileNotFoundError(f"Falta el archivo: {f}")
        y = int(f.stem)
        d = pd.read_csv(f)
        d = canonicalize(d, year_hint=y)
        per_year_cols[y] = list(d.columns)
        dfs.append(d)
    full = pd.concat(dfs, ignore_index=True)

    # EDA
    run_eda(full, per_year_cols)

    # Selección de features (fallback si queda vacío)
    features = select_features(full, per_year_cols)
    if not features:
        candidates = [
            "gdp_per_capita","social_support","healthy_life_expectancy",
            "freedom","generosity","perceptions_of_corruption"
        ]
        features = [c for c in candidates if c in full.columns]

    print(f"[INFO] Features seleccionadas: {features}")

    # Split + modelo + guardados
    _ = train_model(full, features)

    print("\n[OK] Pipeline offline completo: ETL + EDA + split 70/30 + modelo (Ridge CV)")

if __name__ == "__main__":
    main()
