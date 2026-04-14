"""
DataPilot Backend — FastAPI + Pandas + scikit-learn
Real data science. Not fake ML.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import json
import warnings
warnings.filterwarnings("ignore")

# ── ML imports ──────────────────────────────────────
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import scipy.stats as stats

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

app = FastAPI(title="DataPilot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ──────────────────────────
sessions: dict[str, pd.DataFrame] = {}
session_meta: dict[str, dict] = {}

def get_df(session_id: str) -> pd.DataFrame:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Upload a file first.")
    return sessions[session_id]

def safe_json(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return safe_json(obj.tolist())
    if isinstance(obj, pd.Series):
        return safe_json(obj.tolist())
    if isinstance(obj, float):
        return None if np.isnan(obj) or np.isinf(obj) else obj
    return obj


# ═══════════════════════════════════════════════════
# 1. /upload — ingest CSV / Excel
# ═══════════════════════════════════════════════════
@app.post("/upload")
async def upload(file: UploadFile = File(...), session_id: str = "default"):
    content = await file.read()
    fname = file.filename.lower()

    try:
        if fname.endswith(".csv"):
            # Try multiple encodings
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(StringIO(content.decode(enc)))
                    break
                except Exception:
                    continue
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(content))
        elif fname.endswith(".json"):
            df = pd.read_json(StringIO(content.decode("utf-8")))
        elif fname.endswith(".tsv"):
            df = pd.read_csv(StringIO(content.decode("utf-8")), sep="\t")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV, Excel, JSON, or TSV.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Parse error: {str(e)}")

    # Auto-detect and convert types
    for col in df.columns:
        # Try numeric
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col].str.replace(",", "").str.strip())
            except Exception:
                pass
        # Try datetime
        if df[col].dtype == object:
            try:
                converted = pd.to_datetime(df[col], infer_format=True, errors="coerce")
                if converted.notna().sum() > len(df) * 0.6:
                    df[col] = converted
            except Exception:
                pass

    sessions[session_id] = df
    session_meta[session_id] = {
        "filename": file.filename,
        "rows": len(df),
        "cols": len(df.columns)
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "preview": safe_json(df.head(5).where(pd.notnull(df.head(5)), None).to_dict(orient="records")),
        "message": f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns"
    }


# ═══════════════════════════════════════════════════
# 2. /profile — deep column profiling
# ═══════════════════════════════════════════════════
@app.get("/profile")
async def profile(session_id: str = "default"):
    df = get_df(session_id)
    result = []

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        null_count = int(series.isna().sum())
        null_pct = round(null_count / len(df) * 100, 2)
        unique_count = int(series.nunique(dropna=True))
        unique_pct = round(unique_count / len(df) * 100, 2)

        col_info = {
            "column": col,
            "dtype": dtype,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_count,
            "unique_pct": unique_pct,
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                outlier_low = q1 - 1.5 * iqr
                outlier_high = q3 + 1.5 * iqr
                outlier_count = int(((clean < outlier_low) | (clean > outlier_high)).sum())
                _, normality_p = stats.shapiro(clean.sample(min(len(clean), 500), random_state=42))
                skew = float(clean.skew())
                kurtosis = float(clean.kurtosis())
                col_info.update({
                    "kind": "numeric",
                    "mean": safe_json(float(clean.mean())),
                    "median": safe_json(float(clean.median())),
                    "std": safe_json(float(clean.std())),
                    "min": safe_json(float(clean.min())),
                    "max": safe_json(float(clean.max())),
                    "q1": safe_json(float(q1)),
                    "q3": safe_json(float(q3)),
                    "iqr": safe_json(float(iqr)),
                    "sum": safe_json(float(clean.sum())),
                    "outlier_count": outlier_count,
                    "outlier_pct": round(outlier_count / len(clean) * 100, 2),
                    "skewness": safe_json(skew),
                    "kurtosis": safe_json(kurtosis),
                    "is_normal": bool(normality_p > 0.05),
                    "normality_p": safe_json(float(normality_p)),
                    "cv": safe_json(float(clean.std() / clean.mean()) if clean.mean() != 0 else None),
                    "histogram": safe_json(
                        np.histogram(clean, bins=20)[0].tolist()
                    ),
                    "histogram_edges": safe_json(
                        np.histogram(clean, bins=20)[1].tolist()
                    ),
                })
        elif pd.api.types.is_datetime64_any_dtype(series):
            clean = series.dropna()
            col_info.update({
                "kind": "datetime",
                "min": str(clean.min()) if len(clean) else None,
                "max": str(clean.max()) if len(clean) else None,
                "range_days": int((clean.max() - clean.min()).days) if len(clean) > 1 else 0,
            })
        else:
            vc = series.dropna().value_counts()
            col_info.update({
                "kind": "categorical",
                "top_values": safe_json(vc.head(10).to_dict()),
                "top_value": str(vc.index[0]) if len(vc) > 0 else None,
                "top_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "top_pct": round(float(vc.iloc[0]) / len(df) * 100, 2) if len(vc) > 0 else 0,
                "entropy": safe_json(float(stats.entropy(vc.values))),
            })

        result.append(col_info)

    # Correlations between all numeric columns
    num_df = df.select_dtypes(include=[np.number])
    corr_matrix = {}
    if len(num_df.columns) >= 2:
        corr = num_df.corr()
        corr_matrix = safe_json(corr.where(pd.notnull(corr), None).to_dict())

    dup_count = int(df.duplicated().sum())

    return {
        "columns": safe_json(result),
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "num_cols": num_df.columns.tolist(),
        "cat_cols": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "date_cols": df.select_dtypes(include=["datetime64"]).columns.tolist(),
        "duplicate_rows": dup_count,
        "duplicate_pct": round(dup_count / len(df) * 100, 2),
        "correlations": corr_matrix,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
    }


# ═══════════════════════════════════════════════════
# 3. /clean — real data cleaning operations
# ═══════════════════════════════════════════════════
class CleanRequest(BaseModel):
    session_id: str = "default"
    operations: list[dict]  # [{op: "drop_duplicates"}, {op: "impute", col: "age", method: "median"}, ...]

@app.post("/clean")
async def clean(req: CleanRequest):
    df = get_df(req.session_id).copy()
    log = []
    initial_rows = len(df)
    initial_nulls = int(df.isna().sum().sum())

    for op_def in req.operations:
        op = op_def.get("op")

        if op == "drop_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            log.append(f"✅ Removed {removed:,} duplicate rows")

        elif op == "drop_nulls":
            col = op_def.get("col")
            before = len(df)
            if col:
                df = df.dropna(subset=[col])
                log.append(f"✅ Dropped {before - len(df):,} rows with null '{col}'")
            else:
                thresh = op_def.get("thresh", 0.5)
                col_thresh = int(len(df.columns) * thresh)
                df = df.dropna(thresh=col_thresh)
                log.append(f"✅ Dropped {before - len(df):,} rows with >{int(thresh*100)}% nulls")

        elif op == "impute":
            col = op_def.get("col")
            method = op_def.get("method", "median")
            if col and col in df.columns:
                null_count = df[col].isna().sum()
                if pd.api.types.is_numeric_dtype(df[col]):
                    if method == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    elif method == "median":
                        df[col] = df[col].fillna(df[col].median())
                    elif method == "zero":
                        df[col] = df[col].fillna(0)
                    elif method == "forward":
                        df[col] = df[col].ffill()
                    elif method == "knn":
                        from sklearn.impute import KNNImputer
                        imp = KNNImputer(n_neighbors=5)
                        df[[col]] = imp.fit_transform(df[[col]])
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) else "Unknown")
                log.append(f"✅ Imputed {null_count:,} nulls in '{col}' using {method}")

        elif op == "remove_outliers":
            col = op_def.get("col")
            method = op_def.get("method", "iqr")
            if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                before = len(df)
                if method == "iqr":
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q3 - q1
                    df = df[(df[col] >= q1 - 3 * iqr) & (df[col] <= q3 + 3 * iqr)]
                elif method == "zscore":
                    z = np.abs(stats.zscore(df[col].dropna()))
                    df = df[z < 3]
                log.append(f"✅ Removed {before - len(df):,} outliers from '{col}' ({method})")

        elif op == "normalize":
            col = op_def.get("col")
            method = op_def.get("method", "minmax")
            if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if method == "minmax":
                    mn, mx = df[col].min(), df[col].max()
                    df[col] = (df[col] - mn) / (mx - mn) if mx != mn else 0
                elif method == "zscore":
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
                elif method == "log":
                    df[col] = np.log1p(df[col].clip(lower=0))
                log.append(f"✅ Normalized '{col}' using {method}")

        elif op == "drop_column":
            col = op_def.get("col")
            if col and col in df.columns:
                df = df.drop(columns=[col])
                log.append(f"✅ Dropped column '{col}'")

        elif op == "rename_column":
            old = op_def.get("old")
            new = op_def.get("new")
            if old and new and old in df.columns:
                df = df.rename(columns={old: new})
                log.append(f"✅ Renamed '{old}' → '{new}'")

        elif op == "create_column":
            name = op_def.get("name")
            expr = op_def.get("expr")  # e.g. "revenue - cost"
            if name and expr:
                try:
                    df[name] = df.eval(expr)
                    log.append(f"✅ Created column '{name}' = {expr}")
                except Exception as e:
                    log.append(f"❌ Failed to create '{name}': {str(e)}")

        elif op == "filter_rows":
            expr = op_def.get("expr")
            if expr:
                try:
                    before = len(df)
                    df = df.query(expr)
                    log.append(f"✅ Filtered to {len(df):,} rows (removed {before - len(df):,}) via: {expr}")
                except Exception as e:
                    log.append(f"❌ Filter failed: {str(e)}")

        elif op == "cast_type":
            col = op_def.get("col")
            to_type = op_def.get("to_type")
            if col and to_type and col in df.columns:
                try:
                    if to_type == "numeric":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif to_type == "string":
                        df[col] = df[col].astype(str)
                    elif to_type == "datetime":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    elif to_type == "category":
                        df[col] = df[col].astype("category")
                    log.append(f"✅ Cast '{col}' to {to_type}")
                except Exception as e:
                    log.append(f"❌ Cast failed: {str(e)}")

    sessions[req.session_id] = df
    final_nulls = int(df.isna().sum().sum())

    return {
        "operations_applied": len(log),
        "log": log,
        "rows_before": initial_rows,
        "rows_after": len(df),
        "rows_removed": initial_rows - len(df),
        "nulls_before": initial_nulls,
        "nulls_after": final_nulls,
        "nulls_fixed": initial_nulls - final_nulls,
        "preview": safe_json(df.head(5).where(pd.notnull(df.head(5)), None).to_dict(orient="records")),
    }


# ═══════════════════════════════════════════════════
# 4. /analyze — real statistical analysis
# ═══════════════════════════════════════════════════
@app.get("/analyze")
async def analyze(session_id: str = "default"):
    df = get_df(session_id)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    results = {}

    # ── Correlation analysis ──────────────────────────
    if len(num_cols) >= 2:
        corr_df = df[num_cols].corr()
        # Find strongest pairs
        pairs = []
        for i, c1 in enumerate(num_cols):
            for j, c2 in enumerate(num_cols):
                if i < j:
                    r = corr_df.loc[c1, c2]
                    if not np.isnan(r):
                        pairs.append({
                            "col1": c1, "col2": c2,
                            "r": round(float(r), 4),
                            "r_squared": round(float(r**2), 4),
                            "strength": "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak",
                            "direction": "positive" if r > 0 else "negative"
                        })
        pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
        results["correlations"] = {
            "matrix": safe_json(corr_df.where(pd.notnull(corr_df), None).to_dict()),
            "top_pairs": safe_json(pairs[:15]),
        }

    # ── ANOVA: numeric vs categorical ────────────────
    if num_cols and cat_cols:
        anova_results = []
        for cat in cat_cols[:3]:
            for num in num_cols[:3]:
                groups = [group[num].dropna().values for _, group in df.groupby(cat) if len(group[num].dropna()) >= 3]
                if len(groups) >= 2:
                    try:
                        f_stat, p_val = stats.f_oneway(*groups)
                        anova_results.append({
                            "numeric_col": num,
                            "groupby_col": cat,
                            "f_statistic": safe_json(float(f_stat)),
                            "p_value": safe_json(float(p_val)),
                            "significant": bool(p_val < 0.05),
                        })
                    except Exception:
                        pass
        results["anova"] = safe_json(anova_results)

    # ── Distribution tests ────────────────────────────
    dist_tests = []
    for col in num_cols[:5]:
        clean = df[col].dropna()
        if len(clean) >= 8:
            sample = clean.sample(min(len(clean), 2000), random_state=42)
            stat, p = stats.shapiro(sample) if len(sample) <= 5000 else stats.kstest(
                (sample - sample.mean()) / sample.std(), "norm"
            )
            dist_tests.append({
                "col": col,
                "is_normal": bool(p > 0.05),
                "normality_p": safe_json(float(p)),
                "skewness": safe_json(float(clean.skew())),
                "kurtosis": safe_json(float(clean.kurtosis())),
                "recommended_test": "parametric" if p > 0.05 else "non-parametric",
            })
    results["distribution_tests"] = safe_json(dist_tests)

    # ── Time series decomposition ─────────────────────
    if date_cols and num_cols:
        ts_col = num_cols[0]
        date_col = date_cols[0]
        ts_df = df[[date_col, ts_col]].dropna().sort_values(date_col)
        ts_df = ts_df.set_index(date_col)[ts_col].resample("ME").mean().dropna()
        if len(ts_df) >= 12 and HAS_STATSMODELS:
            try:
                decomp = seasonal_decompose(ts_df, model="additive", period=min(12, len(ts_df)//2))
                results["time_series"] = {
                    "col": ts_col,
                    "periods": len(ts_df),
                    "trend": safe_json(decomp.trend.dropna().tolist()),
                    "seasonal": safe_json(decomp.seasonal.dropna().tolist()),
                    "residual": safe_json(decomp.resid.dropna().tolist()),
                    "dates": [str(d) for d in decomp.trend.dropna().index],
                    "trend_direction": "up" if decomp.trend.dropna().iloc[-1] > decomp.trend.dropna().iloc[0] else "down",
                }
            except Exception as e:
                results["time_series"] = {"error": str(e)}

    # ── Chi-square tests between categoricals ─────────
    chi_results = []
    if len(cat_cols) >= 2:
        for i, c1 in enumerate(cat_cols[:3]):
            for j, c2 in enumerate(cat_cols[:3]):
                if i < j:
                    contingency = pd.crosstab(df[c1], df[c2])
                    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                        try:
                            chi2, p, dof, _ = stats.chi2_contingency(contingency)
                            n = contingency.values.sum()
                            cramers_v = float(np.sqrt(chi2 / (n * (min(contingency.shape) - 1))))
                            chi_results.append({
                                "col1": c1, "col2": c2,
                                "chi2": safe_json(float(chi2)),
                                "p_value": safe_json(float(p)),
                                "cramers_v": safe_json(cramers_v),
                                "significant": bool(p < 0.05),
                                "effect": "strong" if cramers_v > 0.3 else "moderate" if cramers_v > 0.1 else "weak",
                            })
                        except Exception:
                            pass
    results["chi_square"] = safe_json(chi_results)

    return {"session_id": session_id, "analysis": safe_json(results)}


# ═══════════════════════════════════════════════════
# 5. /train-model — real ML training
# ═══════════════════════════════════════════════════
class TrainRequest(BaseModel):
    session_id: str = "default"
    model_type: str  # "linear_regression", "logistic", "random_forest_clf", "random_forest_reg", "kmeans", "gradient_boosting"
    target_col: str = None
    feature_cols: list[str] = None
    params: dict = {}

@app.post("/train-model")
async def train_model(req: TrainRequest):
    df = get_df(req.session_id).copy()

    # ── Feature prep ──────────────────────────────────
    if req.feature_cols:
        feat_cols = [c for c in req.feature_cols if c in df.columns]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feat_cols = [c for c in num_cols if c != req.target_col][:10]

    if not feat_cols:
        raise HTTPException(status_code=400, detail="No valid feature columns found.")

    X = df[feat_cols].copy()

    # Label encode categoricals in features
    le_map = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_map[col] = le.classes_.tolist()

    # Impute missing
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=feat_cols)

    # ── CLUSTERING (no target needed) ────────────────
    if req.model_type == "kmeans":
        k = int(req.params.get("k", 3))

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow method
        inertias = []
        sil_scores = []
        k_range = range(2, min(9, len(df)//2))
        for ki in k_range:
            km_temp = KMeans(n_clusters=ki, random_state=42, n_init=10)
            lbls = km_temp.fit_predict(X_scaled)
            inertias.append(float(km_temp.inertia_))
            if ki <= 6:
                sil_scores.append(float(silhouette_score(X_scaled, lbls, sample_size=min(2000, len(X_scaled)))))

        # Fit final
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_scaled))))

        # Cluster profiles
        df_cl = X.copy()
        df_cl["_cluster"] = labels
        profiles = []
        for ci in range(k):
            grp = df_cl[df_cl["_cluster"] == ci]
            profiles.append({
                "cluster": int(ci),
                "size": int(len(grp)),
                "pct": round(float(len(grp) / len(df) * 100), 1),
                "means": safe_json(grp[feat_cols].mean().to_dict()),
            })

        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        return {
            "model_type": "kmeans",
            "k": k,
            "silhouette_score": round(sil, 4),
            "inertia": float(km.inertia_),
            "cluster_labels": safe_json(labels.tolist()),
            "cluster_profiles": safe_json(profiles),
            "pca_coords": safe_json(coords.tolist()),
            "pca_variance_explained": safe_json(pca.explained_variance_ratio_.tolist()),
            "elbow_inertias": safe_json(inertias),
            "elbow_k": list(k_range),
            "silhouette_scores": safe_json(sil_scores),
            "feature_cols": feat_cols,
            "interpretation": f"Silhouette score {sil:.3f} — {'excellent' if sil > 0.7 else 'good' if sil > 0.5 else 'fair' if sil > 0.3 else 'poor'} cluster separation",
        }

    # ── SUPERVISED MODELS ────────────────────────────
    if not req.target_col or req.target_col not in df.columns:
        raise HTTPException(status_code=400, detail="target_col is required for supervised models.")

    y_raw = df[req.target_col]

    # Detect task type
    is_classification = req.model_type in ("logistic", "random_forest_clf") or (
        req.model_type == "auto" and (y_raw.dtype == object or y_raw.nunique() <= 20)
    )

    if is_classification:
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw.astype(str))
    else:
        y = pd.to_numeric(y_raw, errors="coerce").values

    # Align rows
    mask = ~np.isnan(y.astype(float)) if not is_classification else np.ones(len(y), dtype=bool)
    X_clean = X[mask]
    y_clean = y[mask]

    if len(X_clean) < 10:
        raise HTTPException(status_code=400, detail="Too few rows after cleaning for model training.")

    test_size = float(req.params.get("test_size", 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=42,
        stratify=y_clean if is_classification and len(np.unique(y_clean)) < 20 else None
    )

    # ── Scale for linear models ───────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Regression ───────────────────────────────────
    if req.model_type in ("linear_regression", "ridge"):
        alpha = float(req.params.get("alpha", 1.0))
        model = Ridge(alpha=alpha) if req.model_type == "ridge" else LinearRegression()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        # Cross-validation
        cv_scores = cross_val_score(model, scaler.fit_transform(X_clean), y_clean, cv=5, scoring="r2")

        coef = dict(zip(feat_cols, model.coef_.tolist()))
        coef_sorted = sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "model_type": req.model_type,
            "target_col": req.target_col,
            "feature_cols": feat_cols,
            "train_size": len(X_train),
            "test_size_n": len(X_test),
            "metrics": {
                "r2_score": round(r2, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "cv_r2_mean": round(float(cv_scores.mean()), 4),
                "cv_r2_std": round(float(cv_scores.std()), 4),
            },
            "coefficients": safe_json(dict(coef_sorted[:15])),
            "intercept": safe_json(float(model.intercept_)),
            "actual_vs_predicted": safe_json([
                {"actual": float(a), "predicted": float(p)}
                for a, p in zip(y_test[:100], y_pred[:100])
            ]),
            "interpretation": f"R² = {r2:.3f} — model explains {r2*100:.1f}% of variance in {req.target_col}. RMSE = {rmse:.3f}.",
        }

    # ── Random Forest Regressor ───────────────────────
    elif req.model_type == "random_forest_reg":
        n_est = int(req.params.get("n_estimators", 100))
        model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1, max_features="sqrt")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        feat_imp = dict(zip(feat_cols, model.feature_importances_.tolist()))
        feat_imp_sorted = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)

        return {
            "model_type": "random_forest_regressor",
            "target_col": req.target_col,
            "feature_cols": feat_cols,
            "train_size": len(X_train),
            "test_size_n": len(X_test),
            "metrics": {
                "r2_score": round(r2, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
            },
            "feature_importance": safe_json(dict(feat_imp_sorted[:15])),
            "actual_vs_predicted": safe_json([
                {"actual": float(a), "predicted": float(p)}
                for a, p in zip(y_test[:100], y_pred[:100])
            ]),
            "interpretation": f"R² = {r2:.3f} — {r2*100:.1f}% variance explained. Top feature: {feat_imp_sorted[0][0]}.",
        }

    # ── Classification ────────────────────────────────
    elif req.model_type in ("logistic", "random_forest_clf"):
        if req.model_type == "logistic":
            model = LogisticRegression(max_iter=1000, random_state=42, C=float(req.params.get("C", 1.0)))
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
        else:
            n_est = int(req.params.get("n_estimators", 100))
            model = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        classes = le_y.classes_.tolist() if is_classification else []
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Feature importance
        if req.model_type == "logistic":
            if len(classes) == 2:
                imp = dict(zip(feat_cols, np.abs(model.coef_[0]).tolist()))
            else:
                imp = dict(zip(feat_cols, np.abs(model.coef_).mean(axis=0).tolist()))
        else:
            imp = dict(zip(feat_cols, model.feature_importances_.tolist()))

        imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)

        return {
            "model_type": req.model_type,
            "target_col": req.target_col,
            "feature_cols": feat_cols,
            "classes": classes,
            "train_size": len(X_train),
            "test_size_n": len(X_test),
            "metrics": {
                "accuracy": round(acc, 4),
                "weighted_f1": round(float(report.get("weighted avg", {}).get("f1-score", 0)), 4),
                "weighted_precision": round(float(report.get("weighted avg", {}).get("precision", 0)), 4),
                "weighted_recall": round(float(report.get("weighted avg", {}).get("recall", 0)), 4),
            },
            "per_class_report": safe_json({
                k: v for k, v in report.items()
                if k not in ("accuracy", "macro avg", "weighted avg") and isinstance(v, dict)
            }),
            "confusion_matrix": safe_json(cm),
            "feature_importance": safe_json(dict(imp_sorted[:15])),
            "interpretation": f"Accuracy = {acc*100:.1f}%. {'Binary' if len(classes)==2 else f'{len(classes)}-class'} classification on '{req.target_col}'.",
        }

    elif req.model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=int(req.params.get("n_estimators", 100)),
            learning_rate=float(req.params.get("learning_rate", 0.1)),
            max_depth=int(req.params.get("max_depth", 3)),
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        feat_imp = dict(zip(feat_cols, model.feature_importances_.tolist()))
        feat_imp_sorted = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)

        return {
            "model_type": "gradient_boosting",
            "target_col": req.target_col,
            "feature_cols": feat_cols,
            "train_size": len(X_train),
            "test_size_n": len(X_test),
            "metrics": {"r2_score": round(r2, 4), "rmse": round(rmse, 4)},
            "feature_importance": safe_json(dict(feat_imp_sorted[:15])),
            "actual_vs_predicted": safe_json([
                {"actual": float(a), "predicted": float(p)}
                for a, p in zip(y_test[:100], y_pred[:100])
            ]),
            "interpretation": f"Gradient Boosting R² = {r2:.3f}. Top predictor: {feat_imp_sorted[0][0]}.",
        }

    raise HTTPException(status_code=400, detail=f"Unknown model_type: {req.model_type}")


# ═══════════════════════════════════════════════════
# 6. /forecast — real ARIMA / ETS time series
# ═══════════════════════════════════════════════════
class ForecastRequest(BaseModel):
    session_id: str = "default"
    target_col: str
    date_col: str = None
    periods: int = 6
    method: str = "auto"  # "arima", "ets", "auto"
    freq: str = "ME"

@app.post("/forecast")
async def forecast(req: ForecastRequest):
    df = get_df(req.session_id).copy()

    if req.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_col}' not found.")

    # Build time series
    if req.date_col and req.date_col in df.columns:
        ts = df[[req.date_col, req.target_col]].dropna()
        ts[req.date_col] = pd.to_datetime(ts[req.date_col], errors="coerce")
        ts = ts.dropna(subset=[req.date_col])
        ts = ts.sort_values(req.date_col).set_index(req.date_col)[req.target_col]
        ts = ts.resample(req.freq).mean().dropna()
    else:
        # No date col — use integer index
        ts = df[req.target_col].dropna().reset_index(drop=True)
        ts.index = pd.date_range("2020-01", periods=len(ts), freq="ME")

    if len(ts) < 6:
        raise HTTPException(status_code=400, detail="Need at least 6 data points for forecasting.")

    if not HAS_STATSMODELS:
        # Fallback: simple exponential smoothing
        alpha = 0.3
        smoothed = [float(ts.iloc[0])]
        for v in ts.iloc[1:]:
            smoothed.append(alpha * float(v) + (1 - alpha) * smoothed[-1])

        slope = (smoothed[-1] - smoothed[0]) / max(len(smoothed) - 1, 1)
        future_vals = [smoothed[-1] + slope * (i + 1) for i in range(req.periods)]
        std_err = float(np.std(ts.values - smoothed[:len(ts)]) * 1.96)

        future_dates = pd.date_range(ts.index[-1], periods=req.periods + 1, freq=req.freq)[1:]
        return {
            "method": "simple_exponential_smoothing",
            "target_col": req.target_col,
            "periods": req.periods,
            "historical": [{"date": str(d), "value": safe_json(float(v))} for d, v in ts.items()],
            "forecast": [
                {"date": str(d), "value": safe_json(v), "lo": safe_json(v - std_err), "hi": safe_json(v + std_err)}
                for d, v in zip(future_dates, future_vals)
            ],
            "smoothed": [safe_json(v) for v in smoothed],
        }

    # ── ETS (Holt-Winters) ─────────────────────────
    results = {}
    seasonal_periods = min(12, len(ts) // 2)

    try:
        ets_model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add" if len(ts) >= 2 * seasonal_periods else None,
            seasonal_periods=seasonal_periods if len(ts) >= 2 * seasonal_periods else None,
            initialization_method="estimated",
        )
        ets_fit = ets_model.fit(optimized=True)
        ets_fc = ets_fit.forecast(req.periods)
        residuals = ts.values - ets_fit.fittedvalues.values
        std_err = float(np.std(residuals))

        future_dates = pd.date_range(ts.index[-1], periods=req.periods + 1, freq=req.freq)[1:]

        results["ets"] = {
            "aic": safe_json(float(ets_fit.aic)),
            "forecast": [
                {
                    "date": str(d),
                    "value": safe_json(float(v)),
                    "lo": safe_json(float(v) - 1.96 * std_err * np.sqrt(i + 1)),
                    "hi": safe_json(float(v) + 1.96 * std_err * np.sqrt(i + 1)),
                }
                for i, (d, v) in enumerate(zip(future_dates, ets_fc))
            ],
            "smoothed": safe_json(ets_fit.fittedvalues.tolist()),
            "params": safe_json({k: float(v) for k, v in ets_fit.params.items() if isinstance(v, (int, float))}),
        }
    except Exception as e:
        results["ets_error"] = str(e)

    # ── ARIMA ─────────────────────────────────────
    try:
        arima_model = SARIMAX(ts, order=(1, 1, 1), trend="c")
        arima_fit = arima_model.fit(disp=False)
        arima_fc = arima_fit.get_forecast(steps=req.periods)
        fc_mean = arima_fc.predicted_mean
        fc_ci = arima_fc.conf_int()
        future_dates = pd.date_range(ts.index[-1], periods=req.periods + 1, freq=req.freq)[1:]

        results["arima"] = {
            "aic": safe_json(float(arima_fit.aic)),
            "forecast": [
                {
                    "date": str(d),
                    "value": safe_json(float(v)),
                    "lo": safe_json(float(fc_ci.iloc[i, 0])),
                    "hi": safe_json(float(fc_ci.iloc[i, 1])),
                }
                for i, (d, v) in enumerate(zip(future_dates, fc_mean))
            ],
        }
    except Exception as e:
        results["arima_error"] = str(e)

    # Pick best
    if "ets" in results and "arima" in results:
        best = "ets" if results["ets"]["aic"] < results["arima"]["aic"] else "arima"
    elif "ets" in results:
        best = "ets"
    else:
        best = "arima"

    return {
        "method": best,
        "target_col": req.target_col,
        "periods": req.periods,
        "historical": [{"date": str(d), "value": safe_json(float(v))} for d, v in ts.items()],
        "forecast": results.get(best, {}).get("forecast", []),
        "smoothed": results.get("ets", {}).get("smoothed", []),
        "model_comparison": {
            k: {"aic": results[k].get("aic")} for k in results if "error" not in k
        },
        "best_model": best,
    }


# ═══════════════════════════════════════════════════
# 7. /query — natural language → data operations
# ═══════════════════════════════════════════════════
class QueryRequest(BaseModel):
    session_id: str = "default"
    query: str
    execute: bool = True

@app.post("/query")
async def nl_query(req: QueryRequest):
    """
    Parse natural language into data operations and execute them.
    This is real NL→Data, not explanation theater.
    """
    df = get_df(req.session_id)
    q = req.query.strip().lower()
    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    result_df = None
    explanation = ""
    operation = ""

    # ── Correlation ───────────────────────────────────
    import re
    corr_match = re.search(r"corr(?:elation)? (?:between |of )?(.+?) and (.+?)$", q)
    if corr_match:
        c1_raw, c2_raw = corr_match.group(1).strip(), corr_match.group(2).strip()
        c1 = next((c for c in cols if c.lower() == c1_raw), None)
        c2 = next((c for c in cols if c.lower() == c2_raw), None)
        if c1 and c2:
            r = float(df[c1].corr(df[c2]))
            operation = f"correlation({c1}, {c2})"
            explanation = f"Pearson correlation between {c1} and {c2}: r = {r:.4f}. {'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.4 else 'Weak'} {'positive' if r > 0 else 'negative'} relationship."
            return {"operation": operation, "result": {"r": round(r, 4)}, "explanation": explanation}

    # ── Filter rows ───────────────────────────────────
    filter_match = re.search(r"(?:filter|show|where|rows?) (?:where |with )?(.+?) (>|<|>=|<=|==|!=|=) (.+?)$", q)
    if filter_match:
        col_raw, op, val_raw = filter_match.group(1).strip(), filter_match.group(2), filter_match.group(3).strip()
        col = next((c for c in cols if c.lower() == col_raw), None)
        if col:
            try:
                val = float(val_raw) if pd.api.types.is_numeric_dtype(df[col]) else val_raw.strip('"\'')
                op_map = {"=": "=="}
                op_use = op_map.get(op, op)
                if pd.api.types.is_numeric_dtype(df[col]):
                    mask = df[col].map(lambda x: eval(f"x {op_use} {val}"))
                else:
                    mask = df[col].map(lambda x: eval(f"str(x).lower() {op_use} str(val).lower()"))
                result_df = df[mask]
                operation = f"filter({col} {op} {val})"
                explanation = f"Filtered to {len(result_df):,} rows where {col} {op} {val} (from {len(df):,} total)."
            except Exception as e:
                explanation = f"Filter parse error: {str(e)}"

    # ── Create new column ─────────────────────────────
    new_col_match = re.search(r"(?:create|add|new) column (.+?) (?:=|as) (.+?)$", q)
    if new_col_match and not result_df is not None:
        col_name, expr = new_col_match.group(1).strip(), new_col_match.group(2).strip()
        try:
            df_test = df.copy()
            df_test[col_name] = df_test.eval(expr)
            sessions[req.session_id] = df_test
            operation = f"create_column({col_name} = {expr})"
            explanation = f"Created column '{col_name}' = {expr}. Dataset now has {len(df_test.columns)} columns."
            return {"operation": operation, "result": {"new_col": col_name, "rows": len(df_test)}, "explanation": explanation}
        except Exception as e:
            explanation = f"Column creation error: {str(e)}"

    # ── Group by aggregation ──────────────────────────
    grp_match = re.search(r"(?:group|sum|average|mean|count|max|min) (.+?) by (.+?)$", q)
    if grp_match:
        val_raw, grp_raw = grp_match.group(1).strip(), grp_match.group(2).strip()
        val_col = next((c for c in num_cols if c.lower() == val_raw), num_cols[0] if num_cols else None)
        grp_col = next((c for c in cols if c.lower() == grp_raw), cat_cols[0] if cat_cols else None)
        if val_col and grp_col:
            agg_func = "sum" if "sum" in q else "mean" if ("average" in q or "mean" in q) else "count" if "count" in q else "max" if "max" in q else "min"
            result_df = df.groupby(grp_col)[val_col].agg(agg_func).reset_index().sort_values(val_col, ascending=False)
            operation = f"groupby({grp_col}).{agg_func}({val_col})"
            explanation = f"{agg_func.title()} of {val_col} grouped by {grp_col}. Showing {len(result_df)} groups."

    # ── Top N ─────────────────────────────────────────
    top_match = re.search(r"top (\d+) (?:rows? )?(?:by |sorted by |with highest )?(.+?)$", q)
    if top_match and result_df is None:
        n, col_raw = int(top_match.group(1)), top_match.group(2).strip()
        col = next((c for c in num_cols if c.lower() == col_raw), num_cols[0] if num_cols else None)
        if col:
            result_df = df.nlargest(n, col)
            operation = f"top_{n}({col})"
            explanation = f"Top {n} rows by {col}. Range: {float(result_df[col].min()):.2f} – {float(result_df[col].max()):.2f}."

    # ── Describe / summary ────────────────────────────
    if any(word in q for word in ["describe", "summary", "stats", "statistics", "summarize"]):
        col_found = next((c for c in cols if c.lower() in q), None)
        if col_found:
            desc = df[col_found].describe()
            operation = f"describe({col_found})"
            explanation = f"Statistics for {col_found}."
            return {"operation": operation, "result": safe_json(desc.to_dict()), "explanation": explanation}
        else:
            desc = df.describe()
            operation = "describe(all)"
            explanation = f"Summary statistics for all {len(num_cols)} numeric columns."
            return {"operation": operation, "result": safe_json(desc.to_dict()), "explanation": explanation}

    if result_df is not None:
        preview = safe_json(result_df.head(50).where(pd.notnull(result_df.head(50)), None).to_dict(orient="records"))
        return {
            "operation": operation,
            "result": {
                "rows": len(result_df),
                "cols": len(result_df.columns),
                "data": preview,
                "columns": result_df.columns.tolist(),
            },
            "explanation": explanation,
        }

    return {
        "operation": "unknown",
        "result": None,
        "explanation": f"Could not parse query: '{req.query}'. Try: 'top 10 by revenue', 'filter age > 30', 'correlation between X and Y', 'create column profit = revenue - cost', 'group revenue by category'.",
        "suggestions": [
            "top 10 by [column]",
            "filter [column] > [value]",
            "correlation between [col1] and [col2]",
            "sum [column] by [category]",
            "create column [name] = [expression]",
            "describe [column]",
        ],
    }


# ═══════════════════════════════════════════════════
# 8. /export — download cleaned data
# ═══════════════════════════════════════════════════
from fastapi.responses import StreamingResponse

@app.get("/export")
async def export_data(session_id: str = "default", format: str = "csv"):
    df = get_df(session_id)
    if format == "csv":
        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=datapilot_export.csv"}
        )
    elif format == "json":
        return JSONResponse(safe_json(df.to_dict(orient="records")))
    raise HTTPException(status_code=400, detail="Supported formats: csv, json")


# ═══════════════════════════════════════════════════
# Utility endpoints
# ═══════════════════════════════════════════════════
@app.get("/sessions")
async def list_sessions():
    return {"sessions": list(sessions.keys()), "meta": session_meta}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        session_meta.pop(session_id, None)
        return {"deleted": session_id}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "active_sessions": len(sessions),
        "statsmodels": HAS_STATSMODELS,
        "sklearn": True,
    }

@app.get("/")
async def root():
    return {"name": "DataPilot API", "version": "2.0.0", "docs": "/docs"}

# ═══════════════════════════════════════════════════
# 9. /eda — chart-ready data for the frontend
# ═══════════════════════════════════════════════════
@app.get("/eda")
async def eda(session_id: str = "default", max_scatter: int = 500):
    df = get_df(session_id)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    out = {}

    # ── Histograms for every numeric col ─────────────
    histograms = {}
    for col in num_cols[:8]:
        clean = df[col].dropna()
        if len(clean) < 3:
            continue
        counts, edges = np.histogram(clean, bins=25)
        mean_v = float(clean.mean())
        std_v = float(clean.std()) or 1
        # KDE overlay (Gaussian approx)
        centers = [(float(edges[i]) + float(edges[i+1])) / 2 for i in range(len(edges)-1)]
        kde = [
            float(len(clean)) * float(edges[1] - edges[0]) *
            np.exp(-0.5 * ((x - mean_v) / std_v) ** 2) / (std_v * np.sqrt(2 * np.pi))
            for x in centers
        ]
        histograms[col] = {
            "counts": safe_json(counts.tolist()),
            "edges": safe_json(edges.tolist()),
            "centers": safe_json(centers),
            "kde": safe_json(kde),
            "mean": safe_json(mean_v),
            "median": safe_json(float(clean.median())),
            "std": safe_json(std_v),
            "n": int(len(clean)),
        }
    out["histograms"] = histograms

    # ── Scatter pairs (first 3 numeric cols) ─────────
    scatters = []
    sample = df.sample(min(max_scatter, len(df)), random_state=42)
    for i in range(min(len(num_cols), 4)):
        for j in range(i + 1, min(len(num_cols), 4)):
            cx, cy = num_cols[i], num_cols[j]
            pts = sample[[cx, cy]].dropna()
            r = float(pts[cx].corr(pts[cy]))
            # regression line
            if len(pts) >= 3:
                slope = float(np.polyfit(pts[cx], pts[cy], 1)[0])
                intercept = float(np.polyfit(pts[cx], pts[cy], 1)[1])
                x_range = [float(pts[cx].min()), float(pts[cx].max())]
                reg_line = {"x": x_range, "y": [slope * x + intercept for x in x_range]}
            else:
                reg_line = None
            scatters.append({
                "x_col": cx, "y_col": cy,
                "r": round(r, 4),
                "r2": round(r ** 2, 4),
                "points": safe_json(pts.rename(columns={cx: "x", cy: "y"}).to_dict(orient="records")),
                "regression": safe_json(reg_line),
                "n": int(len(pts)),
            })
    out["scatters"] = scatters

    # ── Category bar charts ───────────────────────────
    cat_charts = []
    for col in cat_cols[:4]:
        vc = df[col].value_counts().head(12)
        cat_charts.append({
            "col": col,
            "labels": safe_json(vc.index.tolist()),
            "counts": safe_json(vc.values.tolist()),
            "unique": int(df[col].nunique()),
        })
        # If numeric companion exists, show grouped means
        if num_cols:
            nc = num_cols[0]
            grp = df.groupby(col)[nc].mean().reindex(vc.index)
            cat_charts[-1]["means_col"] = nc
            cat_charts[-1]["means"] = safe_json(grp.tolist())
    out["cat_charts"] = cat_charts

    # ── Box-plot data ─────────────────────────────────
    box_data = []
    for col in num_cols[:6]:
        clean = df[col].dropna().sort_values()
        n = len(clean)
        if n < 4:
            continue
        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1
        lo_fence = q1 - 1.5 * iqr
        hi_fence = q3 + 1.5 * iqr
        outliers = clean[(clean < lo_fence) | (clean > hi_fence)]
        box_data.append({
            "col": col,
            "min": safe_json(float(clean[clean >= lo_fence].min())),
            "q1": safe_json(q1),
            "median": safe_json(float(clean.median())),
            "q3": safe_json(q3),
            "max": safe_json(float(clean[clean <= hi_fence].max())),
            "mean": safe_json(float(clean.mean())),
            "outliers": safe_json(outliers.sample(min(len(outliers), 30), random_state=42).tolist()),
            "outlier_count": int(len(outliers)),
        })
    out["box_data"] = box_data

    # ── Time series (if date col present) ────────────
    if date_cols and num_cols:
        dc = date_cols[0]
        nc = num_cols[0]
        ts = df[[dc, nc]].dropna().sort_values(dc)
        ts[dc] = pd.to_datetime(ts[dc], errors="coerce")
        ts = ts.dropna(subset=[dc])
        # Aggregate to monthly if many points
        if len(ts) > 200:
            ts = ts.set_index(dc)[nc].resample("ME").mean().dropna().reset_index()
            ts.columns = [dc, nc]
        # Rolling 7-point average
        ts["rolling"] = ts[nc].rolling(min(7, len(ts)//4 or 1), center=True, min_periods=1).mean()
        out["time_series"] = {
            "date_col": dc, "val_col": nc,
            "dates": [str(d)[:10] for d in ts[dc]],
            "values": safe_json(ts[nc].tolist()),
            "rolling": safe_json(ts["rolling"].tolist()),
            "n": int(len(ts)),
        }

    # ── Correlation matrix (full) ─────────────────────
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        out["correlation_matrix"] = {
            "cols": num_cols,
            "matrix": safe_json(corr.where(pd.notnull(corr), None).values.tolist()),
        }

    return {"session_id": session_id, "eda": out, "num_cols": num_cols, "cat_cols": cat_cols, "date_cols": date_cols}


# ═══════════════════════════════════════════════════
# 10. /data — return rows as JSON for client-side use
# ═══════════════════════════════════════════════════
@app.get("/data")
async def get_data(session_id: str = "default", limit: int = 1000, offset: int = 0):
    df = get_df(session_id)
    slice_df = df.iloc[offset:offset + limit]
    return {
        "total_rows": len(df),
        "returned": len(slice_df),
        "offset": offset,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "rows": safe_json(slice_df.where(pd.notnull(slice_df), None).to_dict(orient="records")),
    }


# ═══════════════════════════════════════════════════
# 11. /compare — A/B segment comparison
# ═══════════════════════════════════════════════════
class CompareRequest(BaseModel):
    session_id: str = "default"
    group_col: str
    group_a: str
    group_b: str
    metric_cols: list[str] = []

@app.post("/compare")
async def compare_groups(req: CompareRequest):
    df = get_df(req.session_id)
    if req.group_col not in df.columns:
        raise HTTPException(400, f"Column '{req.group_col}' not found")
    a = df[df[req.group_col].astype(str) == req.group_a]
    b = df[df[req.group_col].astype(str) == req.group_b]
    if len(a) == 0 or len(b) == 0:
        raise HTTPException(400, "One or both groups have no rows")
    num_cols = req.metric_cols or df.select_dtypes(include=[np.number]).columns.tolist()[:6]
    results = []
    for col in num_cols:
        if col not in df.columns:
            continue
        va, vb = a[col].dropna(), b[col].dropna()
        if len(va) < 3 or len(vb) < 3:
            continue
        t_stat, p_val = stats.ttest_ind(va, vb, equal_var=False)
        effect = (va.mean() - vb.mean()) / np.sqrt((va.std()**2 + vb.std()**2) / 2)
        results.append({
            "col": col,
            "a_mean": safe_json(float(va.mean())),
            "b_mean": safe_json(float(vb.mean())),
            "a_std": safe_json(float(va.std())),
            "b_std": safe_json(float(vb.std())),
            "a_n": int(len(va)),
            "b_n": int(len(vb)),
            "t_stat": safe_json(float(t_stat)),
            "p_value": safe_json(float(p_val)),
            "cohens_d": safe_json(float(effect)),
            "significant": bool(p_val < 0.05),
            "pct_diff": safe_json(float((va.mean() - vb.mean()) / abs(vb.mean()) * 100) if vb.mean() != 0 else None),
        })
    return {
        "group_col": req.group_col,
        "group_a": req.group_a, "a_rows": int(len(a)),
        "group_b": req.group_b, "b_rows": int(len(b)),
        "comparisons": safe_json(results),
    }
