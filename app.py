"""
DataPilot v3 — Monetized Full-Stack App
========================================
Changes from v2:
  ✅ JWT auth on every endpoint
  ✅ Tier-gated features (Free / Pro / Team)
  ✅ Per-tier rate limiting
  ✅ Persistent sessions (PostgreSQL + parquet files)
  ✅ Stripe checkout + webhook
  ✅ Save/load trained models
  ✅ Auth routes: /auth/register, /auth/login, /auth/me
  ✅ Pricing route: /stripe/pricing

  python app.py
  → http://localhost:8765
"""

import os, sys, warnings, uuid
warnings.filterwarnings("ignore")

from pathlib import Path
from io import StringIO, BytesIO

import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ── ML ──────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingRegressor)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              accuracy_score, classification_report,
                              confusion_matrix, silhouette_score)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import scipy.stats as stats

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ── Auth & Payments ──────────────────────────────────────
from auth import (CurrentUser, get_current_user, require_tier,
                  check_row_limit, check_file_size, create_token, TIERS)
from payments import router as stripe_router
import db

# ── Lifespan (startup/shutdown) ──────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.get_pool()   # warm up DB connection
    yield

# ── App ──────────────────────────────────────────────────
app = FastAPI(title="DataPilot", version="3.0.0",
              docs_url="/api/docs", redoc_url=None, lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(stripe_router)

# ── In-memory session cache (loaded from DB on demand) ──
_session_cache: dict[str, pd.DataFrame] = {}


def safe_json(obj):
    if isinstance(obj, dict):   return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [safe_json(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj); return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray): return safe_json(obj.tolist())
    if isinstance(obj, pd.Series):  return safe_json(obj.tolist())
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


async def get_session_df(session_id: str, user: CurrentUser) -> pd.DataFrame:
    """Load DataFrame from cache or DB."""
    if session_id in _session_cache:
        return _session_cache[session_id]
    df = await db.load_dataset(session_id, user.user_id)
    if df is None:
        raise HTTPException(404, f"Dataset '{session_id}' not found. Upload a file first.")
    _session_cache[session_id] = df
    return df


# ════════════════════════════════════════════════════════
# FRONTEND
# ════════════════════════════════════════════════════════
FRONTEND_PATH  = Path(__file__).parent / "frontend" / "index.html"
LANDING_PATH   = Path(__file__).parent / "frontend" / "landing.html"
PRICING_PATH   = Path(__file__).parent / "frontend" / "pricing.html"

@app.get("/", response_class=HTMLResponse)
async def serve_landing():
    """Landing page — shown to everyone, no auth required."""
    if LANDING_PATH.exists():
        return HTMLResponse(LANDING_PATH.read_text(encoding="utf-8"))
    # Fallback: redirect to /app if landing page not deployed yet
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/app")

@app.get("/app", response_class=HTMLResponse)
async def serve_app():
    """Main app — authenticated users land here after login."""
    if not FRONTEND_PATH.exists():
        raise HTTPException(404, "frontend/index.html not found")
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))

@app.get("/pricing", response_class=HTMLResponse)
async def serve_pricing():
    if PRICING_PATH.exists():
        return HTMLResponse(PRICING_PATH.read_text(encoding="utf-8"))
    raise HTTPException(404, "pricing.html not found")


# ════════════════════════════════════════════════════════
# DEMO ENDPOINTS  — no auth required, hard row cap
# ════════════════════════════════════════════════════════
DEMO_ROW_LIMIT = 100
_demo_cache: dict[str, "pd.DataFrame"] = {}   # keyed by demo session id

@app.post("/demo/upload")
async def demo_upload(file: UploadFile = File(...)):
    """
    Unauthenticated upload for the try-demo flow.
    Accepts any CSV/Excel/JSON up to 2 MB and 100 rows.
    Returns basic profile + paywall hint when limits are exceeded.
    """
    content = await file.read()
    if len(content) > 2 * 1_000_000:
        return JSONResponse({
            "locked": True,
            "reason": "file_size",
            "message": f"Your file is {len(content)//1000:,} KB — the free demo supports up to 2 MB. "
                       "Upgrade to Pro for files up to 50 MB.",
            "upgrade_url": "/pricing"
        }, status_code=200)

    fname = file.filename.lower() if file.filename else ""
    try:
        if fname.endswith(".csv") or not any(fname.endswith(x) for x in (".xlsx",".xls",".json",".tsv")):
            for enc in ["utf-8","latin-1","cp1252"]:
                try: df = pd.read_csv(StringIO(content.decode(enc))); break
                except: continue
        elif fname.endswith((".xlsx",".xls")):
            df = pd.read_excel(BytesIO(content))
        elif fname.endswith(".json"):
            df = pd.read_json(StringIO(content.decode("utf-8")))
        else:
            df = pd.read_csv(StringIO(content.decode("utf-8",errors="replace")))
    except Exception as e:
        raise HTTPException(422, f"Could not parse file: {e}")

    total_rows = len(df)
    truncated  = total_rows > DEMO_ROW_LIMIT
    if truncated:
        df = df.head(DEMO_ROW_LIMIT)

    # Auto type detection
    for col in df.columns:
        if df[col].dtype == object:
            try: df[col] = pd.to_numeric(df[col].str.replace(",","").str.strip()); continue
            except: pass
        if df[col].dtype == object:
            try:
                converted = pd.to_datetime(df[col], infer_format=True, errors="coerce")
                if converted.notna().sum() > len(df) * 0.6: df[col] = converted
            except: pass

    demo_id = str(uuid.uuid4())
    _demo_cache[demo_id] = df

    response = {
        "demo_id":      demo_id,
        "filename":     file.filename,
        "rows_loaded":  len(df),
        "total_rows":   total_rows,
        "columns":      len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes":       {c: str(df[c].dtype) for c in df.columns},
        "preview":      safe_json(df.head(5).where(pd.notnull(df.head(5)), None).to_dict("records")),
        "message":      f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns",
    }
    if truncated:
        response["paywall"] = {
            "triggered":   True,
            "reason":      "row_limit",
            "rows_loaded": DEMO_ROW_LIMIT,
            "total_rows":  total_rows,
            "message":     f"Your file has {total_rows:,} rows — the free demo shows the first {DEMO_ROW_LIMIT}. "
                           f"Upgrade to Pro to analyse all {total_rows:,} rows.",
            "upgrade_url": "/pricing"
        }
    return response


@app.get("/demo/profile")
async def demo_profile(demo_id: str):
    """Profile endpoint for demo sessions — no auth required."""
    df = _demo_cache.get(demo_id)
    if df is None:
        raise HTTPException(404, "Demo session not found. Please upload a file first.")

    result = []
    for col in df.columns:
        series = df[col]
        null_count   = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        info = {"column": col, "dtype": str(series.dtype),
                "null_count": null_count,
                "null_pct": round(null_count / max(len(df),1) * 100, 2),
                "unique_count": unique_count}
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                info.update({"kind":"numeric","mean":safe_json(float(clean.mean())),
                    "median":safe_json(float(clean.median())),"std":safe_json(float(clean.std())),
                    "min":safe_json(float(clean.min())),"max":safe_json(float(clean.max()))})
        else:
            vc = series.dropna().value_counts()
            info.update({"kind":"categorical","top_values":safe_json(vc.head(5).to_dict())})
        result.append(info)

    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols  = df.select_dtypes(include=["object","category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    return {
        "demo_id":       demo_id,
        "columns":       safe_json(result),
        "total_rows":    len(df),
        "total_cols":    len(df.columns),
        "duplicate_rows":int(df.duplicated().sum()),
        "num_cols":      num_cols,
        "cat_cols":      cat_cols,
        "date_cols":     date_cols,
        "memory_mb":     round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "correlations":  {},   # not computed in demo
        # Remind the frontend what is locked
        "locked_features": ["ml_training","forecasting","ai_query","pdf_export","full_eda"],
        "upgrade_url": "/pricing"
    }


@app.get("/demo/eda")
async def demo_eda(demo_id: str):
    """EDA charts for demo sessions — no auth required."""
    df = _demo_cache.get(demo_id)
    if df is None:
        raise HTTPException(404, "Demo session not found. Please upload a file first.")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    out = {}

    # Histograms (up to 4 columns)
    histograms = {}
    for col in num_cols[:4]:
        clean = df[col].dropna()
        if len(clean) < 3: continue
        counts, edges = np.histogram(clean, bins=20)
        histograms[col] = {
            "counts": safe_json(counts.tolist()),
            "edges":  safe_json(edges.tolist()),
            "mean":   safe_json(float(clean.mean())),
            "median": safe_json(float(clean.median())),
        }
    out["histograms"] = histograms

    # Correlation matrix
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        out["correlation_matrix"] = {
            "cols":   num_cols,
            "matrix": safe_json(corr.where(pd.notnull(corr), None).values.tolist())
        }

    # Top category counts
    cat_charts = []
    for col in cat_cols[:2]:
        vc = df[col].value_counts().head(8)
        cat_charts.append({"col": col, "labels": safe_json(vc.index.tolist()),
                           "counts": safe_json(vc.values.tolist())})
    out["cat_charts"] = cat_charts

    return {
        "demo_id": demo_id, "eda": out,
        "num_cols": num_cols, "cat_cols": cat_cols,
        # ML, forecasting, AI locked
        "locked_features": {
            "ml_training":  {"locked": True, "message": "Train ML models with Pro — upgrade at /pricing"},
            "forecasting":  {"locked": True, "message": "Time series forecasting requires Pro"},
            "ai_query":     {"locked": True, "message": "AI natural language queries require Pro"},
            "pdf_export":   {"locked": True, "message": "PDF report export requires Pro"},
            "full_dataset": {"locked": True, "message": f"Full dataset analysis (500k rows) requires Pro"},
        },
        "upgrade_url": "/pricing"
    }


# ════════════════════════════════════════════════════════
# AUTH ROUTES
# ════════════════════════════════════════════════════════
class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/auth/register")
async def register(req: RegisterRequest):
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    try:
        user = await db.create_user(req.email, req.password)
    except (ValueError, Exception) as e:
        if "already" in str(e).lower() or "unique" in str(e).lower():
            raise HTTPException(409, "Email already registered")
        raise HTTPException(500, str(e))
    token = create_token(user["id"], user["email"], user.get("tier", "free"))
    return {"token": token, "user": {"id": user["id"], "email": user["email"],
                                      "tier": user.get("tier", "free")}}

@app.post("/auth/login")
async def login(req: LoginRequest):
    user = await db.verify_password(req.email, req.password)
    if not user:
        raise HTTPException(401, "Invalid email or password")
    token = create_token(user["id"], user["email"], user.get("tier", "free"))
    return {"token": token, "user": {"id": user["id"], "email": user["email"],
                                      "tier": user.get("tier", "free")}}

@app.get("/auth/me")
async def me(user: CurrentUser = Depends(get_current_user)):
    return {
        "user_id": user.user_id,
        "email":   user.email,
        "tier":    user.tier,
        "limits":  user.limits,
        "datasets": await db.list_datasets(user.user_id),
        "models":   await db.list_models(user.user_id),
    }


# ════════════════════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0.0",
            "statsmodels": HAS_STATSMODELS, "sklearn": True}


# ════════════════════════════════════════════════════════
# 1. UPLOAD  — tier-gated by row/file-size limits
# ════════════════════════════════════════════════════════
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    session_id: str = None,
    user: CurrentUser = Depends(get_current_user),
):
    content = await file.read()
    check_file_size(user, len(content))

    fname = file.filename.lower()
    try:
        if fname.endswith(".csv"):
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try: df = pd.read_csv(StringIO(content.decode(enc))); break
                except: continue
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(content))
        elif fname.endswith(".json"):
            df = pd.read_json(StringIO(content.decode("utf-8")))
        elif fname.endswith(".tsv"):
            df = pd.read_csv(StringIO(content.decode("utf-8")), sep="\t")
        else:
            raise HTTPException(400, "Unsupported type. Use CSV, Excel, JSON, or TSV.")
    except HTTPException: raise
    except Exception as e: raise HTTPException(422, f"Parse error: {e}")

    check_row_limit(user, len(df))

    # Free tier column limit
    if not user.limits["ml_allowed"] and len(df.columns) > user.limits["max_cols"]:
        df = df.iloc[:, :user.limits["max_cols"]]

    # Auto type detection
    for col in df.columns:
        if df[col].dtype == object:
            try: df[col] = pd.to_numeric(df[col].str.replace(",","").str.strip()); continue
            except: pass
        if df[col].dtype == object:
            try:
                converted = pd.to_datetime(df[col], infer_format=True, errors="coerce")
                if converted.notna().sum() > len(df) * 0.6: df[col] = converted
            except: pass

    ds_id = session_id or str(uuid.uuid4())
    _session_cache[ds_id] = df
    saved_id = await db.save_dataset(user.user_id, file.filename, file.filename, df)

    return {
        "session_id":   saved_id,
        "filename":     file.filename,
        "rows":         len(df),
        "columns":      len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes":       {c: str(df[c].dtype) for c in df.columns},
        "preview":      safe_json(df.head(5).where(pd.notnull(df.head(5)), None).to_dict("records")),
        "tier_limits":  {"max_rows": user.limits["max_rows"],
                         "ml_allowed": user.limits["ml_allowed"]},
        "message":      f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns",
    }


# ════════════════════════════════════════════════════════
# 2. PROFILE
# ════════════════════════════════════════════════════════
@app.get("/profile")
async def profile(session_id: str = "default",
                  user: CurrentUser = Depends(get_current_user)):
    df = await get_session_df(session_id, user)
    result = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        info = {"column": col, "dtype": str(series.dtype),
                "null_count": null_count,
                "null_pct": round(null_count / len(df) * 100, 2),
                "unique_count": unique_count,
                "unique_pct": round(unique_count / len(df) * 100, 2)}
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                out_count = int(((clean < lo) | (clean > hi)).sum())
                _, norm_p = stats.shapiro(clean.sample(min(len(clean), 500), random_state=42))
                info.update({"kind": "numeric", "mean": safe_json(float(clean.mean())),
                    "median": safe_json(float(clean.median())), "std": safe_json(float(clean.std())),
                    "min": safe_json(float(clean.min())), "max": safe_json(float(clean.max())),
                    "q1": safe_json(float(q1)), "q3": safe_json(float(q3)),
                    "outlier_count": out_count, "outlier_pct": round(out_count/len(clean)*100,2),
                    "skewness": safe_json(float(clean.skew())),
                    "kurtosis": safe_json(float(clean.kurtosis())),
                    "is_normal": bool(norm_p > 0.05), "normality_p": safe_json(float(norm_p)),
                    "histogram": safe_json(np.histogram(clean, bins=20)[0].tolist()),
                    "histogram_edges": safe_json(np.histogram(clean, bins=20)[1].tolist())})
        elif pd.api.types.is_datetime64_any_dtype(series):
            clean = series.dropna()
            info.update({"kind": "datetime",
                "min": str(clean.min()) if len(clean) else None,
                "max": str(clean.max()) if len(clean) else None,
                "range_days": int((clean.max()-clean.min()).days) if len(clean)>1 else 0})
        else:
            vc = series.dropna().value_counts()
            info.update({"kind": "categorical",
                "top_values": safe_json(vc.head(10).to_dict()),
                "top_value": str(vc.index[0]) if len(vc) else None,
                "top_count": int(vc.iloc[0]) if len(vc) else 0,
                "entropy": safe_json(float(stats.entropy(vc.values)))})
        result.append(info)

    num_df = df.select_dtypes(include=[np.number])
    corr_matrix = {}
    if len(num_df.columns) >= 2:
        corr = num_df.corr()
        corr_matrix = safe_json(corr.where(pd.notnull(corr), None).to_dict())

    return {"columns": safe_json(result), "total_rows": len(df),
            "total_cols": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "correlations": corr_matrix,
            "memory_mb": round(df.memory_usage(deep=True).sum()/1e6,3)}


# ════════════════════════════════════════════════════════
# 3. CLEAN
# ════════════════════════════════════════════════════════
class CleanRequest(BaseModel):
    session_id: str = "default"
    operations: list[dict]

@app.post("/clean")
async def clean(req: CleanRequest, user: CurrentUser = Depends(get_current_user)):
    df = (await get_session_df(req.session_id, user)).copy()
    log = []
    for op in req.operations:
        t = op.get("op"); col = op.get("col")
        if t == "drop_duplicates":
            b = len(df); df = df.drop_duplicates()
            log.append(f"✅ Removed {b-len(df):,} duplicate rows")
        elif t == "drop_nulls":
            b = len(df)
            if col: df = df.dropna(subset=[col])
            else: df = df.dropna(thresh=int(len(df.columns)*op.get("thresh",0.5)))
            log.append(f"✅ Dropped {b-len(df):,} rows")
        elif t == "impute":
            method = op.get("method","median")
            if col and col in df.columns:
                n = int(df[col].isna().sum())
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill = {"mean":df[col].mean(),"median":df[col].median(),"zero":0}.get(method,df[col].median())
                    df[col] = df[col].fillna(fill) if method != "forward" else df[col].ffill()
                else:
                    mode = df[col].mode()
                    df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
                log.append(f"✅ Imputed {n:,} nulls in '{col}'")
        elif t == "normalize":
            method = op.get("method","minmax")
            if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if method == "minmax":
                    mn,mx = df[col].min(),df[col].max()
                    df[col] = (df[col]-mn)/(mx-mn) if mx!=mn else 0
                elif method == "zscore":
                    df[col] = (df[col]-df[col].mean())/df[col].std()
                elif method == "log":
                    df[col] = np.log1p(df[col].clip(lower=0))
                log.append(f"✅ Normalized '{col}' → {method}")
        elif t == "drop_column":
            if col and col in df.columns:
                df = df.drop(columns=[col]); log.append(f"✅ Dropped '{col}'")
        elif t == "create_column":
            name = op.get("name") or col; expr = op.get("expr","")
            if name and expr:
                try: df[name] = df.eval(expr); log.append(f"✅ Created '{name}'={expr}")
                except Exception as e: log.append(f"❌ {e}")

    _session_cache[req.session_id] = df
    await db.save_dataset(user.user_id, req.session_id, req.session_id, df)
    return {"session_id": req.session_id, "rows": len(df),
            "cols": len(df.columns), "log": log}


# ════════════════════════════════════════════════════════
# 4. ANALYZE
# ════════════════════════════════════════════════════════
@app.get("/analyze")
async def analyze(session_id: str = "default",
                  user: CurrentUser = Depends(get_current_user)):
    df = await get_session_df(session_id, user)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    out = {}

    # Pearson correlations
    if len(num_cols) >= 2:
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                c1,c2 = num_cols[i], num_cols[j]
                r,p = stats.pearsonr(df[c1].dropna(), df[c2].dropna())
                pairs.append({"col1":c1,"col2":c2,"r":safe_json(round(float(r),4)),
                               "p":safe_json(float(p)),"significant":bool(p<0.05)})
        out["correlations"] = sorted(pairs, key=lambda x: abs(x["r"]), reverse=True)[:20]

    # ANOVA
    if cat_cols and num_cols:
        cat,num = cat_cols[0], num_cols[0]
        groups = [df[df[cat]==v][num].dropna().values for v in df[cat].unique() if len(df[df[cat]==v]) >= 3]
        if len(groups) >= 2:
            f,p = stats.f_oneway(*groups)
            out["anova"] = {"cat_col":cat,"num_col":num,"f_stat":safe_json(float(f)),"p":safe_json(float(p))}

    # Normality
    norm_results = []
    for col in num_cols[:6]:
        clean = df[col].dropna()
        if len(clean) >= 8:
            _,p = stats.shapiro(clean.sample(min(len(clean),500),random_state=42))
            norm_results.append({"col":col,"p":safe_json(float(p)),"is_normal":bool(p>0.05)})
    out["normality"] = norm_results

    return {"session_id": session_id, "analysis": out}


# ════════════════════════════════════════════════════════
# 5. TRAIN MODEL — Pro/Team only
# ════════════════════════════════════════════════════════
class TrainRequest(BaseModel):
    session_id: str = "default"
    model_type: str = "linear_regression"
    target_col: str = ""
    feature_cols: list[str] = []
    test_size: float = 0.2
    n_clusters: int = 3
    save_model: bool = False
    model_name: str = ""

@app.post("/train-model")
async def train_model(
    req: TrainRequest,
    user: CurrentUser = require_tier("pro", "team"),
):
    df = await get_session_df(req.session_id, user)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ── K-Means clustering ───────────────────────────────
    if req.model_type == "kmeans":
        X = df[num_cols].dropna()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        km = KMeans(n_clusters=req.n_clusters, random_state=42, n_init=10)
        km.fit(Xs)
        labels = km.labels_.tolist()
        sil = safe_json(float(silhouette_score(Xs, labels)))
        pca2 = PCA(n_components=2).fit_transform(Xs)
        inertias = []
        for k in range(2, min(11, len(X))):
            inertias.append(float(KMeans(n_clusters=k,random_state=42,n_init=10).fit(Xs).inertia_))

        if req.save_model and req.model_name:
            await db.save_model(user.user_id, req.session_id, req.model_name,
                                "kmeans", "", km, {"silhouette": sil, "clusters": req.n_clusters})

        return {"model_type":"kmeans","n_clusters":req.n_clusters,"silhouette":sil,
                "labels":labels[:500],
                "pca_points":safe_json([{"x":float(pca2[i,0]),"y":float(pca2[i,1]),"cluster":int(labels[i])} for i in range(min(500,len(pca2)))]),
                "elbow":safe_json(inertias)}

    # ── Supervised ──────────────────────────────────────
    if not req.target_col or req.target_col not in df.columns:
        raise HTTPException(400, "target_col required for supervised models")

    feats = req.feature_cols or [c for c in num_cols if c != req.target_col]
    sub = df[feats + [req.target_col]].dropna()
    if len(sub) < 10:
        raise HTTPException(400, "Need at least 10 complete rows")

    # Encode categoricals
    X = sub[feats].copy()
    for c in X.select_dtypes(include=["object","category"]).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    y = sub[req.target_col]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=req.test_size, random_state=42)

    MODELS = {
        "linear_regression":    LinearRegression(),
        "ridge_regression":     Ridge(alpha=1.0),
        "random_forest_reg":    RandomForestRegressor(n_estimators=100,random_state=42),
        "gradient_boosting":    GradientBoostingRegressor(n_estimators=100,random_state=42),
        "logistic_regression":  LogisticRegression(max_iter=1000,random_state=42),
        "random_forest_clf":    RandomForestClassifier(n_estimators=100,random_state=42),
    }

    if req.model_type not in MODELS:
        raise HTTPException(400, f"Unknown model_type: {req.model_type}")

    model = MODELS[req.model_type]
    is_clf = req.model_type in ("logistic_regression","random_forest_clf")

    if is_clf:
        le = LabelEncoder(); y_tr_enc = le.fit_transform(y_tr); y_te_enc = le.transform(y_te)
        model.fit(X_tr, y_tr_enc)
        y_pred = model.predict(X_te)
        acc = float(accuracy_score(y_te_enc, y_pred))
        cm  = confusion_matrix(y_te_enc, y_pred).tolist()
        metrics = {"accuracy": round(acc,4), "confusion_matrix": cm}
        result = {"model_type":req.model_type,"target":req.target_col,
                  "features":feats,"train_rows":len(X_tr),"test_rows":len(X_te),
                  "metrics":metrics,"task":"classification"}
    else:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mse = float(mean_squared_error(y_te, y_pred))
        metrics = {"r2":round(float(r2_score(y_te,y_pred)),4),
                   "rmse":round(float(np.sqrt(mse)),4),
                   "mae":round(float(mean_absolute_error(y_te,y_pred)),4)}
        fi = []
        if hasattr(model,"feature_importances_"):
            fi = [{"feature":f,"importance":round(float(i),4)}
                  for f,i in sorted(zip(feats,model.feature_importances_),key=lambda x:-x[1])]
        result = {"model_type":req.model_type,"target":req.target_col,
                  "features":feats,"train_rows":len(X_tr),"test_rows":len(X_te),
                  "metrics":metrics,"feature_importances":fi,
                  "actual_vs_predicted":safe_json([{"actual":float(a),"predicted":float(p)}
                      for a,p in zip(y_te.values[:200],y_pred[:200])]),
                  "task":"regression"}

    if req.save_model and req.model_name:
        mid = await db.save_model(user.user_id, req.session_id, req.model_name,
                                   req.model_type, req.target_col, model, metrics)
        result["saved_model_id"] = mid

    return result


# ════════════════════════════════════════════════════════
# 6. FORECAST — Pro/Team only
# ════════════════════════════════════════════════════════
class ForecastRequest(BaseModel):
    session_id: str = "default"
    target_col: str
    date_col: str = None
    periods: int = 8
    freq: str = "ME"

@app.post("/forecast")
async def forecast(
    req: ForecastRequest,
    user: CurrentUser = require_tier("pro", "team"),
):
    df = (await get_session_df(req.session_id, user)).copy()
    if req.target_col not in df.columns:
        raise HTTPException(400, f"'{req.target_col}' not found")

    if req.date_col and req.date_col in df.columns:
        ts = df[[req.date_col, req.target_col]].dropna()
        ts[req.date_col] = pd.to_datetime(ts[req.date_col], errors="coerce")
        ts = ts.dropna(subset=[req.date_col]).sort_values(req.date_col)
        ts = ts.set_index(req.date_col)[req.target_col].resample(req.freq).mean().dropna()
    else:
        ts = df[req.target_col].dropna().reset_index(drop=True)
        ts.index = pd.date_range("2020-01", periods=len(ts), freq=req.freq)

    if len(ts) < 6:
        raise HTTPException(400, "Need at least 6 data points")

    if not HAS_STATSMODELS:
        # Simple fallback
        sm = [float(ts.iloc[0])]
        for v in ts.iloc[1:]: sm.append(0.3*float(v)+0.7*sm[-1])
        slope = (sm[-1]-sm[0]) / max(len(sm)-1,1)
        fv = [sm[-1]+slope*(i+1) for i in range(req.periods)]
        se = float(np.std(ts.values-sm[:len(ts)])*1.96)
        fdates = pd.date_range(ts.index[-1],periods=req.periods+1,freq=req.freq)[1:]
        return {"method":"ses","target_col":req.target_col,"periods":req.periods,
                "historical":[{"date":str(d),"value":safe_json(float(v))} for d,v in ts.items()],
                "forecast":[{"date":str(d),"value":safe_json(v),"lo":safe_json(v-se),"hi":safe_json(v+se)} for d,v in zip(fdates,fv)]}

    results = {}
    sp = min(12, len(ts)//2)
    fdates = pd.date_range(ts.index[-1], periods=req.periods+1, freq=req.freq)[1:]

    try:
        ets = ExponentialSmoothing(ts, trend="add",
            seasonal="add" if len(ts)>=2*sp else None,
            seasonal_periods=sp if len(ts)>=2*sp else None,
            initialization_method="estimated").fit(optimized=True)
        fc = ets.forecast(req.periods)
        se = float(np.std(ts.values-ets.fittedvalues.values))
        results["ets"] = {"aic":safe_json(float(ets.aic)),
            "forecast":[{"date":str(d),"value":safe_json(float(v)),
                         "lo":safe_json(float(v)-1.96*se*np.sqrt(i+1)),
                         "hi":safe_json(float(v)+1.96*se*np.sqrt(i+1))}
                        for i,(d,v) in enumerate(zip(fdates,fc))],
            "smoothed":safe_json(ets.fittedvalues.tolist())}
    except Exception as e: results["ets_error"] = str(e)

    try:
        arima = SARIMAX(ts, order=(1,1,1), trend="c").fit(disp=False)
        fc_a  = arima.get_forecast(steps=req.periods)
        ci    = fc_a.conf_int()
        results["arima"] = {"aic":safe_json(float(arima.aic)),
            "forecast":[{"date":str(d),"value":safe_json(float(v)),
                         "lo":safe_json(float(ci.iloc[i,0])),"hi":safe_json(float(ci.iloc[i,1]))}
                        for i,(d,v) in enumerate(zip(fdates,fc_a.predicted_mean))]}
    except Exception as e: results["arima_error"] = str(e)

    best = "ets" if ("ets" in results and ("arima" not in results or results["ets"]["aic"]<=results["arima"]["aic"])) else "arima"
    return {"method":best,"target_col":req.target_col,"periods":req.periods,
            "historical":[{"date":str(d),"value":safe_json(float(v))} for d,v in ts.items()],
            "forecast":results.get(best,{}).get("forecast",[]),
            "model_comparison":{k:{"aic":results[k].get("aic")} for k in results if "error" not in k},
            "best_model":best}


# ════════════════════════════════════════════════════════
# 7. AI QUERY — Pro/Team only (uses Claude API)
# ════════════════════════════════════════════════════════
class QueryRequest(BaseModel):
    session_id: str = "default"
    query: str

@app.post("/query")
async def nl_query(
    req: QueryRequest,
    user: CurrentUser = require_tier("pro", "team"),
):
    import re, httpx
    df = await get_session_df(req.session_id, user)
    q  = req.query.strip().lower()
    cols     = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

    # ── Rule engine (fast, no LLM cost) ─────────────────
    result_df = None; explanation = ""; operation = "unknown"

    m = re.search(r"corr(?:elation)? (?:between |of )?(.+?) and (.+?)$", q)
    if m:
        c1 = next((c for c in cols if c.lower()==m.group(1).strip()), None)
        c2 = next((c for c in cols if c.lower()==m.group(2).strip()), None)
        if c1 and c2:
            r = float(df[c1].corr(df[c2]))
            return {"operation":f"correlation({c1},{c2})",
                    "result":{"r":round(r,4)},
                    "explanation":f"Pearson r={r:.4f}. {'Strong' if abs(r)>.7 else 'Moderate' if abs(r)>.4 else 'Weak'} {'positive' if r>0 else 'negative'} relationship."}

    m = re.search(r"top (\d+)(?: by| sorted by)? (.+?)$", q)
    if m and result_df is None:
        n,col_raw = int(m.group(1)), m.group(2).strip()
        col = next((c for c in num_cols if c.lower()==col_raw), num_cols[0] if num_cols else None)
        if col:
            result_df = df.nlargest(n, col)
            operation = f"top_{n}({col})"; explanation = f"Top {n} rows by {col}."

    m = re.search(r"(?:group|sum|average|mean|count) (.+?) by (.+?)$", q)
    if m and result_df is None:
        vc = next((c for c in num_cols if c.lower()==m.group(1).strip()), num_cols[0] if num_cols else None)
        gc = next((c for c in cols if c.lower()==m.group(2).strip()), cat_cols[0] if cat_cols else None)
        if vc and gc:
            agg = "sum" if "sum" in q else "mean" if any(w in q for w in ("average","mean")) else "count"
            result_df = df.groupby(gc)[vc].agg(agg).reset_index().sort_values(vc, ascending=False)
            operation = f"groupby({gc}).{agg}({vc})"; explanation = f"{agg.title()} of {vc} by {gc}."

    if result_df is not None:
        return {"operation":operation,"explanation":explanation,
                "result":{"rows":len(result_df),"columns":result_df.columns.tolist(),
                           "data":safe_json(result_df.head(50).where(pd.notnull(result_df.head(50)),None).to_dict("records"))}}

    # ── Claude API fallback ──────────────────────────────
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY","")
    if not anthropic_key:
        return {"operation":"unknown","result":None,
                "explanation":"Couldn't parse query. Set ANTHROPIC_API_KEY for AI-powered queries."}

    schema_summary = f"Columns: {', '.join(cols[:20])}. Rows: {len(df):,}. Sample: {df.head(3).to_dict('records')}"
    prompt = (f"You are a data analyst. Given this dataset:\n{schema_summary}\n\n"
              f"Answer this question concisely: {req.query}\n\n"
              f"If you can suggest a pandas operation, include it. Be brief and direct.")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": anthropic_key,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model":"claude-sonnet-4-20250514","max_tokens":500,
                      "messages":[{"role":"user","content":prompt}]}
            )
            answer = resp.json()["content"][0]["text"]
    except Exception as e:
        answer = f"AI query failed: {e}"

    return {"operation":"ai_query","result":None,"explanation":answer}


# ════════════════════════════════════════════════════════
# 8. EDA
# ════════════════════════════════════════════════════════
@app.get("/eda")
async def eda(session_id: str = "default", max_scatter: int = 600,
              user: CurrentUser = Depends(get_current_user)):
    df = await get_session_df(session_id, user)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    out = {}; sample = df.sample(min(max_scatter, len(df)), random_state=42)

    histograms = {}
    for col in num_cols[:8]:
        clean = df[col].dropna()
        if len(clean)<3: continue
        counts, edges = np.histogram(clean, bins=25)
        mean_v, std_v = float(clean.mean()), float(clean.std()) or 1
        centers = [(float(edges[i])+float(edges[i+1]))/2 for i in range(len(edges)-1)]
        kde = [float(len(clean))*float(edges[1]-edges[0])*np.exp(-0.5*((x-mean_v)/std_v)**2)/(std_v*np.sqrt(2*np.pi)) for x in centers]
        histograms[col] = {"counts":safe_json(counts.tolist()),"edges":safe_json(edges.tolist()),
            "centers":safe_json(centers),"kde":safe_json(kde),"mean":safe_json(mean_v),
            "median":safe_json(float(clean.median())),"std":safe_json(std_v),"n":int(len(clean))}
    out["histograms"] = histograms

    scatters = []
    for i in range(min(len(num_cols),4)):
        for j in range(i+1, min(len(num_cols),4)):
            cx,cy = num_cols[i],num_cols[j]
            pts = sample[[cx,cy]].dropna()
            if len(pts)<3: continue
            r = float(pts[cx].corr(pts[cy]))
            slope,intercept = np.polyfit(pts[cx],pts[cy],1)
            xr = [float(pts[cx].min()),float(pts[cx].max())]
            scatters.append({"x_col":cx,"y_col":cy,"r":round(r,4),
                "points":safe_json(pts.rename(columns={cx:"x",cy:"y"}).to_dict("records")),
                "regression":safe_json({"x":xr,"y":[float(slope)*x+float(intercept) for x in xr]})})
    out["scatters"] = scatters

    cat_charts = []
    for col in cat_cols[:4]:
        vc = df[col].value_counts().head(12)
        entry = {"col":col,"labels":safe_json(vc.index.tolist()),"counts":safe_json(vc.values.tolist())}
        if num_cols:
            nc = num_cols[0]; grp = df.groupby(col)[nc].mean().reindex(vc.index)
            entry["means_col"] = nc; entry["means"] = safe_json(grp.tolist())
        cat_charts.append(entry)
    out["cat_charts"] = cat_charts

    if date_cols and num_cols:
        dc,nc = date_cols[0],num_cols[0]
        ts = df[[dc,nc]].dropna().sort_values(dc)
        ts[dc] = pd.to_datetime(ts[dc],errors="coerce"); ts = ts.dropna(subset=[dc])
        if len(ts)>200:
            ts = ts.set_index(dc)[nc].resample("ME").mean().dropna().reset_index(); ts.columns=[dc,nc]
        ts["rolling"] = ts[nc].rolling(min(7,len(ts)//4 or 1),center=True,min_periods=1).mean()
        out["time_series"] = {"date_col":dc,"val_col":nc,
            "dates":[str(d)[:10] for d in ts[dc]],
            "values":safe_json(ts[nc].tolist()),"rolling":safe_json(ts["rolling"].tolist())}

    if len(num_cols)>=2:
        corr = df[num_cols].corr()
        out["correlation_matrix"] = {"cols":num_cols,"matrix":safe_json(corr.where(pd.notnull(corr),None).values.tolist())}

    return {"session_id":session_id,"eda":out,"num_cols":num_cols,"cat_cols":cat_cols,"date_cols":date_cols}


# ════════════════════════════════════════════════════════
# 9. DATA (paginated)
# ════════════════════════════════════════════════════════
@app.get("/data")
async def get_data(session_id: str = "default", limit: int = 1000, offset: int = 0,
                   user: CurrentUser = Depends(get_current_user)):
    df = await get_session_df(session_id, user)
    sl = df.iloc[offset:offset+limit]
    return {"total_rows":len(df),"returned":len(sl),"offset":offset,
            "columns":df.columns.tolist(),
            "rows":safe_json(sl.where(pd.notnull(sl),None).to_dict("records"))}


# ════════════════════════════════════════════════════════
# 10. COMPARE (A/B)
# ════════════════════════════════════════════════════════
class CompareRequest(BaseModel):
    session_id: str = "default"
    group_col: str; group_a: str; group_b: str; metric_cols: list[str] = []

@app.post("/compare")
async def compare_groups(req: CompareRequest, user: CurrentUser = Depends(get_current_user)):
    df = await get_session_df(req.session_id, user)
    a = df[df[req.group_col].astype(str)==req.group_a]
    b = df[df[req.group_col].astype(str)==req.group_b]
    if len(a)==0 or len(b)==0: raise HTTPException(400,"One or both groups have no rows")
    num_cols = req.metric_cols or df.select_dtypes(include=[np.number]).columns.tolist()[:6]
    results = []
    for col in num_cols:
        if col not in df.columns: continue
        va,vb = a[col].dropna(), b[col].dropna()
        if len(va)<3 or len(vb)<3: continue
        t,p = stats.ttest_ind(va,vb,equal_var=False)
        d = (va.mean()-vb.mean())/np.sqrt((va.std()**2+vb.std()**2)/2)
        results.append({"col":col,"a_mean":safe_json(float(va.mean())),"b_mean":safe_json(float(vb.mean())),
                        "t_stat":safe_json(float(t)),"p_value":safe_json(float(p)),
                        "cohens_d":safe_json(float(d)),"significant":bool(p<0.05)})
    return {"group_a":req.group_a,"group_b":req.group_b,"comparisons":safe_json(results)}


# ════════════════════════════════════════════════════════
# 11. EXPORT
# ════════════════════════════════════════════════════════
@app.get("/export")
async def export_data(session_id: str = "default", format: str = "csv",
                      user: CurrentUser = Depends(get_current_user)):
    df = await get_session_df(session_id, user)
    if format == "csv":
        out = StringIO(); df.to_csv(out, index=False); out.seek(0)
        content = out.getvalue()
        if user.limits["export_watermark"]:
            content = "# Generated by DataPilot Free — upgrade at datapilot.app/pricing\n" + content
        return StreamingResponse(iter([content]), media_type="text/csv",
            headers={"Content-Disposition":"attachment; filename=datapilot_export.csv"})
    if format == "json":
        data = safe_json(df.to_dict("records"))
        return JSONResponse(data)
    raise HTTPException(400, "Use format=csv or format=json")


# ════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8765))
    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║      DataPilot v3  —  Monetized 🚀       ║")
    print(f"  ╠══════════════════════════════════════════╣")
    print(f"  ║  App     →  http://localhost:{port}      ║")
    print(f"  ║  Pricing →  http://localhost:{port}/pricing ║")
    print(f"  ║  Docs    →  http://localhost:{port}/api/docs ║")
    print(f"  ╚══════════════════════════════════════════╝\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
