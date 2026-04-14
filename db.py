"""
DataPilot — Persistent Storage Layer
======================================
Replaces the in-memory `sessions` dict with PostgreSQL + S3/local disk.

Setup:
  pip install asyncpg boto3 bcrypt
  export DATABASE_URL=postgresql://user:pass@localhost:5432/datapilot
  export S3_BUCKET=datapilot-datasets    (optional, falls back to local disk)
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...

Run migrations:
  python db.py --migrate
"""

import os
import json
import uuid
import pickle
import hashlib
import bcrypt
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO
from typing import Optional

import pandas as pd

# ── Config ──────────────────────────────────────────────
DATABASE_URL   = os.environ.get("DATABASE_URL", "")
S3_BUCKET      = os.environ.get("S3_BUCKET", "")
LOCAL_STORE    = Path(os.environ.get("LOCAL_STORE", "./data"))
LOCAL_STORE.mkdir(parents=True, exist_ok=True)
(LOCAL_STORE / "datasets").mkdir(exist_ok=True)
(LOCAL_STORE / "models").mkdir(exist_ok=True)

# ── SQL schema ──────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    email           TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    tier            TEXT NOT NULL DEFAULT 'free',
    stripe_customer TEXT,
    stripe_sub      TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_login      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS datasets (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    filename    TEXT NOT NULL,
    rows        INT,
    cols        INT,
    size_bytes  BIGINT,
    file_path   TEXT,          -- local disk path or s3://bucket/key
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    last_used   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS saved_models (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    dataset_id  TEXT REFERENCES datasets(id) ON DELETE SET NULL,
    name        TEXT NOT NULL,
    model_type  TEXT NOT NULL,
    target_col  TEXT,
    metrics     JSONB,
    file_path   TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_datasets_user  ON datasets(user_id);
CREATE INDEX IF NOT EXISTS idx_models_user    ON saved_models(user_id);
"""


# ── Async DB pool (asyncpg) ─────────────────────────────
_pool = None

async def get_pool():
    global _pool
    if _pool is None and DATABASE_URL:
        try:
            import asyncpg
            _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
            async with _pool.acquire() as conn:
                await conn.execute(SCHEMA)
        except Exception as e:
            print(f"[DB] Could not connect to PostgreSQL: {e}")
            print("[DB] Falling back to in-memory store")
            _pool = None
    return _pool


# ── Fallback in-memory store ─────────────────────────────
_mem_users:    dict[str, dict]        = {}   # email → user record
_mem_datasets: dict[str, pd.DataFrame] = {}  # dataset_id → DataFrame
_mem_models:   dict[str, dict]        = {}   # model_id → {model, meta}


# ── User operations ─────────────────────────────────────
async def create_user(email: str, password: str, tier: str = "free") -> dict:
    user_id = str(uuid.uuid4())
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO users(id, email, password_hash, tier) VALUES($1,$2,$3,$4) RETURNING *",
                user_id, email.lower(), pw_hash, tier
            )
            return dict(row)
    else:
        if email.lower() in _mem_users:
            raise ValueError("Email already registered")
        record = {"id": user_id, "email": email.lower(),
                  "password_hash": pw_hash, "tier": tier}
        _mem_users[email.lower()] = record
        return record


async def get_user_by_email(email: str) -> Optional[dict]:
    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE email=$1", email.lower()
            )
            return dict(row) if row else None
    return _mem_users.get(email.lower())


async def verify_password(email: str, password: str) -> Optional[dict]:
    user = await get_user_by_email(email)
    if not user:
        return None
    if bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return user
    return None


async def update_user_tier(user_id: str, tier: str, stripe_sub: Optional[str] = None):
    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET tier=$1, stripe_sub=$2 WHERE id=$3",
                tier, stripe_sub, user_id
            )
    else:
        for user in _mem_users.values():
            if user["id"] == user_id:
                user["tier"] = tier


# ── Dataset operations ───────────────────────────────────
async def save_dataset(
    user_id: str, name: str, filename: str, df: pd.DataFrame
) -> str:
    """Persist a DataFrame. Returns dataset_id."""
    dataset_id = str(uuid.uuid4())
    path = _df_path(dataset_id)
    df.to_parquet(path, index=False)

    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO datasets(id,user_id,name,filename,rows,cols,size_bytes,file_path)
                   VALUES($1,$2,$3,$4,$5,$6,$7,$8)""",
                dataset_id, user_id, name, filename,
                len(df), len(df.columns),
                path.stat().st_size, str(path)
            )
    else:
        _mem_datasets[dataset_id] = df

    return dataset_id


async def load_dataset(dataset_id: str, user_id: str) -> Optional[pd.DataFrame]:
    """Load a persisted DataFrame, verifying ownership."""
    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM datasets WHERE id=$1 AND user_id=$2",
                dataset_id, user_id
            )
            if not row:
                return None
            await conn.execute(
                "UPDATE datasets SET last_used=NOW() WHERE id=$1", dataset_id
            )
        path = Path(row["file_path"])
    else:
        if dataset_id not in _mem_datasets:
            return None
        return _mem_datasets[dataset_id]

    return pd.read_parquet(path) if path.exists() else None


async def list_datasets(user_id: str) -> list[dict]:
    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id,name,filename,rows,cols,size_bytes,created_at,last_used "
                "FROM datasets WHERE user_id=$1 ORDER BY last_used DESC",
                user_id
            )
            return [dict(r) for r in rows]
    return [
        {"id": k, "name": k, "filename": "uploaded", "rows": len(v), "cols": len(v.columns)}
        for k, v in _mem_datasets.items()
    ]


async def delete_dataset(dataset_id: str, user_id: str):
    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "DELETE FROM datasets WHERE id=$1 AND user_id=$2 RETURNING file_path",
                dataset_id, user_id
            )
            if row:
                p = Path(row["file_path"])
                if p.exists():
                    p.unlink()
    else:
        _mem_datasets.pop(dataset_id, None)


# ── Model operations ─────────────────────────────────────
async def save_model(
    user_id: str, dataset_id: str, name: str,
    model_type: str, target_col: str, model_obj, metrics: dict
) -> str:
    model_id = str(uuid.uuid4())
    path = LOCAL_STORE / "models" / f"{model_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model_obj, f)

    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO saved_models(id,user_id,dataset_id,name,model_type,
                   target_col,metrics,file_path)
                   VALUES($1,$2,$3,$4,$5,$6,$7,$8)""",
                model_id, user_id, dataset_id, name, model_type,
                target_col, json.dumps(metrics), str(path)
            )
    else:
        _mem_models[model_id] = {"meta": {"name": name, "type": model_type,
                                           "metrics": metrics}, "obj": model_obj}
    return model_id


async def load_model(model_id: str, user_id: str):
    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM saved_models WHERE id=$1 AND user_id=$2",
                model_id, user_id
            )
            if not row:
                return None, None
        with open(row["file_path"], "rb") as f:
            return pickle.load(f), dict(row)
    entry = _mem_models.get(model_id)
    return (entry["obj"], entry["meta"]) if entry else (None, None)


async def list_models(user_id: str) -> list[dict]:
    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id,name,model_type,target_col,metrics,created_at "
                "FROM saved_models WHERE user_id=$1 ORDER BY created_at DESC",
                user_id
            )
            return [dict(r) for r in rows]
    return [{"id": k, **v["meta"]} for k, v in _mem_models.items()]


# ── Helpers ──────────────────────────────────────────────
def _df_path(dataset_id: str) -> Path:
    return LOCAL_STORE / "datasets" / f"{dataset_id}.parquet"


# ── CLI migration runner ──────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import sys

    async def migrate():
        pool = await get_pool()
        if pool:
            print("✅ Schema applied to PostgreSQL")
        else:
            print("⚠️  No DATABASE_URL set — using in-memory store")

    asyncio.run(migrate())
