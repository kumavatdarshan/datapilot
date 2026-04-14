# DataPilot v3 — Monetized Data Science Platform

```
Browser  ←→  FastAPI  ←→  pandas · scikit-learn · statsmodels
                ↕               ↕
             Stripe          PostgreSQL + local disk
                ↕               ↕
            JWT Auth        Parquet files
```

## What's new in v3

| | v2 | v3 |
|---|---|---|
| Auth | ❌ None | ✅ JWT (register/login) |
| Payments | ❌ None | ✅ Stripe subscriptions |
| Tiers | ❌ None | ✅ Free / Pro ($29) / Team ($149) |
| Rate limiting | ❌ None | ✅ Per-user, per-tier |
| Data persistence | ❌ In-memory (lost on restart) | ✅ PostgreSQL + Parquet |
| Save models | ❌ None | ✅ Pickle + DB metadata |
| AI queries | ⚠️ Regex only | ✅ Claude API fallback |
| Pricing page | ❌ None | ✅ `/pricing` |

## Quick start

```bash
# 1. Copy env vars
cp .env.example .env
# Edit .env with your values (Stripe keys, DB URL, etc.)

# 2. Start
chmod +x start.sh
./start.sh

# 3. Open
open http://localhost:8765/pricing   # pricing + sign-up
open http://localhost:8765           # the app
```

## Without any setup (zero config)

The app works out of the box using in-memory storage:
- No database needed — sessions live in RAM
- No Stripe keys — auth endpoints work, checkout is disabled
- No Claude key — AI query falls back to rule engine

This is fine for development. For production, set `DATABASE_URL` at minimum.

## Tier limits

| | Free | Pro | Team |
|---|---|---|---|
| Max rows | 500 | 500,000 | 5,000,000 |
| Max file | 2 MB | 50 MB | 500 MB |
| Saved datasets | 1 | 50 | 500 |
| ML training | ❌ | ✅ | ✅ |
| Forecasting | ❌ | ✅ | ✅ |
| AI queries | ❌ | ✅ | ✅ |
| PDF reports | ❌ | ✅ | ✅ |
| API calls/day | 20 | 2,000 | 20,000 |

## API auth

All endpoints (except `/health`, `/auth/*`, `/stripe/pricing`) require a Bearer token:

```bash
# Register
curl -X POST http://localhost:8765/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"yourpassword"}'
# → {"token": "eyJ...", "user": {...}}

# Use the token
curl http://localhost:8765/auth/me \
  -H "Authorization: Bearer eyJ..."

# Upload a file
curl -X POST "http://localhost:8765/upload" \
  -H "Authorization: Bearer eyJ..." \
  -F "file=@data.csv"
```

## Stripe setup (10 minutes)

1. Create account at [stripe.com](https://stripe.com)
2. Go to **Products** → Create two products:
   - "DataPilot Pro" — $29/month recurring → copy the **Price ID** (`price_xxx`)
   - "DataPilot Team" — $149/month recurring → copy the **Price ID**
3. Go to **Developers → API keys** → copy Secret key (`sk_live_xxx`)
4. Go to **Developers → Webhooks** → Add endpoint:
   - URL: `https://yourdomain.com/stripe/webhook`
   - Events: `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted`
   - Copy the signing secret (`whsec_xxx`)
5. Paste all four values in `.env`

## Database setup (Supabase, free)

1. Go to [supabase.com](https://supabase.com) → New project
2. Settings → Database → Connection string → copy URI
3. Set `DATABASE_URL=postgresql://...` in `.env`
4. Schema auto-applies on first run

## Production deployment

### Railway (recommended, ~$5/mo)
```bash
railway login
railway init
railway add --database postgresql
railway up
railway variables set JWT_SECRET=... STRIPE_SECRET_KEY=... # etc
```

### Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8765
CMD ["python", "app.py"]
```

## Revenue model

| Action | Revenue |
|---|---|
| User upgrades Free → Pro | $29/mo recurring |
| User upgrades to Team | $149/mo recurring |
| Annual Pro (20% discount) | $23/mo = $276/yr |
| Annual Team (20% discount) | $119/mo = $1,428/yr |
| 100 Pro users | $2,900/mo |
| 10 Team users + 90 Pro | $4,100/mo |

## File structure

```
datapilot/
├── app.py            ← Main FastAPI app (all endpoints)
├── auth.py           ← JWT auth, tier enforcement, rate limiting
├── payments.py       ← Stripe checkout, webhooks, pricing API
├── db.py             ← PostgreSQL + parquet storage layer
├── requirements.txt  ← All dependencies
├── start.sh          ← One-command startup
├── .env.example      ← Config template
└── frontend/
    ├── index.html    ← Main app UI (from v2)
    └── pricing.html  ← Pricing page with auth modal
```
