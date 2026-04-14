"""
DataPilot — Payments (Razorpay)
================================
Currently in setup mode — no Razorpay account connected yet.
The app runs perfectly without it. Upgrade buttons will show
a "coming soon" message until you connect Razorpay.

When you're ready to connect Razorpay, set these env vars:
    RAZORPAY_KEY_ID       = rzp_live_xxxxxxxxxxxx
    RAZORPAY_KEY_SECRET   = xxxxxxxxxxxxxxxxxxxx
    RAZORPAY_PRO_PLAN_ID  = plan_xxxxxxxxxxxxxxxxx
    RAZORPAY_TEAM_PLAN_ID = plan_xxxxxxxxxxxxxxxxx
"""

import os
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from auth import CurrentUser, get_current_user, TIERS

# ── Config ──────────────────────────────────────────────────
RAZORPAY_KEY_ID       = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET   = os.environ.get("RAZORPAY_KEY_SECRET", "")
RAZORPAY_PRO_PLAN_ID  = os.environ.get("RAZORPAY_PRO_PLAN_ID", "")
RAZORPAY_TEAM_PLAN_ID = os.environ.get("RAZORPAY_TEAM_PLAN_ID", "")

PAYMENTS_ENABLED = bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET)

router = APIRouter(prefix="/stripe", tags=["payments"])


# ── Pricing info ─────────────────────────────────────────────
@router.get("/pricing")
async def get_pricing():
    """Returns tier features for the pricing page."""
    return {
        "tiers": [
            {
                "id":    "free",
                "name":  "Free",
                "price": 0,
                "features": [
                    "Up to 100 rows (demo)",
                    "Profile & EDA charts",
                    "Data cleaning tools",
                    "CSV export",
                ],
                "locked": [
                    "ML model training",
                    "Time series forecasting",
                    "AI natural language queries",
                    "PDF report generation",
                ],
                "cta": "Get started free",
                "stripe_price_id": None,
            },
            {
                "id":      "pro",
                "name":    "Pro",
                "price":   15,
                "popular": True,
                "features": [
                    "Up to 500,000 rows",
                    "50 saved datasets",
                    "Full ML suite (6 algorithms)",
                    "ARIMA & ETS forecasting",
                    "AI natural language queries",
                    "PDF reports",
                    "Files up to 50 MB",
                    "2,000 API calls/day",
                ],
                "locked": [],
                "cta": "Start Pro — ₹15/mo",
                "stripe_price_id": RAZORPAY_PRO_PLAN_ID or None,
            },
            {
                "id":   "team",
                "name": "Team",
                "price": 45,
                "features": [
                    "Up to 5 million rows",
                    "500 saved datasets",
                    "Everything in Pro",
                    "20 team seats",
                    "White-label PDF reports",
                    "REST API access",
                    "Files up to 500 MB",
                    "20,000 API calls/day",
                    "Priority support",
                ],
                "locked": [],
                "cta": "Start Team — ₹45/mo",
                "stripe_price_id": RAZORPAY_TEAM_PLAN_ID or None,
            },
        ]
    }


# ── Checkout ─────────────────────────────────────────────────
class CheckoutRequest(BaseModel):
    tier: str
    success_url: str = "/?upgraded=1"
    cancel_url:  str = "/pricing"


@router.post("/create-checkout")
async def create_checkout(
    req: CheckoutRequest,
    user: CurrentUser = Depends(get_current_user),
):
    if not PAYMENTS_ENABLED:
        return JSONResponse({
            "error":      "payments_not_configured",
            "message":    "Payments are being set up. Please check back soon or contact support.",
            "contact":    "kumavatdarshan8@gmail.com",
            "upgrade_url": "/pricing",
        }, status_code=503)

    # ── Razorpay live integration (runs when keys are set) ──
    try:
        import razorpay
        client = razorpay.Client(
            auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
        )
        plan_id = (
            RAZORPAY_PRO_PLAN_ID
            if req.tier == "pro"
            else RAZORPAY_TEAM_PLAN_ID
        )
        subscription = client.subscription.create({
            "plan_id":       plan_id,
            "total_count":   120,   # 10 years max
            "quantity":      1,
            "customer_notify": 1,
            "notes": {
                "user_id": user.user_id,
                "email":   user.email,
                "tier":    req.tier,
            }
        })
        return {
            "subscription_id": subscription["id"],
            "razorpay_key":    RAZORPAY_KEY_ID,
            "tier":            req.tier,
            "amount":          1500 if req.tier == "pro" else 4500,
            "currency":        "INR",
            "name":            "DataPilot",
            "description":     f"DataPilot {req.tier.title()} — Monthly",
            "prefill": {
                "email": user.email,
            }
        }
    except ImportError:
        return JSONResponse({
            "error":   "razorpay_not_installed",
            "message": "Add razorpay to requirements.txt",
        }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "error":   "checkout_failed",
            "message": str(e),
        }, status_code=500)


# ── Customer portal ───────────────────────────────────────────
@router.post("/create-portal")
async def create_portal(
    user: CurrentUser = Depends(get_current_user),
):
    if not PAYMENTS_ENABLED:
        return JSONResponse({
            "error":   "payments_not_configured",
            "message": "Payments are being set up. Please check back soon.",
        }, status_code=503)
    return {"portal_url": "https://razorpay.com"}


# ── Webhook ───────────────────────────────────────────────────
from fastapi import Request, HTTPException
import hmac, hashlib, json

@router.post("/webhook")
async def razorpay_webhook(request: Request):
    """
    Razorpay sends payment events here.
    Upgrades user tier when subscription is activated or charged.
    Downgrades to free when subscription is cancelled.
    """
    payload   = await request.body()
    signature = request.headers.get("x-razorpay-signature", "")
    secret    = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")

    # Verify signature if secret is set
    if secret:
        expected = hmac.new(
            secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise HTTPException(400, "Invalid webhook signature")

    try:
        event = json.loads(payload)
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    event_type = event.get("event", "")
    entity     = event.get("payload", {}).get("subscription", {}).get("entity", {})
    notes      = entity.get("notes", {})
    user_id    = notes.get("user_id")
    tier       = notes.get("tier", "pro")

    import db
    if event_type in ("subscription.activated", "subscription.charged"):
        if user_id:
            await db.update_user_tier(user_id, tier)
            print(f"[Payments] User {user_id} upgraded to {tier}")

    elif event_type == "subscription.cancelled":
        if user_id:
            await db.update_user_tier(user_id, "free")
            print(f"[Payments] User {user_id} downgraded to free")

    return {"received": True}