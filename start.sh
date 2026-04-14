#!/bin/bash
set -e
echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║      DataPilot v3  —  Monetized 🚀       ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

if ! command -v python3 &>/dev/null; then echo "  ❌ python3 not found"; exit 1; fi

# Install dependencies if needed
python3 -c "import fastapi,pandas,sklearn,statsmodels,scipy,jwt,bcrypt,stripe,httpx" 2>/dev/null || {
  echo "  📦 Installing dependencies..."
  pip install -r requirements.txt -q
}

# Load .env if it exists
if [ -f .env ]; then
  echo "  ✅ Loading .env"
  export $(grep -v '^#' .env | xargs)
fi

# Warn about missing config
[ -z "$JWT_SECRET" ]          && echo "  ⚠️  JWT_SECRET not set — using insecure default"
[ -z "$STRIPE_SECRET_KEY" ]   && echo "  ⚠️  STRIPE_SECRET_KEY not set — payments disabled"
[ -z "$DATABASE_URL" ]        && echo "  ⚠️  DATABASE_URL not set — using in-memory store (data lost on restart)"
[ -z "$ANTHROPIC_API_KEY" ]   && echo "  ⚠️  ANTHROPIC_API_KEY not set — AI queries disabled"

PORT=${PORT:-8765}
echo ""
echo "  → App:      http://localhost:${PORT}"
echo "  → Pricing:  http://localhost:${PORT}/pricing"
echo "  → API docs: http://localhost:${PORT}/api/docs"
echo "  Press Ctrl+C to stop."
echo ""
python3 app.py
