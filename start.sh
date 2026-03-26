#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PIDS=()

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}  [ok]${NC} $*"; }
info() { echo -e "${CYAN}  -->  ${NC}$*"; }
warn() { echo -e "${YELLOW}  [!]  ${NC}$*"; }
fail() { echo -e "${RED}  [x]  ${NC}$*"; exit 1; }
step() { echo -e "\n${BOLD}$*${NC}"; }

# ── Cleanup ─────────────────────────────────────────────────────────────────
cleanup() {
  echo -e "\n${YELLOW}Stopping all services...${NC}"
  for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

# ── 1. System dependencies ──────────────────────────────────────────────────
step "Checking system dependencies..."

# Homebrew
if ! command -v brew &>/dev/null; then
  warn "Homebrew not found — installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/homebrew/install/HEAD/install.sh)"
fi
ok "Homebrew"

# Python 3.11+
if ! command -v python3 &>/dev/null || ! python3 -c "import sys; assert sys.version_info >= (3,11)" 2>/dev/null; then
  info "Installing Python 3.11 via Homebrew..."
  brew install python@3.11
fi
PYTHON=$(command -v python3.11 || command -v python3)
ok "Python ($($PYTHON --version))"

# Node.js
if ! command -v node &>/dev/null; then
  info "Installing Node.js via Homebrew..."
  brew install node
fi
ok "Node.js ($(node --version))"

# ffmpeg
if ! command -v ffmpeg &>/dev/null; then
  info "Installing ffmpeg via Homebrew..."
  brew install ffmpeg
fi
ok "ffmpeg"

# PostgreSQL
if ! command -v psql &>/dev/null; then
  info "Installing PostgreSQL 16 via Homebrew..."
  brew install postgresql@16
  brew link postgresql@16 --force
fi
ok "PostgreSQL"

# ── 2. PostgreSQL running ────────────────────────────────────────────────────
step "Checking PostgreSQL..."

if ! pg_isready -q 2>/dev/null; then
  info "Starting PostgreSQL..."
  # Detect installed postgresql service name
  PG_SERVICE=$(brew services list | awk '/postgresql/ {print $1}' | head -1)
  if [[ -n "$PG_SERVICE" ]]; then
    brew services start "$PG_SERVICE" 2>/dev/null || true
    sleep 2
  fi
fi

if ! pg_isready -q 2>/dev/null; then
  PG_SERVICE=$(brew services list | awk '/postgresql/ {print $1}' | head -1)
  fail "PostgreSQL is not running. Start it with: brew services start ${PG_SERVICE:-postgresql}"
fi
ok "PostgreSQL is running"

# ── 3. Database + user ──────────────────────────────────────────────────────
step "Setting up database..."

# Create role if missing
psql postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='barter'" | grep -q 1 || \
  psql postgres -c "CREATE ROLE barter LOGIN PASSWORD 'barter';"

# Create DB if missing
psql postgres -tAc "SELECT 1 FROM pg_database WHERE datname='barter_db'" | grep -q 1 || \
  psql postgres -c "CREATE DATABASE barter_db OWNER barter;"

ok "Database barter_db ready"

# Run migrations
info "Running migrations..."
cd "$ROOT/backend"
MIGRATION_DIR="$ROOT/backend/migrations"
for f in "$MIGRATION_DIR"/*.sql; do
  psql "postgresql://barter:barter@localhost:5432/barter_db" -f "$f" -q 2>/dev/null || true
done
ok "Migrations applied"

# ── 4. Python venvs + packages ───────────────────────────────────────────────
step "Checking Python environments..."

setup_venv() {
  local dir="$1"
  local name="$2"
  cd "$ROOT/$dir"
  if [[ ! -d venv ]]; then
    info "Creating venv for ${name}..."
    $PYTHON -m venv venv
  fi
  # shellcheck disable=SC1091
  source venv/bin/activate
  info "Installing packages for ${name}..."
  pip install -q -r requirements.txt
  deactivate
  ok "${name}"
  cd "$ROOT"
}

setup_venv backend           "Backend"
setup_venv audio_pipeline    "Audio Pipeline"
setup_venv semantic_analysis "Semantic Analysis"
setup_venv warning_engine    "Warning Engine"

# ── 5. Frontend node_modules ────────────────────────────────────────────────
step "Checking frontend..."

cd "$ROOT/frontend"
if [[ ! -d node_modules ]]; then
  info "Installing npm packages..."
  npm install --silent
fi
ok "Frontend"
cd "$ROOT"

# ── 6. Launch services ───────────────────────────────────────────────────────
step "Starting services..."

launch() {
  local label="$1" dir="$2" entry="$3" port="$4"
  cd "$ROOT/$dir"
  # shellcheck disable=SC1091
  source venv/bin/activate
  uvicorn "$entry" --host 0.0.0.0 --port "$port" --log-level warning 2>&1 \
    | sed "s/^/  [${label}] /" &
  PIDS+=($!)
  deactivate
  cd "$ROOT"
}

launch "backend " backend           app.main:app 8000
launch "audio   " audio_pipeline    main:app     8001
launch "semantic" semantic_analysis main:app     8002
launch "warning " warning_engine    main:app     8003

# Wait for services to be up
info "Waiting for services to start..."
sleep 3

for port in 8000 8001 8002 8003; do
  if curl -sf "http://localhost:$port/health" &>/dev/null; then
    ok "Port $port"
  else
    warn "Port $port not responding yet (still loading model?)"
  fi
done

# ── 7. Frontend dev server ───────────────────────────────────────────────────
cd "$ROOT/frontend"
npm run dev --silent 2>&1 | sed 's/^/  [frontend] /' &
PIDS+=($!)
cd "$ROOT"

echo ""
echo -e "${GREEN}${BOLD}All services running.${NC}"
echo -e "  Backend:   http://localhost:8000"
echo -e "  Frontend:  http://localhost:5173"
echo ""
echo -e "${YELLOW}Ctrl+C to stop everything.${NC}"
echo ""

wait
