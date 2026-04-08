#!/bin/bash
# ============================================================
#  Prostate3D — Start Script
#  Starts the web app and optionally opens SSH tunnel to HPC
# ============================================================

set -e
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'; BOLD='\033[1m'

echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         Prostate3D Pipeline              ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""

# Load env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo -e "${GREEN}[✓]${NC} Loaded .env"
else
    echo -e "${YELLOW}[!]${NC} No .env found — copying from .env.example"
    cp .env.example .env
    echo -e "    Edit .env with your HPC credentials then re-run"
    exit 1
fi

# Check Python deps
echo -e "${CYAN}[→]${NC} Checking dependencies..."
cd backend
pip install -q -r requirements.txt
cd ..

# Start backend
echo -e "${CYAN}[→]${NC} Starting web app on http://localhost:${LOCAL_PORT:-5000}"
cd backend
python app.py &
APP_PID=$!
cd ..

# Wait for app to start
sleep 2

echo ""
echo -e "${GREEN}✓ Prostate3D is running${NC}"
echo -e "  Open: ${CYAN}http://localhost:${LOCAL_PORT:-5000}${NC}"
echo ""
echo -e "  Upload T2-weighted DICOM files"
echo -e "  Pipeline runs on Minerva GPU automatically"
echo ""
echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop"

# Trap cleanup
trap "echo ''; echo 'Stopping...'; kill $APP_PID 2>/dev/null; exit 0" INT TERM

wait $APP_PID
