#!/bin/bash
# ============================================================
#  MONAILabel Server Entrypoint
# ============================================================

set -e

APP_PATH="${APP_PATH:-/app/radiology}"
STUDIES_PATH="${STUDIES_PATH:-/data/studies}"
MODEL="${MODEL:-prostate_mri_anatomy}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "╔══════════════════════════════════════════╗"
echo "║   Prostate3D — MONAILabel Server         ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  App:     $APP_PATH"
echo "  Studies: $STUDIES_PATH"
echo "  Model:   $MODEL"
echo "  URL:     http://0.0.0.0:$PORT"
echo ""

# Verify app
if [ ! -f "$APP_PATH/main.py" ]; then
    echo "ERROR: $APP_PATH/main.py not found"
    ls /app/ 2>/dev/null
    exit 1
fi

# On first run, model weights are downloaded from HuggingFace
# This can take 2-5 minutes — subsequent starts are instant
echo "Starting server (first run downloads model weights ~500MB)..."
echo ""

exec monailabel start_server \
    --app "$APP_PATH" \
    --studies "$STUDIES_PATH" \
    --conf models "$MODEL" \
    --host "$HOST" \
    --port "$PORT"
