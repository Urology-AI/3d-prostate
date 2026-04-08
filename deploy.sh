#!/usr/bin/env bash
# ============================================================
#  deploy.sh  —  run locally
#  Creates the Minerva directory structure + copies scripts.
#  The .sif is built on Minerva via pull_sif.sh (not here).
# ============================================================

set -euo pipefail

MINERVA_USER="${MINERVA_USER:-dixita06}"
MINERVA_HOST="${MINERVA_HOST:-minerva.hpc.mssm.edu}"
PROJ="/sc/arion/projects/video_rarp/3dprostate"

SSH="ssh ${MINERVA_USER}@${MINERVA_HOST}"
SCP="scp"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'
log() { echo -e "${CYAN}[→]${NC} $*"; }
ok()  { echo -e "${GREEN}[✓]${NC} $*"; }

echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║      Prostate3D — Minerva Deploy         ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── Directories ───────────────────────────────────────────────────────────────
log "Creating directories on Minerva…"
${SSH} bash -s <<REMOTE
set -euo pipefail
mkdir -p "${PROJ}/seg_jobs"
# Model weights shared across all jobs — downloaded on first run
mkdir -p "${PROJ}/radiology_model"
# MONAILabel server logs
mkdir -p "${PROJ}/radiology_logs"
# Singularity cache/tmp — use project dir not \$HOME to avoid quota issues
mkdir -p "/sc/arion/projects/video_rarp/.singularity_ad/cache"
mkdir -p "/sc/arion/projects/video_rarp/.singularity_ad/tmp"
chmod 755 "${PROJ}" "${PROJ}/seg_jobs" "${PROJ}/radiology_model" "${PROJ}/radiology_logs"
echo "  Done:"
ls -la "${PROJ}/"
REMOTE
ok "Directories ready"

# ── Scripts ───────────────────────────────────────────────────────────────────
log "Copying scripts to Minerva…"
${SCP} scripts/run_segmentation.sh "${MINERVA_USER}@${MINERVA_HOST}:${PROJ}/run_segmentation.sh"
${SCP} scripts/pull_sif.sh         "${MINERVA_USER}@${MINERVA_HOST}:${PROJ}/pull_sif.sh"
${SSH} "chmod +x ${PROJ}/run_segmentation.sh ${PROJ}/pull_sif.sh"
ok "Scripts deployed"

echo ""
echo -e "${GREEN}${BOLD}Deploy complete!${NC}"
echo ""
echo -e "  ${BOLD}Next — build & push the Docker image from your Mac:${NC}"
echo -e "  ${CYAN}bash build_and_push.sh${NC}"
echo ""
echo -e "  ${BOLD}Then on Minerva (login node):${NC}"
echo -e "  ${CYAN}bash ${PROJ}/pull_sif.sh${NC}"
echo ""
echo -e "  ${BOLD}Start web server (interactive compute node):${NC}"
echo -e "  ${CYAN}bsub -P acc_video_rarp -q interactive -n 2 -W 8:00 \\${NC}"
echo -e "  ${CYAN}     -R \"rusage[mem=16000]\" -Is \\${NC}"
echo -e "  ${CYAN}     singularity run --contain \\${NC}"
echo -e "  ${CYAN}       --bind /sc/arion/projects/video_rarp:/sc/arion/projects/video_rarp \\${NC}"
echo -e "  ${CYAN}       --bind /tmp:/tmp \\${NC}"
echo -e "  ${CYAN}       ${PROJ}/pipeline.sif${NC}"
