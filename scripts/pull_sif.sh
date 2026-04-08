#!/bin/bash
# ============================================================
#  pull_sif.sh  —  run this ON Minerva (login node)
#
#  Pulls the Docker image from GHCR and converts it to a .sif file.
#  The image is built automatically by GitHub Actions on every push to main.
#
#  Usage:
#    bash pull_sif.sh
# ============================================================

set -euo pipefail

PROJ="/sc/arion/projects/video_rarp/3dprostate"
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/urology-ai/prostate3d:latest}"
SIF="${PROJ}/pipeline.sif"

# ── Load Singularity ──────────────────────────────────────────────────────────
ml singularity

# ── Cache / tmp in project dir (avoids $HOME quota and path-doubling bug) ────
SING_DIR="/sc/arion/projects/video_rarp/.singularity_ad"
export SINGULARITY_CACHEDIR="${SING_DIR}/cache"
export SINGULARITY_TMPDIR="${SING_DIR}/tmp"
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

echo "==> Pulling docker://${DOCKER_IMAGE}"
echo "    cache : ${SINGULARITY_CACHEDIR}"
echo "    tmp   : ${SINGULARITY_TMPDIR}"
echo "    output: ${SIF}"
echo ""

# Back up existing .sif if present
if [[ -f "${SIF}" ]]; then
  mv "${SIF}" "${SIF}.bak"
  echo "    Backed up old pipeline.sif → pipeline.sif.bak"
fi

# Build .sif directly from Docker Hub (no root needed)
singularity build "${SIF}" "docker://${DOCKER_IMAGE}"

echo ""
echo "==> Done: $(du -sh "${SIF}" | cut -f1)  →  ${SIF}"
echo ""
echo "    Clean up cache when done:"
echo "      singularity cache clean"
echo ""
echo "    Start web server (interactive node):"
echo "      bsub -P acc_video_rarp -q interactive -n 2 -W 8:00 \\"
echo "           -R \"rusage[mem=16000]\" -Is \\"
echo "           singularity run --contain \\"
echo "             --bind /sc/arion/projects/video_rarp:/sc/arion/projects/video_rarp \\"
echo "             --bind /tmp:/tmp \\"
echo "             ${SIF}"
