#!/bin/bash
# ============================================================
#  pull_sif.sh  —  run this ON a Minerva compute node
#
#  First get an interactive node:
#    bsub -P acc_video_rarp -q interactive -n 4 -W 2:00 \
#         -R "rusage[mem=32000]" -Is /bin/bash
#
#  Then run:
#    ml singularity && bash pull_sif.sh
# ============================================================

PROJ="/sc/arion/projects/video_rarp/3dprostate"
DOCKER_IMAGE="ghcr.io/urology-ai/prostate3d:latest"
SIF="${PROJ}/pipeline.sif"

export SINGULARITY_CACHEDIR="/sc/arion/projects/video_rarp/.singularity_ad/cache"
export SINGULARITY_TMPDIR="/sc/arion/projects/video_rarp/.singularity_ad/tmp"
export SINGULARITY_MKSQUASHFS_PROCS=4

mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}" "${PROJ}"

if ! command -v singularity &>/dev/null; then
    echo "ERROR: singularity not in PATH. Run: ml singularity"
    exit 1
fi

echo "==> singularity: $(singularity --version)"
echo "==> Building: ${SIF}"
echo "    from: docker://${DOCKER_IMAGE}"
echo ""

[ -f "${SIF}" ] && mv "${SIF}" "${SIF}.bak" && echo "==> Backed up old pipeline.sif"

singularity build "${SIF}" "docker://${DOCKER_IMAGE}"

echo ""
echo "==> Done: $(du -sh ${SIF})"
