#!/bin/bash
# ============================================================
#  pull_sif.sh  —  run this ON Minerva (login node)
#  Pulls ghcr.io/urology-ai/prostate3d:latest → pipeline.sif
# ============================================================

PROJ="/sc/arion/projects/video_rarp/3dprostate"
DOCKER_IMAGE="ghcr.io/urology-ai/prostate3d:latest"
SIF="${PROJ}/pipeline.sif"

export SINGULARITY_CACHEDIR="/sc/arion/projects/video_rarp/.singularity_ad/cache"
export SINGULARITY_TMPDIR="/sc/arion/projects/video_rarp/.singularity_ad/tmp"
export SINGULARITY_MKSQUASHFS_PROCS=1

mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}" "${PROJ}"

echo "==> Loading singularity module..."
ml singularity 2>&1 | grep -v flatpak | grep -v OPENSSL || true

if ! command -v singularity &>/dev/null; then
    echo "ERROR: singularity not in PATH after ml singularity"
    exit 1
fi
echo "    singularity: $(singularity --version)"

echo ""
echo "==> Submitting build job to LSF..."
echo "    image : docker://${DOCKER_IMAGE}"
echo "    output: ${SIF}"
echo "    cache : ${SINGULARITY_CACHEDIR}"
echo ""

bsub \
  -P acc_video_rarp \
  -q express \
  -n 4 \
  -W 2:00 \
  -R "rusage[mem=32000]" \
  -J pull_sif \
  -o "${PROJ}/pull_sif_%J.log" \
  -e "${PROJ}/pull_sif_%J.err" \
  /bin/bash -c "
    ml singularity 2>/dev/null || true
    export SINGULARITY_CACHEDIR='${SINGULARITY_CACHEDIR}'
    export SINGULARITY_TMPDIR='${SINGULARITY_TMPDIR}'
    export SINGULARITY_MKSQUASHFS_PROCS=4
    [ -f '${SIF}' ] && mv '${SIF}' '${SIF}.bak' && echo 'Backed up old pipeline.sif'
    echo 'Building ${SIF} ...'
    singularity build '${SIF}' 'docker://${DOCKER_IMAGE}'
    echo 'Done:' \$(du -sh '${SIF}')
  "

echo "==> Job submitted. Monitor with:"
echo "      bjobs -J pull_sif"
echo "      tail -f ${PROJ}/pull_sif_*.log"
echo ""
echo "    Once done, start the web server:"
echo "      bsub -P acc_video_rarp -q interactive -n 2 -W 8:00 \\"
echo "           -R 'rusage[mem=16000]' -Is \\"
echo "           singularity run --contain \\"
echo "             --bind /sc/arion/projects/video_rarp:/sc/arion/projects/video_rarp \\"
echo "             --bind /tmp:/tmp \\"
echo "             ${SIF}"
