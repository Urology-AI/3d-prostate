#!/bin/bash
# ============================================================
#  pull_sif.sh  —  run this ON Minerva (login node)
#
#  Pulls the Docker image from GHCR and converts it to a .sif file.
#  The image is built automatically by GitHub Actions on every push to main.
#
#  Usage:
#    bash pull_sif.sh           # submits as bsub job (default, recommended)
#    INLINE=1 bash pull_sif.sh  # run directly on login node (may hit thread limits)
# ============================================================

set -euo pipefail

PROJ="/sc/arion/projects/video_rarp/3dprostate"
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/urology-ai/prostate3d:latest}"
SIF="${PROJ}/pipeline.sif"

# ── Load Singularity ──────────────────────────────────────────────────────────
# '|| true' suppresses flatpak/OpenSSL stderr noise from anaconda3 conflict
# that causes ml to exit non-zero even though singularity loads fine
ml singularity || true

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

# ── Decide: run inline or submit as bsub job ─────────────────────────────────
# Login nodes restrict thread creation which kills mksquashfs.
# Default is to submit as bsub (safer). Set INLINE=1 to override.
INLINE="${INLINE:-0}"

if [[ "$INLINE" == "1" ]]; then
  # Single-threaded squashfs to survive login-node thread limits
  export SINGULARITY_MKSQUASHFS_PROCS=1

  if [[ -f "${SIF}" ]]; then
    mv "${SIF}" "${SIF}.bak"
    echo "    Backed up old pipeline.sif → pipeline.sif.bak"
  fi

  singularity build "${SIF}" "docker://${DOCKER_IMAGE}"
  echo ""
  echo "==> Done: $(du -sh "${SIF}" | cut -f1)  →  ${SIF}"

else
  echo "==> Submitting singularity build as a bsub job..."
  echo "    (avoids login-node thread limits that kill mksquashfs)"
  echo ""

  bsub -P acc_video_rarp \
       -q express \
       -n 4 \
       -W 2:00 \
       -R "rusage[mem=32000]" \
       -J pull_sif \
       -o "${PROJ}/pull_sif_%J.log" \
       -e "${PROJ}/pull_sif_%J.err" \
       <<BSUB_SCRIPT
#!/bin/bash
set -euo pipefail

ml singularity || true

export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR}"
export SINGULARITY_MKSQUASHFS_PROCS=4

if [[ -f "${SIF}" ]]; then
  mv "${SIF}" "${SIF}.bak"
  echo "Backed up old pipeline.sif -> pipeline.sif.bak"
fi

echo "Building ${SIF} from docker://${DOCKER_IMAGE} ..."
singularity build "${SIF}" "docker://${DOCKER_IMAGE}"
echo "Done: \$(du -sh "${SIF}") -> ${SIF}"
BSUB_SCRIPT

  echo "==> Job submitted. Monitor with:"
  echo "      bjobs -J pull_sif"
  echo "      tail -f ${PROJ}/pull_sif_*.log"
fi

echo ""
echo "    Once pipeline.sif is ready, start the web server:"
echo "      bsub -P acc_video_rarp -q interactive -n 2 -W 8:00 \\"
echo "           -R \"rusage[mem=16000]\" -Is \\"
echo "           singularity run --contain \\"
echo "             --bind /sc/arion/projects/video_rarp:/sc/arion/projects/video_rarp \\"
echo "             --bind /tmp:/tmp \\"
echo "             ${SIF}"
