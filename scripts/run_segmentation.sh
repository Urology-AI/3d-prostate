#!/bin/bash
# ============================================================
#  run_segmentation.sh — LSF GPU job script
#
#  Submitted automatically by app.py via  bsub < job.sh
#  Can also be submitted manually:
#    JOB_ID=abc12345 bash scripts/run_segmentation.sh
# ============================================================

# ── LSF directives ────────────────────────────────────────────────────────────
#BSUB -J prostate3d_${JOB_ID:-manual}
#BSUB -P acc_video_rarp
#BSUB -q gpu
#BSUB -n 4
#BSUB -W 4:00
#BSUB -R "rusage[mem=32000,ngpus_phys=1]"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -o /sc/arion/projects/video_rarp/3dprostate/seg_jobs/${JOB_ID:-manual}/job.log
#BSUB -e /sc/arion/projects/video_rarp/3dprostate/seg_jobs/${JOB_ID:-manual}/job.err

set -euo pipefail

JOB_ID="${1:-${JOB_ID:-}}"
if [[ -z "$JOB_ID" ]]; then
  echo "Usage: JOB_ID=<hex8> $0" >&2
  exit 1
fi

PROJ="/sc/arion/projects/video_rarp/3dprostate"
SIF_PATH="${SIF_PATH:-${PROJ}/pipeline.sif}"
JOBS_BASE="${JOBS_DIR:-${PROJ}/seg_jobs}"
JOB_DIR="${JOBS_BASE}/${JOB_ID}"

echo "============================================================"
echo "  Prostate3D  |  job=${JOB_ID}  |  node=$(hostname)"
echo "  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "  SIF: ${SIF_PATH}"
echo "  Started: $(date)"
echo "============================================================"

[[ -f "${SIF_PATH}" ]]    || { echo "ERROR: SIF not found: ${SIF_PATH}"; exit 1; }
[[ -d "${JOB_DIR}/input" ]] || { echo "ERROR: input dir missing: ${JOB_DIR}/input"; exit 1; }

mkdir -p "${JOB_DIR}/output"

ml singularity

# Proxy — needed for first-run model weight download
export http_proxy=http://172.28.7.1:3128
export https_proxy=http://172.28.7.1:3128
export all_proxy=http://172.28.7.1:3128
export no_proxy=localhost,*.chimera.hpc.mssm.edu,172.28.0.0/16

# radiology_model: model weights shared across all jobs (downloaded once)
# radiology_logs:  server logs on writable /sc/arion storage
singularity exec \
    --nv \
    --contain \
    --bind /sc/arion/projects/video_rarp:/sc/arion/projects/video_rarp \
    --bind "${PROJ}/radiology_model:/app/radiology/model" \
    --bind "${PROJ}/radiology_logs:/app/radiology/logs" \
    --bind /tmp:/tmp \
    "${SIF_PATH}" \
    python3 /app/segment_prostate.py \
        --job-id    "${JOB_ID}" \
        --jobs-base "${JOBS_BASE}"

EXIT=$?
echo "Job finished with exit code ${EXIT} at $(date)"
exit ${EXIT}
