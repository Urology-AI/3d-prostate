#!/usr/bin/env bash
# ============================================================
#  build_and_push.sh  —  run locally on your Mac (optional)
#
#  NOTE: Production builds happen automatically via GitHub Actions
#  (see .github/workflows/docker.yml) whenever you push to main.
#  The image is pushed to ghcr.io/adidix/prostate3d:latest for free,
#  no Docker Hub account or secrets needed.
#
#  Use this script only when you want to test a local build before
#  pushing, or to iterate faster without waiting for CI.
#
#  On Minerva you then run:
#    bash /sc/arion/projects/video_rarp/3dprostate/pull_sif.sh
#  which does:
#    singularity build pipeline.sif docker://ghcr.io/adidix/prostate3d:latest
#
#  Why cross-compile, not build on Minerva directly?
#    Singularity build from a definition file needs root.
#    You don't have root on Minerva.
#    But singularity build/pull from docker:// works fine without root.
#    So: build Docker image locally → push to GHCR → pull on Minerva.
#
#  Usage:
#    bash build_and_push.sh
# ============================================================

set -euo pipefail

GHCR_USER="${GHCR_USER:-urology-ai}"
IMAGE="ghcr.io/${GHCR_USER}/prostate3d:latest"

MINERVA_USER="${MINERVA_USER:-dixita06}"
MINERVA_HOST="${MINERVA_HOST:-minerva.hpc.mssm.edu}"
MINERVA_PROJ="/sc/arion/projects/video_rarp/3dprostate"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'; BOLD='\033[1m'
log()  { echo -e "${CYAN}[→]${NC} $*"; }
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }

echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Prostate3D — local cross-build (GHCR)   ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── Architecture warning ──────────────────────────────────────────────────────
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
  warn "Mac arm64 detected → cross-compiling to linux/amd64 via QEMU."
  warn "First build is slow (30-60 min) due to monailabel + CUDA torch."
  warn "Consider letting GitHub Actions do production builds instead."
  echo ""
fi

# ── Ensure a buildx builder with multi-arch / QEMU support ───────────────────
if ! docker buildx inspect prostate3d-builder &>/dev/null; then
  log "Creating buildx builder 'prostate3d-builder'…"
  docker buildx create --name prostate3d-builder --use --bootstrap
else
  docker buildx use prostate3d-builder
  log "Using existing buildx builder 'prostate3d-builder'"
fi

# ── Log in to GHCR (gh auth token or a PAT with write:packages) ──────────────
if ! docker login ghcr.io -u "${GHCR_USER}" --password-stdin \
       <<< "$(gh auth token 2>/dev/null)" 2>/dev/null; then
  warn "Could not auto-login via gh CLI. Run:"
  warn "  echo \$CR_PAT | docker login ghcr.io -u ${GHCR_USER} --password-stdin"
  exit 1
fi

# ── Build for linux/amd64 and push to GHCR ───────────────────────────────────
log "Building ${IMAGE} for linux/amd64…"
docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    -t "${IMAGE}" \
    -f Dockerfile \
    --push \
    .
ok "Pushed to GHCR: docker pull ${IMAGE}"

# ── Copy pull_sif.sh to Minerva ───────────────────────────────────────────────
log "Copying pull_sif.sh to Minerva…"
scp scripts/pull_sif.sh \
    "${MINERVA_USER}@${MINERVA_HOST}:${MINERVA_PROJ}/pull_sif.sh"
ssh "${MINERVA_USER}@${MINERVA_HOST}" \
    "chmod +x ${MINERVA_PROJ}/pull_sif.sh"
ok "pull_sif.sh ready"

echo ""
echo -e "${GREEN}${BOLD}Done!${NC}"
echo ""
echo -e "  Now on Minerva (login node):"
echo -e "  ${CYAN}bash ${MINERVA_PROJ}/pull_sif.sh${NC}"
echo ""
echo -e "  Or just push to GitHub and let CI build it automatically."
