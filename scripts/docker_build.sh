#!/bin/bash
# ============================================================
#  Build and push all Docker images to Docker Hub
#  Usage: ./scripts/docker_build.sh [dockerhub-user] [tag] [platforms]
#  Example (HPC-safe default):
#    ./scripts/docker_build.sh adidix latest
#  Example (optional multi-arch):
#    ./scripts/docker_build.sh adidix latest linux/amd64,linux/arm64
# ============================================================

set -e

REGISTRY="${1:-adidix}"
VERSION="${2:-latest}"
PLATFORMS="${3:-linux/amd64}"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'

echo -e "${BOLD}Prostate3D — Docker Build & Push${NC}"
echo "Registry: $REGISTRY"
echo "Version:  $VERSION"
echo "Platform: $PLATFORMS"
echo ""

cd "$(dirname "$0")/.."

# ── GPU image ─────────────────────────────────────────────
echo -e "${CYAN}[1/3]${NC} Building GPU image (CUDA 11.8)..."
docker buildx build \
    --platform "$PLATFORMS" \
    -t "$REGISTRY/prostate3d-monai:$VERSION" \
    -t "$REGISTRY/prostate3d-monai:gpu" \
    -f docker/Dockerfile.monai \
    --push \
    .
echo -e "${GREEN}[✓]${NC} GPU image pushed: $REGISTRY/prostate3d-monai:$VERSION"

# ── CPU image ─────────────────────────────────────────────
echo -e "${CYAN}[2/3]${NC} Building CPU image..."
docker buildx build \
    --platform "$PLATFORMS" \
    -t "$REGISTRY/prostate3d-monai-cpu:$VERSION" \
    -t "$REGISTRY/prostate3d-monai-cpu:cpu" \
    -f docker/Dockerfile.cpu \
    --push \
    .
echo -e "${GREEN}[✓]${NC} CPU image pushed: $REGISTRY/prostate3d-monai-cpu:$VERSION"

# ── Web image ─────────────────────────────────────────────
echo -e "${CYAN}[3/3]${NC} Building web app image..."
docker buildx build \
    --platform "$PLATFORMS" \
    -t "$REGISTRY/prostate3d-web:$VERSION" \
    -f docker/Dockerfile.web \
    --push \
    .
echo -e "${GREEN}[✓]${NC} Web image pushed: $REGISTRY/prostate3d-web:$VERSION"

echo ""
echo -e "${BOLD}All images pushed to Docker Hub!${NC}"
echo ""
echo "Pull on HPC:"
echo "  singularity pull monai.sif docker://$REGISTRY/prostate3d-monai:$VERSION"
echo ""
echo "Run locally (CPU):"
echo "  docker run -p 8000:8000 -v ./dicom:/data/studies $REGISTRY/prostate3d-monai-cpu:$VERSION"
