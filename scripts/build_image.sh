#!/bin/bash
# Build and push the DSL Debug training image to Docker Hub
#
# This image bundles:
#   - verlai/verl:sgl056.latest base (torch 2.9.1, sglang 0.5.6, flashinfer)
#   - verl 0.7.0 + retry patch for TP>=2 startup
#   - dsl-debug package + CLI
#   - Qwen2.5-7B-Instruct model (~14GB, cached in image layer)
#   - SFT + RL training data parquets
#
# Usage:
#   bash scripts/build_image.sh          # build + push latest
#   bash scripts/build_image.sh --no-push  # build only (local test)
#
# Prerequisites:
#   docker login                         # must be logged in to Docker Hub
#   docker buildx create --use           # only needed once
#
# Image: ${DOCKER_REPO:-andrewlngdn}/dsl-debug-train:latest
#
# Notes:
#   - First build takes ~30-45 min (model download, pip installs)
#   - Code-only rebuilds are fast: model layer is cached above COPY . .
#   - Uses --platform linux/amd64 for Vast.ai (required on Apple Silicon)

set -euo pipefail

IMAGE="${DOCKER_REPO:-andrewlngdn}/dsl-debug-train"
TAG="${TAG:-latest}"
FULL_IMAGE="${IMAGE}:${TAG}"
PUSH=true

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --no-push) PUSH=false ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# Run from repo root
cd "$(dirname "$0")/.."

echo "Building: $FULL_IMAGE"
echo "Platform: linux/amd64"
echo "Push: $PUSH"
echo ""

if $PUSH; then
    docker buildx build \
        --platform linux/amd64 \
        -t "$FULL_IMAGE" \
        --push \
        .
else
    docker buildx build \
        --platform linux/amd64 \
        -t "$FULL_IMAGE" \
        --load \
        .
fi

echo ""
echo "Done! Image: $FULL_IMAGE"
if $PUSH; then
    echo "Use on Vast.ai: bash scripts/vast.sh create"
fi
