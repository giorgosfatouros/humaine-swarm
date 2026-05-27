#!/usr/bin/env bash
# Build humaine-swarm, tag for Docker Hub, optionally push.
# Usage (from repo root):
#   bash .cursor/skills/humaine-swarm-docker-publish/scripts/publish-amd64.sh
#   SKIP_PUSH=1 bash ...          # build + tag only
#   PLATFORM=linux/arm64 bash ... # override platform (default: auto-detect)

set -euo pipefail

REPO="gfatouros/humaine-swarm"
LOCAL_TAG="humaine-swarm:latest"
HUB_TAG="${REPO}:amd64"

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

if [[ -z "${PLATFORM:-}" ]]; then
  ARCH="$(uname -m)"
  if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
    PLATFORM="linux/amd64"
    echo "Detected ARM host — building for ${PLATFORM}"
  else
    PLATFORM=""
    echo "Detected x86 host — native build"
  fi
fi

BUILD_ARGS=(-t "$LOCAL_TAG" .)
if [[ -n "$PLATFORM" ]]; then
  BUILD_ARGS=(--platform "$PLATFORM" "${BUILD_ARGS[@]}")
fi

echo "==> docker build ${BUILD_ARGS[*]}"
docker build "${BUILD_ARGS[@]}"

echo "==> Tagging ${HUB_TAG}"
docker tag "$LOCAL_TAG" "$HUB_TAG"
docker tag "$LOCAL_TAG" "${REPO}:latest" 2>/dev/null || true

echo "==> Image IDs (must match):"
docker images --format '  {{.Repository}}:{{.Tag}} {{.ID}}' | grep -E 'humaine-swarm|gfatouros/humaine' || true

if [[ "${SKIP_PUSH:-0}" == "1" ]]; then
  echo "SKIP_PUSH=1 — done (not pushing)"
  exit 0
fi

echo "==> docker push ${HUB_TAG}"
docker push "$HUB_TAG"

echo "==> Push complete. Digest above — share with ops before they pull."
