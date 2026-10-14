#!/usr/bin/env bash
# Build humaine-swarm, tag for Docker Hub, optionally push.
# Every release gets an immutable versioned tag (YYYYMMDD-<git sha>), so
# previous versions are never lost; :latest is the only moving alias.
# The image always targets typical Linux servers (linux/amd64), whatever the
# build host (Apple Silicon builds under emulation).
# Usage (from repo root):
#   bash .cursor/skills/humaine-swarm-docker-publish/scripts/publish-amd64.sh
#   SKIP_PUSH=1 bash ...          # build + tag only
#   VERSION_TAG=x bash ...        # override the versioned tag
#   PLATFORM=linux/arm64 bash ... # override platform (default: linux/amd64)

set -euo pipefail

REPO="gfatouros/humaine-swarm"
LOCAL_TAG="humaine-swarm:latest"

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean; commit first so the version tag matches the code." >&2
  exit 1
fi
VERSION_TAG="${VERSION_TAG:-$(date +%Y%m%d)-$(git rev-parse --short HEAD)}"
VERSIONED="${REPO}:${VERSION_TAG}"

PLATFORM="${PLATFORM:-linux/amd64}"
BUILD_ARGS=(--platform "$PLATFORM" -t "$LOCAL_TAG" .)

echo "==> docker build ${BUILD_ARGS[*]}"
docker build "${BUILD_ARGS[@]}"

echo "==> Tagging ${VERSIONED}, ${REPO}:latest"
docker tag "$LOCAL_TAG" "$VERSIONED"
docker tag "$LOCAL_TAG" "${REPO}:latest"

echo "==> Image platform: $(docker image inspect "$LOCAL_TAG" --format '{{.Os}}/{{.Architecture}}')"
echo "==> Image IDs (must match):"
docker images --format '  {{.Repository}}:{{.Tag}} {{.ID}}' | grep -E 'humaine-swarm|gfatouros/humaine' || true

if [[ "${SKIP_PUSH:-0}" == "1" ]]; then
  echo "SKIP_PUSH=1 — done (not pushing)"
  exit 0
fi

for tag in "$VERSIONED" "${REPO}:latest"; do
  echo "==> docker push ${tag}"
  docker push "$tag"
done

echo "==> Push complete. Give ops the versioned tag: ${VERSIONED}"
echo ""
echo "Deploy reminder: mount chat history volume and set CHAINLIT_DB in .env:"
echo "  docker volume create humaine-chainlit-data"
echo "  -v humaine-chainlit-data:/data"
echo "  CHAINLIT_DB=sqlite+aiosqlite:////data/chat_history.db"
