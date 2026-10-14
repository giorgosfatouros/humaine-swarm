#!/usr/bin/env bash
# Build humaine-swarm, tag for Docker Hub, optionally push.
# Every release gets an immutable versioned tag (amd64-YYYYMMDD-<git sha>);
# :amd64 and :latest are moving aliases to the newest release. Before :amd64
# moves, the image it points to is archived under its own versioned tag on Hub,
# so previous versions are never lost.
# Usage (from repo root):
#   bash .cursor/skills/humaine-swarm-docker-publish/scripts/publish-amd64.sh
#   SKIP_PUSH=1 bash ...          # build + tag only
#   VERSION_TAG=amd64-x bash ...  # override the versioned tag
#   PLATFORM=linux/arm64 bash ... # override platform (default: auto-detect)

set -euo pipefail

REPO="gfatouros/humaine-swarm"
LOCAL_TAG="humaine-swarm:latest"
HUB_TAG="${REPO}:amd64"

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean; commit first so the version tag matches the code." >&2
  exit 1
fi
VERSION_TAG="${VERSION_TAG:-amd64-$(date +%Y%m%d)-$(git rev-parse --short HEAD)}"
VERSIONED="${REPO}:${VERSION_TAG}"

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

echo "==> Tagging ${VERSIONED}, ${HUB_TAG}, ${REPO}:latest"
docker tag "$LOCAL_TAG" "$VERSIONED"
docker tag "$LOCAL_TAG" "$HUB_TAG"
docker tag "$LOCAL_TAG" "${REPO}:latest"

echo "==> Image IDs (must match):"
docker images --format '  {{.Repository}}:{{.Tag}} {{.ID}}' | grep -E 'humaine-swarm|gfatouros/humaine' || true

if [[ "${SKIP_PUSH:-0}" == "1" ]]; then
  echo "SKIP_PUSH=1 — done (not pushing)"
  exit 0
fi

# Archive the current :amd64 under a versioned tag (server-side copy, no pull).
if CURRENT_CREATED="$(docker buildx imagetools inspect "$HUB_TAG" --format '{{json .Image}}' 2>/dev/null \
    | python3 -c 'import json,sys;d=json.load(sys.stdin);d=d.get("linux/amd64",d);print(d["created"][:10].replace("-",""))' 2>/dev/null)"; then
  CURRENT_DIGEST="$(docker buildx imagetools inspect "$HUB_TAG" --format '{{.Manifest.Digest}}')"
  ARCHIVE_TAG="${REPO}:amd64-${CURRENT_CREATED}-${CURRENT_DIGEST:7:7}"
  echo "==> Archiving current ${HUB_TAG} (${CURRENT_DIGEST}) as ${ARCHIVE_TAG}"
  docker buildx imagetools create -t "$ARCHIVE_TAG" "${REPO}@${CURRENT_DIGEST}"
fi

for tag in "$VERSIONED" "$HUB_TAG" "${REPO}:latest"; do
  echo "==> docker push ${tag}"
  docker push "$tag"
done

echo "==> Push complete. Give ops the versioned tag: ${VERSIONED}"
echo ""
echo "Deploy reminder: mount chat history volume and set CHAINLIT_DB in .env:"
echo "  docker volume create humaine-chainlit-data"
echo "  -v humaine-chainlit-data:/data"
echo "  CHAINLIT_DB=sqlite+aiosqlite:////data/chat_history.db"
