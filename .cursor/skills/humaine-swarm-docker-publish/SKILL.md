---
name: humaine-swarm-docker-publish
description: Build, tag, push, and deploy the humaine-swarm Chainlit Docker image to Docker Hub (gfatouros/humaine-swarm) and run it locally or on Linux servers. Use when updating the Docker image, publishing to Docker Hub, deploying amd64, docker build/push/pull, or asking Roberto/ops to refresh the running container.
---

# HumAIne Swarm — Docker publish & deploy

## Constants

| Item | Value |
|------|--------|
| Hub repo | `gfatouros/humaine-swarm` |
| Production tag | `amd64` (x86_64 servers) |
| Optional tag | `latest` |
| Local build tag | `humaine-swarm:latest` |
| Container name | `humaine-swarm-assistant` (adjust if different on server) |
| Port | `8000` |
| Env file | `.env` at project root (never bake into image) |

**Not in the image:** `.docs/`, `.md_files/` (gitignored). Pinecone RAG is updated separately via `python data_injection.py` — deploying a new image does not re-index docs.

---

## Golden rules

1. **Build once, tag many** — never run separate builds for `:latest` and `:amd64`; duplicate builds create different image IDs with the same displayed size.
2. **Verify IMAGE ID before push** — `humaine-swarm:latest` and `gfatouros/humaine-swarm:amd64` must share the same ID.
3. **Push only after retag** — pushing an old `gfatouros/humaine-swarm:amd64` tag uploads nothing new (all layers "Already exists").
4. **Wait for success** — push must end with `amd64: digest: sha256:...` and no `failed to copy`. Retry on proxy/network errors.

---

## Workflow checklist

```
- [ ] Code changes committed / tested locally (optional: chainlit run app.py)
- [ ] Docker logged in to Hub
- [ ] Build (platform per host — see below)
- [ ] Tag gfatouros/humaine-swarm:amd64 (+ optional :latest)
- [ ] Verify matching IMAGE IDs
- [ ] docker push (retry if network fails mid-upload)
- [ ] Note new digest for ops email
- [ ] Server: pull + recreate container
- [ ] Smoke test http://host:8000/
```

---

## Step 1 — Docker Hub login

Web login often fails on Docker Desktop. Use a PAT:

```bash
echo 'DOCKER_HUB_PAT' | docker login --username gfatouros --password-stdin
```

Create PAT: Docker Hub → Account Settings → Security.

If `docker pull python:3.12-slim` hangs for minutes, fix Docker Desktop **Settings → Proxies** (disable broken manual proxy), restart Docker, retry.

---

## Step 2 — Build

Run from **project root** (`humaine-swarm/`).

### macOS (Apple Silicon) → deploy to **linux/amd64** server

Emulation is required; first build is slow (~10–30 min for `poetry install`).

```bash
docker build --platform linux/amd64 -t humaine-swarm:latest .
```

### Linux x86_64 (native amd64)

No platform flag:

```bash
docker build -t humaine-swarm:latest .
```

### Linux ARM64 server (uncommon for this project)

```bash
docker build -t humaine-swarm:latest .
# Tag and push as arm64 or a distinct tag — coordinate with ops; production Hub tag today is :amd64
```

---

## Step 3 — Tag for Hub

```bash
docker tag humaine-swarm:latest gfatouros/humaine-swarm:amd64
docker tag humaine-swarm:latest gfatouros/humaine-swarm:latest   # optional
docker tag humaine-swarm:latest humaine-swarm:amd64             # optional local alias
```

**Verify:**

```bash
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep humaine-swarm
```

All production tags must show the **same** IMAGE ID.

---

## Step 4 — Push

```bash
docker push gfatouros/humaine-swarm:amd64
docker push gfatouros/humaine-swarm:latest   # optional
```

Expect layer uploads on a real update. If every layer is `Already exists` and digest unchanged, the tag was not repointed to the new build — go back to Step 3.

**Retry after failure** (common: `use of closed network connection` via Docker proxy):

```bash
docker push gfatouros/humaine-swarm:amd64
```

Partial uploads resume; already-pushed layers skip quickly.

---

## Step 5 — Local test (optional)

```bash
docker run -d \
  --name humaine-swarm-assistant \
  -p 8000:8000 \
  --env-file .env \
  --rm \
  humaine-swarm:latest
```

Open `http://localhost:8000`. Stop with `docker stop humaine-swarm-assistant`.

On Mac, confirm arch if debugging platform issues:

```bash
docker image inspect humaine-swarm:latest --format '{{.Os}}/{{.Architecture}}'
# Production build target: linux/amd64
```

---

## Step 6 — Linux server deploy

SSH to host, then:

```bash
docker pull gfatouros/humaine-swarm:amd64
docker stop humaine-swarm-assistant || true
docker rm humaine-swarm-assistant || true
docker run -d \
  --name humaine-swarm-assistant \
  -p 8000:8000 \
  --env-file /path/to/.env \
  --restart unless-stopped \
  gfatouros/humaine-swarm:amd64
```

Adjust container name, port bind, and env path to match the existing deployment.

**Smoke test:** Chainlit UI loads; auth works; sample chat + `get_docs` if Pinecone keys are set in `.env`.

---

## Ops email template

When asking infra (e.g. Roberto) to update:

- Image: `gfatouros/humaine-swarm:amd64`
- New digest: `sha256:...` (from push output or Docker Hub)
- Steps: `pull` → stop/remove old container → `run` with same `--env-file`
- Note: RAG index unchanged unless `data_injection.py` was run separately

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| Build stuck on `load metadata for python:3.12-slim` | Docker daemon ↔ Hub | Fix proxy; restart Docker; `docker pull python:3.12-slim` |
| `docker login` hangs at `Username:` | Waiting for input after failed browser login | Use PAT + `--password-stdin` |
| Push all `Already exists`, old digest | Pushed before retag | Retag from `humaine-swarm:latest`, push again |
| Two local tags, same 755MB, different IDs | Two separate builds | Remove stale tag; retag from one build |
| Push fails mid-upload | Proxy/network drop | Retry `docker push` |
| App runs but RAG docs missing | Pinecone not re-indexed | Run `python data_injection.py --sources md` (or `pdf,md`) locally (not in container) |

---

## Helper script

For the standard amd64 publish path from Mac/Linux project root:

```bash
bash .cursor/skills/humaine-swarm-docker-publish/scripts/publish-amd64.sh
```

See script for `SKIP_PUSH=1` and `PLATFORM` overrides.
