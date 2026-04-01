# Getting started

## Prerequisites

- **Access**: The assistant is typically deployed with **Keycloak** login. You need an account and a browser.
- **Run location**: Either use the deployed instance (URL provided by your admin) or run locally/Docker; see the root [README.md](../../README.md) for setup.

## First use

1. **Log in** via the Keycloak prompt (if the app is configured for OAuth).
2. **Open a chat**. The assistant loads your session: it stores your OAuth token and, when possible, fetches MinIO credentials and your Kubeflow namespace from the token. If token or MinIO setup fails (e.g. expired token), you may see a “session expired” message — refresh the page and log in again.
3. **Starter prompts**: The UI can show suggested first messages (e.g. “Which are HumAIne’s AI paradigms?”, “What kind of data have I available?”, “Check out our ML Pipelines”). You can click one or type your own.

## Where to run

- **Local**: See root README — `env.sh` (from `env.sh.example`), `poetry install`, `chainlit run app.py -w`.
- **Docker**: See root README — build or pull image, `.env` from `.env-example`, `docker run` with `--env-file .env`.

Deployment details and required environment variables are in the root [README.md](../../README.md); this doc does not duplicate them.

## What to do first

- **“What buckets do I have access to?”** or **“List my buckets”** — See your MinIO buckets (needed for any MinIO artifact/metrics questions).
- **“Show me available pipelines”** or **“List Kubeflow pipelines”** — See pipelines you can run or inspect.
- **“Tell me about the HumAIne project”** or **“What are HumAIne’s goals?”** — Query the project documentation via RAG.

From there you can run pipelines, list runs, ask for metrics or visualizations, or compare runs; see [Capabilities](03-capabilities.md) and [Workflows](05-workflows.md).
