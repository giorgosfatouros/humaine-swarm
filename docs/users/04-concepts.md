# Concepts

## run_id vs run_name

- **run_id**: Assigned by **Kubeflow** when a pipeline run is created. It is a unique ID (e.g. UUID-like). Use it when you ask for “run details”, “status of run [id]”, or “information about run [id]” in the Kubeflow sense.
- **run_name**: Used in **MinIO** object paths to group artifacts for a given run (e.g. `kubeflow/<pipeline_name>/<run_name>/metrics/...`). It is often a string that includes the pipeline name and a run identifier. Use it when you ask for “metrics for run X”, “artifacts for run X”, “visualizations for run X”, or “compare runs X and Y”.

**Which tools use which?**

- **run_id**: Listing runs, getting run details (Kubeflow run status, timing, parameters).
- **run_name**: Listing or fetching artifacts, getting metrics, getting visualizations, comparing runs (all MinIO-based).

If you only have a run_id, you may need to list runs or artifacts to find the corresponding run_name used in MinIO paths; the assistant can guide you.

## MinIO buckets

- Your access to **buckets** is determined by your **user policies** (from Keycloak/OAuth). You only see and use buckets you are allowed to access.
- **Always discover buckets first**: ask “What buckets do I have access to?” or “List my buckets.” Then use that bucket name in any follow-up (e.g. “metrics for pipeline X in bucket Y”).
- All MinIO-related answers (artifacts, metrics, visualizations, compare) require a **bucket_name**; the assistant will ask for it if you don’t specify one and it can’t infer it.

## Kubeflow namespace

- **Namespace** is the Kubernetes namespace used for your Kubeflow resources (experiments, runs). It scopes what you see: you typically see only experiments and runs in your namespace.
- It is **derived from your OAuth token** (e.g. from groups or roles like `kubeflow-<username>`, or from your preferred_username/email). You don’t set it manually in the chat.
- You can ask “What’s my Kubeflow namespace?” to see what the assistant is using.

## Sessions

- The app keeps **per-user session state**: message history, OAuth token, MinIO credentials (from Keycloak STS), Kubeflow namespace, and optionally Kubeflow username/password if you entered them when prompted.
- **Session expiry**: Chainlit and the app define session timeouts; see the deployment configuration. If you are idle too long, you may need to start a new chat or refresh.
- **Token expiry**: Your Keycloak access token expires. When it does, MinIO credential refresh can fail. The app then **clears your session** and shows a message like “Your session has expired. Please refresh the page to log in again.” **Refresh the page and log in again** to continue; no data is retained after session clear.
