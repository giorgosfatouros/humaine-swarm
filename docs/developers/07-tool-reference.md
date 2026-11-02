# Tool reference

Quick lookup for developers. Implementation: [agents/code.py](../../agents/code.py). Schemas: [agents/definition.py](../../agents/definition.py).

| Tool name | Purpose | Required params | Implementation | Dependency |
|-----------|---------|------------------|-----------------|------------|
| get_docs | RAG over project/Kubeflow docs | query | get_docs | Pinecone index (LlamaIndex) |
| get_kf_pipelines | List Kubeflow pipelines | — | get_kf_pipelines | Kubeflow client (Keycloak bearer) |
| get_minio_info | List objects in a bucket | bucket_name | get_minio_info | MinIO client (user) |
| get_pipeline_details | Pipeline definition by ID | pipeline_id | get_pipeline_details | Kubeflow client |
| get_pipeline_version_details | Pipeline version (spec, components, params) | pipeline_id, pipeline_version_id | get_pipeline_version_details | Kubeflow client |
| run_pipeline | Execute a pipeline run (latest version when version_id is omitted); fills declared MinIO credential params from the session | experiment_id, job_name | run_pipeline | Kubeflow client; MinIO STS creds (session) |
| list_runs | List pipeline runs | — | list_runs | Kubeflow client |
| get_run_details | Run status and details, plus `failed_tasks` (each failed step's error message and pod); MinIO credential params are redacted | run_id | get_run_details | Kubeflow client |
| get_pipeline_artifacts_from_MinIO | List or fetch pipeline artifacts in MinIO | bucket_name | get_pipeline_artifacts | MinIO client (user) |
| get_model_metrics | Model metrics (accuracy, etc.) from MinIO; run_name may be a run ID or display name | bucket_name | get_model_metrics | MinIO client (user); Kubeflow client for display-name lookup |
| get_pipeline_visualization | HTML viz (confusion matrix, ROC, feature importance), drawn as a chart in the chat when the page is a recognised table; run_name may be a run ID or display name | bucket_name | get_pipeline_visualization | MinIO client (user); Kubeflow client for display-name lookup |
| compare_pipeline_runs | Compare metrics across runs; run names may be run IDs or display names | bucket_name, pipeline_name, run_names | compare_pipeline_runs | MinIO client (user); Kubeflow client for display-name lookup |
| list_user_buckets | List MinIO buckets available to user | — | list_user_buckets | MinIO client (user) |
| list_experiments | List Kubeflow experiments | — | list_experiments | Kubeflow client |
| get_experiment_details | Experiment by ID or name | experiment_id or experiment_name | get_experiment_details | Kubeflow client |
| list_kubeflow_namespaces | User's Kubeflow profile namespaces, roles and default | — | list_kubeflow_namespaces | Dashboard env-info (session) |
| create_experiment | Create a Kubeflow experiment | name | create_experiment | Kubeflow client |
| get_pipeline_id | Resolve pipeline ID by name | name | get_pipeline_id | Kubeflow client |
| create_pipeline_draft | Validate and compile LLM-written KFP v2 Python, or load an attached .py/.yaml, into a session draft (no upload) | source_code or attachment_name | create_pipeline_draft | utils/pipeline_builder.py (isolated compile subprocess) |
| upload_pipeline_draft | Register a draft in the user's private namespace (new pipeline or new version) after a Register/Cancel prompt | draft_id | upload_pipeline_draft | Kubeflow client; `cl.AskActionMessage` |
| parse_pdf_from_minio | Download, extract text, optional summarization | bucket_name, object_path | parse_pdf_from_minio | MinIO client (user); LangChain for summarization |
| plot_data | Build Plotly chart from data | data | plot_data | None (Plotly); app.py renders result |
| query_haic_benchmark | Fetch pilot's HAIC evaluation results | action | query_haic_benchmark | HAIC API (`HAIC_BASE_URL`); pilot map in utils/haic_pilot_map.py |

**run_pipeline MinIO credentials**: if the pipeline (by `pipeline_id`, latest version when `version_id` is omitted) declares a `minio_session_token` input, any of `minio_access_key` / `minio_secret_key` / `minio_session_token` that the caller left missing or empty are filled from the user's MinIO STS session (`UserSessionManager.get_or_refresh_minio_credentials()`). Caller values win. No session credentials → `{"error": ...}` and no run. The returned `params` show `"<injected from session>"` instead of the secrets. The temporary STS keys are visible in the KFP run's parameters and expire with the session. See [03-agents-and-tools.md](03-agents-and-tools.md#kubeflow).

**Dependency key**: “MinIO client (user)” = `get_user_minio_client()` → UserSessionManager MinIO credentials. “Kubeflow client” = `get_user_kubeflow_client()` (Keycloak bearer token from the session). “Pinecone index” = LlamaIndex VectorStoreIndex over Pinecone (PINECONE_INDEX). “HAIC API” = public read API at `HAIC_BASE_URL`; access scoped per user via static pilot map.
