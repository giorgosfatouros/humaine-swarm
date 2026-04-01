# Tool reference

Quick lookup for developers. Implementation: [agents/code.py](../../agents/code.py). Schemas: [agents/definition.py](../../agents/definition.py).

| Tool name | Purpose | Required params | Implementation | Dependency |
|-----------|---------|------------------|-----------------|------------|
| get_docs | RAG over project/Kubeflow docs | query | get_docs | Pinecone index (LlamaIndex) |
| get_kf_pipelines | List Kubeflow pipelines | — | get_kf_pipelines | Kubeflow client (Dex) |
| get_minio_info | List objects in a bucket | bucket_name | get_minio_info | MinIO client (user) |
| get_pipeline_details | Pipeline definition by ID | pipeline_id | get_pipeline_details | Kubeflow client |
| get_pipeline_version_details | Pipeline version (spec, components, params) | pipeline_id, pipeline_version_id | get_pipeline_version_details | Kubeflow client |
| run_pipeline | Execute a pipeline run | experiment_id, job_name | run_pipeline | Kubeflow client |
| list_runs | List pipeline runs | — | list_runs | Kubeflow client |
| get_run_details | Run status and details | run_id | get_run_details | Kubeflow client |
| get_pipeline_artifacts_from_MinIO | List or fetch pipeline artifacts in MinIO | bucket_name | get_pipeline_artifacts | MinIO client (user) |
| get_model_metrics | Model metrics (accuracy, etc.) from MinIO | bucket_name | get_model_metrics | MinIO client (user) |
| get_pipeline_visualization | HTML viz (confusion matrix, ROC, feature importance) | bucket_name | get_pipeline_visualization | MinIO client (user) |
| compare_pipeline_runs | Compare metrics across runs | bucket_name, pipeline_name, run_names | compare_pipeline_runs | MinIO client (user) |
| list_user_buckets | List MinIO buckets available to user | — | list_user_buckets | MinIO client (user) |
| list_experiments | List Kubeflow experiments | — | list_experiments | Kubeflow client |
| get_experiment_details | Experiment by ID or name | experiment_id or experiment_name | get_experiment_details | Kubeflow client |
| get_user_kubeflow_namespace | User's Kubeflow namespace | — | get_user_namespace | Kubeflow client |
| create_experiment | Create a Kubeflow experiment | name | create_experiment | Kubeflow client |
| get_pipeline_id | Resolve pipeline ID by name | name | get_pipeline_id | Kubeflow client |
| parse_pdf_from_minio | Download, extract text, optional summarization | bucket_name, object_path | parse_pdf_from_minio | MinIO client (user); LangChain for summarization |
| plot_data | Build Plotly chart from data | data | plot_data | None (Plotly); app.py renders result |

**Dependency key**: “MinIO client (user)” = `get_user_minio_client()` → UserSessionManager MinIO credentials. “Kubeflow client” = `get_user_kubeflow_client()` (Dex; may prompt for credentials). “Pinecone index” = LlamaIndex VectorStoreIndex over Pinecone (PINECONE_INDEX).
