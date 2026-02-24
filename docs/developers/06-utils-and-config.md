# Utils and config

## config.py

[utils/config.py](../../utils/config.py) imports **`functions`** from [agents/definition.py](../../agents/definition.py) and builds env-derived settings.

### Environment-derived constants

| Name | Env var | Default / note |
|------|---------|----------------|
| KUBEFLOW_HOST | KUBEFLOW_HOST | `http://huanew-kubeflow.ddns.net/pipeline` |
| MINIO_ENDPOINT | MINIO_ENDPOINT | `s3-minio.humaine-horizon.eu` |
| MINIO_SECURE | MINIO_SECURE | `"true"` → True |
| MINIO_API_ENDPOINT | MINIO_API_ENDPOINT | None |
| PINECONE_API_KEY | PINECONE_API_KEY | Required (no default) |
| PINECONE_INDEX | — | `"humaine"` (hardcoded) |
| LLM_MODEL | — | `"gpt-4.1-mini"` |
| LLM_MAX_TOKENS | — | 2000 |
| LLM_TEMPERATURE | — | 0 |
| EMBEDDING_MODEL | — | `"text-embedding-3-small"` |

### settings dict

Passed as **kwargs** to **OpenAI `chat.completions.create()`**:

- **model**, **temperature**, **max_tokens**, **top_p**, **frequency_penalty**, **presence_penalty**
- **tools**: `functions` (from agents/definition.py)
- **tool_choice**: `"auto"`
- **parallel_tool_calls**: True
- **stream**: True

### Context length

- **MAX_CONTEXT_LENGTH**: 60000  
- **MAX_INPUT_TOKENS**: MAX_CONTEXT_LENGTH - settings["max_tokens"]  
Used in [classes/user_handler.py](../../classes/user_handler.py) for message-history truncation.

---

## helper_functions.py

[utils/helper_functions.py](../../utils/helper_functions.py) provides shared utilities.

### Logging

- **setup_logging(logger_name, level)**: Creates logger with console handler; level can be overridden by env **LOG_LEVEL** (DEBUG, INFO, WARNING, ERROR, CRITICAL).

### JWT / token (optional use)

- **decode_jwt(token)**: Decodes with CHAINLIT_AUTH_SECRET; used for custom auth flows if needed.
- **extract_token_from_headers(headers)**: Tries Authorization Bearer then referer query param.
- **extract_user_from_payload(payload)**: Builds Chainlit User from payload (e.g. userId, email).

### Tool helpers

- **get_required_arguments(function_name)**: Returns required parameter names for a tool in `functions`.
- **rag_extract_deliverables(retrieved_nodes)**: Maps LlamaIndex nodes to list of `{text: node.get_content()}` for RAG responses.
- **manage_chat_history(message_history)**: Character-based truncation (sliding window 100k); **not used** by the app (app uses UserSessionManager token-based truncation).

### MinIO

- **fetch_minio_credentials_from_keycloak(access_token)**: STS AssumeRoleWithWebIdentity; returns access_key, secret_key, session_token, expiry. See [Session and auth](04-session-and-auth.md).
- **get_minio_client(user_credentials=None)**: Returns **Minio** client; endpoint from MINIO_API_ENDPOINT or MINIO_ENDPOINT/MINIO_SECURE; credentials from argument or env (MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SESSION_TOKEN).

### Kubeflow

- **extract_user_namespace_from_token(access_token)**: Decodes JWT (no verify), extracts namespace from claims/groups/roles/preferred_username; default "kubeflow".
- **KFPClientManager**: Holds api_url, skip_tls_verify, Dex username/password, auth_type ("local" or "ldap"). **`_get_session_cookies()`**: Performs Dex login flow (GET redirects, POST login, optional approval); returns cookie string. **`create_kfp_client(namespace=None)`**: Gets cookies, (optionally) patches kfp.Client for SSL, returns **kfp.Client(host, cookies, namespace)**.
- **get_kubeflow_client(user_namespace, user_token, user_username, user_password)**: Instantiates KFPClientManager with KUBEFLOW_HOST and Dex credentials (user_username, user_password from params; empty string if not provided), then **create_kfp_client(namespace=user_namespace)**.
- **get_kubeflow_old_client**: Legacy helper; not used by the main app path.

### Artifact helpers (KFP DSL)

Used when authoring Kubeflow pipeline components (e.g. in [kubeflow/](../../kubeflow/)), not by the chat app:

- **create_dataset_artifact**, **create_model_artifact**, **create_metrics_artifact**, **create_classification_metrics_artifact**
- **read_artifact_metadata**, **log_artifact_properties**
- **read_dataset_from_artifact**, **save_model_to_artifact**, **save_metrics_to_artifact**

Reference only; no effect on the assistant runtime.
