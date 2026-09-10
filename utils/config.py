import os

# Kubeflow connection settings (base URL only — auth is the Keycloak bearer token)
KUBEFLOW_HOST = os.environ.get("KUBEFLOW_HOST", "http://huanew-kubeflow.ddns.net/pipeline")
# Empty string (default) = shared catalog via list_pipelines(namespace=None).
# Set explicitly (e.g. "kubeflow") only if shared pipelines live in a named namespace.
KUBEFLOW_SHARED_NAMESPACE = os.environ.get("KUBEFLOW_SHARED_NAMESPACE", "")

# HAIC Benchmark Suite settings
HAIC_BASE_URL = os.environ.get(
    "HAIC_BASE_URL", "https://benchmark.humaine-horizon.eu/api"
)

# Smart Manufacturing semantic-inference service (JSI)
SEMANTIC_INFERENCE_BASE_URL = os.environ.get(
    "SEMANTIC_INFERENCE_BASE_URL", "http://atena.ijs.si:5008"
)

# MinIO connection settings
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "s3-minio.humaine-horizon.eu")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "true").lower() == "true"
MINIO_API_ENDPOINT = os.environ.get("MINIO_API_ENDPOINT", None)

# Pinecone settings
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = "humaine"

# LLM settings
LLM_MODEL = "gpt-5.6-luna"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "medium")

MAX_CONTEXT_LENGTH = 60000
# Headroom reserved for the model's response, including reasoning tokens.
OUTPUT_TOKEN_RESERVE = 8000
MAX_INPUT_TOKENS = 2000

settings = {
    "model": LLM_MODEL,
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "stream": True,
    "reasoning": {"effort": LLM_REASONING_EFFORT},
    "max_output_tokens": OUTPUT_TOKEN_RESERVE,
}