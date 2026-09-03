from agents.definition import functions
import os
from utils.responses_adapter import convert_tools_for_responses

# Kubeflow connection settings (base URL only - user credentials via OAuth)
KUBEFLOW_HOST = os.environ.get("KUBEFLOW_HOST", "http://huanew-kubeflow.ddns.net/pipeline")

# Legacy: Kept for backward compatibility during development
# These should not be used in production - use OAuth instead
# KUBEFLOW_USERNAME and KUBEFLOW_PASSWORD from env vars (optional fallback)
# KUBEFLOW_NAMESPACE per-user from OAuth token

# HAIC Benchmark Suite settings
HAIC_BASE_URL = os.environ.get(
    "HAIC_BASE_URL", "https://benchmark.humaine-horizon.eu/api"
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
    "tools": convert_tools_for_responses(functions),
    "tool_choice": "auto",
    "stream": True,
    "reasoning": {"effort": LLM_REASONING_EFFORT},
    "max_output_tokens": OUTPUT_TOKEN_RESERVE,
}