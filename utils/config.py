from agents.definition import functions
import os

# Kubeflow connection settings (base URL only - user credentials via OAuth)
KUBEFLOW_HOST = os.environ.get("KUBEFLOW_HOST", "http://huanew-kubeflow.ddns.net/pipeline")

# Legacy: Kept for backward compatibility during development
# These should not be used in production - use OAuth instead
# KUBEFLOW_USERNAME and KUBEFLOW_PASSWORD from env vars (optional fallback)
# KUBEFLOW_NAMESPACE per-user from OAuth token

# MinIO connection settings
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "s3-minio.humaine-horizon.eu")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "true").lower() == "true"
MINIO_API_ENDPOINT = os.environ.get("MINIO_API_ENDPOINT", None)

# Pinecone settings
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = "humaine"

# LLM settings
LLM_MODEL = "gpt-4.1-mini"
LLM_MAX_TOKENS = 2000
LLM_TEMPERATURE = 0
EMBEDDING_MODEL = "text-embedding-3-small"

settings = {
    "model": LLM_MODEL,
    "temperature": LLM_TEMPERATURE,
    "max_tokens": LLM_MAX_TOKENS,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0.6,
    "parallel_tool_calls": True,
    "tools": functions,
    "tool_choice": "auto",    
    "stream": True
}

MAX_CONTEXT_LENGTH = 60000  
MAX_INPUT_TOKENS = MAX_CONTEXT_LENGTH - settings["max_tokens"]