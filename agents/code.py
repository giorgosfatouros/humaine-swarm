import os
from datetime import datetime
import json
import asyncio
import chainlit as cl
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from pinecone import Pinecone
from llama_index.core import Settings
from llama_index.llms.openai import AsyncOpenAI, OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from utils.helper_functions import setup_logging, rag_extract_deliverables, get_minio_client
from utils.config import *  # Import configuration settings
import logging
import kfp
from typing import Dict, List, Optional, Any
from minio import Minio
from minio.error import S3Error
from utils.helper_functions import get_kubeflow_client
from classes.user_handler import UserSessionManager
from io import BytesIO
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
import plotly.graph_objects as go
import pandas as pd

# Set up the LLM
Settings.llm  = OpenAI(model=LLM_MODEL, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE)
Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
logger = setup_logging('CODE', level=logging.INFO)
client = AsyncOpenAI()


# Set up the vector store
pc = Pinecone(api_key=PINECONE_API_KEY)
vector_store = PineconeVectorStore(pinecone_index=pc.Index(PINECONE_INDEX))
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


# Helper functions to get user-specific clients
async def get_user_minio_client() -> Minio:
    """
    Get a MinIO client configured with user-specific credentials from session.
    Falls back to environment variables if user credentials not available.
    """
    try:
        # Try to get and refresh credentials if needed
        credentials = await UserSessionManager.get_or_refresh_minio_credentials()
        if credentials:
            logger.info("Using user-specific MinIO credentials")
            return get_minio_client(user_credentials=credentials)
        else:
            logger.warning("User MinIO credentials not available, falling back to environment")
            return get_minio_client()
    except Exception as e:
        logger.error(f"Error getting user MinIO client: {str(e)}")
        # Fall back to environment credentials
        return get_minio_client()


async def prompt_for_kubeflow_credentials() -> Optional[Dict[str, str]]:
    """
    Prompt the user for Kubeflow credentials using Chainlit's AskUserMessage.
    
    Returns:
        Dictionary with 'username', 'password', and optionally 'namespace' keys,
        or None if user cancels or times out
    """
    try:
        # Prompt for username
        username_response = await cl.AskUserMessage(
            content="Please provide your Kubeflow username to connect to Kubeflow pipelines.",
            timeout=120,
            raise_on_timeout=False
        ).send()
        
        if not username_response or not username_response.get('output'):
            logger.warning("User cancelled or timed out while providing Kubeflow username")
            await cl.Message(
                content="Kubeflow connection cancelled. Please provide credentials when prompted to use Kubeflow features.",
                author="System"
            ).send()
            return None
        
        username = username_response['output'].strip()
        if not username:
            logger.warning("User provided empty username")
            await cl.Message(
                content="Username cannot be empty. Please try again.",
                author="System"
            ).send()
            return None
        
        # Prompt for password
        password_response = await cl.AskUserMessage(
            content="Please provide your Kubeflow password.",
            timeout=120,
            raise_on_timeout=False
        ).send()
        
        if not password_response or not password_response.get('output'):
            logger.warning("User cancelled or timed out while providing Kubeflow password")
            await cl.Message(
                content="Kubeflow connection cancelled. Please provide credentials when prompted to use Kubeflow features.",
                author="System"
            ).send()
            return None
        
        password = password_response['output'].strip()
        if not password:
            logger.warning("User provided empty password")
            await cl.Message(
                content="Password cannot be empty. Please try again.",
                author="System"
            ).send()
            return None
        
        # Optionally prompt for namespace
        namespace_response = await cl.AskUserMessage(
            content="Please provide your Kubeflow namespace (optional - press Enter to skip and use default).",
            timeout=120,
            raise_on_timeout=False
        ).send()
        
        namespace = None
        if namespace_response and namespace_response.get('output'):
            namespace = namespace_response['output'].strip()
            if not namespace:
                namespace = None
        
        credentials = {
            "username": username,
            "password": password,
            "namespace": namespace
        }
        
        logger.info("Kubeflow credentials collected from user (username logged, password not logged)")
        return credentials
        
    except Exception as e:
        logger.error(f"Error prompting for Kubeflow credentials: {str(e)}")
        try:
            await cl.Message(
                content=f"An error occurred while collecting credentials: {str(e)}",
                author="System"
            ).send()
        except:
            pass
        return None


async def get_user_kubeflow_client() -> kfp.Client:
    """
    Get a Kubeflow client configured with user-specific credentials from session.
    Checks for credentials in this order:
    1. Session credentials (username/password)
    2. OAuth token + namespace extraction
    If neither is available, prompts the user for credentials.
    
    Raises:
        ValueError: If user cancels credential prompt or credentials are invalid
        RuntimeError: If connection to Kubeflow fails
    """
    max_retries = 2
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # First, check for session credentials
            session_creds = UserSessionManager.get_kubeflow_credentials()
            if session_creds and UserSessionManager.has_kubeflow_credentials():
                username = session_creds.get("username")
                password = session_creds.get("password")
                namespace = session_creds.get("namespace") or UserSessionManager.get_kubeflow_namespace()
                
                logger.info("Using session-stored Kubeflow credentials")
                try:
                    return get_kubeflow_client(
                        user_namespace=namespace,
                        user_username=username,
                        user_password=password
                    )
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    # Check if it's an authentication error
                    if "login" in error_msg or "credential" in error_msg or "invalid" in error_msg:
                        logger.warning(f"Stored credentials appear to be invalid: {str(e)}")
                        # Clear invalid credentials and re-prompt
                        UserSessionManager.clear_kubeflow_credentials()
                        if retry_count < max_retries:
                            retry_count += 1
                            await cl.Message(
                                content="The stored Kubeflow credentials appear to be invalid. Please provide new credentials.",
                                author="System"
                            ).send()
                            continue
                        else:
                            raise ValueError("Invalid Kubeflow credentials provided. Please try again.") from e
                    else:
                        # Other connection errors
                        raise RuntimeError(f"Failed to connect to Kubeflow: {str(e)}") from e
            
            # Second, check for OAuth token + namespace
            namespace = UserSessionManager.get_kubeflow_namespace()
            oauth_token = UserSessionManager.get_oauth_token()
            
            if namespace and oauth_token:
                logger.info(f"OAuth token available with namespace: {namespace}")
                # Note: OAuth token-based auth not fully implemented yet
                # For now, we'll still need credentials
                # This is a placeholder for future SSO implementation
                pass
            
            # If no credentials available, prompt the user
            if retry_count == 0:
                logger.info("No Kubeflow credentials found in session, prompting user")
            else:
                logger.info("Re-prompting for Kubeflow credentials after invalid credentials")
            
            credentials = await prompt_for_kubeflow_credentials()
            
            if not credentials:
                raise ValueError("User cancelled or failed to provide Kubeflow credentials. Cannot connect to Kubeflow.")
            
            # Validate credentials format
            if not credentials.get("username") or not credentials.get("password"):
                raise ValueError("Invalid credentials format: username and password are required")
            
            # Store credentials in session
            UserSessionManager.set_kubeflow_credentials(
                username=credentials["username"],
                password=credentials["password"],
                namespace=credentials.get("namespace")
            )
            
            # Use the stored credentials
            namespace = credentials.get("namespace") or UserSessionManager.get_kubeflow_namespace()
            logger.info("Attempting to connect with newly provided Kubeflow credentials")
            
            try:
                return get_kubeflow_client(
                    user_namespace=namespace,
                    user_username=credentials["username"],
                    user_password=credentials["password"]
                )
            except RuntimeError as e:
                error_msg = str(e).lower()
                # Check if it's an authentication error
                if "login" in error_msg or "credential" in error_msg or "invalid" in error_msg:
                    logger.warning(f"Provided credentials appear to be invalid: {str(e)}")
                    # Clear invalid credentials
                    UserSessionManager.clear_kubeflow_credentials()
                    if retry_count < max_retries:
                        retry_count += 1
                        await cl.Message(
                            content="The provided Kubeflow credentials appear to be invalid. Please try again.",
                            author="System"
                        ).send()
                        continue
                    else:
                        raise ValueError("Invalid Kubeflow credentials provided. Please verify your username and password.") from e
                else:
                    # Other connection errors
                    raise RuntimeError(f"Failed to connect to Kubeflow: {str(e)}") from e
        
        except ValueError as e:
            # User cancellation or validation errors - don't retry
            logger.error(f"Kubeflow credential error: {str(e)}")
            raise
        except RuntimeError as e:
            # Connection errors - don't retry
            logger.error(f"Kubeflow connection error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting user Kubeflow client: {str(e)}")
            if retry_count < max_retries:
                retry_count += 1
                await cl.Message(
                    content=f"An error occurred: {str(e)}. Retrying...",
                    author="System"
                ).send()
                continue
            else:
                raise RuntimeError(f"Failed to get Kubeflow client after {max_retries} retries: {str(e)}") from e
    
    # Should not reach here, but just in case
    raise RuntimeError("Failed to get Kubeflow client after maximum retries")




def read_prompt(type: str):
    current_date = datetime.now().strftime("%Y-%m-%d")

    if type == "system":
        with open('agents/system.md', 'r') as file:
            prompt = file.read()
            update_info = f"\n\n*Note:  Today's date is {current_date}.*\n\nWhen you receive function results, incorporate them into your responses to provide accurate and helpful information to the user."
            prompt += update_info
    return prompt


@cl.step(type="llm", name="Optimizing User Query")
async def optimize_query(search_query: str) -> str:
    prompt = f"""
    Refine the following search query to include related terms and context relevant to EU-funded research projects.

    Original query: 
    {search_query}

    Provide your response in the following JSON format:
    {{
        "optimized_query": "<optimized query>"
    }}
    """

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant expert in crafting prompts for LLMs in EU-funded research projects. Your responses must be a valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0
    )
    # Extract the expanded query from the response
    if not response or not response.choices or not response.choices[0].message:
        logger.error("Invalid response from the LLM")
        return search_query  # Return original query if optimization fails
        
    response_content = response.choices[0].message.content.strip()
    logger.info(f"Response content: {response_content}")

    # Parse the response content to extract the optimized query
    optimized_query = json.loads(response_content).get("optimized_query", "")
    logger.info(f"Optimized query: {optimized_query}")

    return optimized_query


# RAG functions
@cl.step(type="tool", name="Documentation", show_input=False)
async def get_docs(query: str):
    # optimized_query = await optimize_query(query)
    retriever_humaine = VectorIndexRetriever(index=index, similarity_top_k=10)
    retrieved_documents = await retriever_humaine.aretrieve(query)
    logger.info(f"Retrieved documents: {retrieved_documents}")
    return rag_extract_deliverables(retrieved_documents)

# MinIO functions

@cl.step(type="tool", name="Bucket Info", show_input=False)
async def get_minio_info(bucket_name: Optional[str] = None, prefix: Optional[str] = None, max_items: int = 10) -> Dict:
    """
    Retrieve information about objects in a specific MinIO bucket.
    
    Args:
        bucket_name: Name of the MinIO bucket to query
        prefix: Optional prefix to filter objects (like a folder path)
        max_items: Maximum number of items to return (default: 10)
        
    Returns:
        Dict containing object information from the specified bucket
    """
    try:
        client = await get_user_minio_client()
        
        if not bucket_name:
            return {"error": "bucket_name parameter is required. Use list_user_buckets() to discover available buckets."}
        
        # Check if specified bucket exists
        if not client.bucket_exists(bucket_name):
            return {"error": f"Bucket '{bucket_name}' does not exist use the list_user_buckets function to get a list of the user's buckets"}
        
        # List all objects in the bucket with the given prefix
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
        
        # Extract minimal info for each object
        object_list = []
        count = 0
        
        for obj in objects:
            if count >= max_items:
                break
                
            # Only include essential information
            object_info = {
                "name": obj.object_name,
                "size": obj.size,
                "last_modified": str(obj.last_modified)
            }
            object_list.append(object_info)
            count += 1
        
        total_objects = sum(1 for _ in client.list_objects(bucket_name, prefix=prefix, recursive=True))
        
        return {
            "bucket": bucket_name,
            "prefix": prefix,
            "total_object_count": total_objects,
            "displayed_objects": len(object_list),
            "objects": object_list
        }
            
    except S3Error as e:
        logger.error(f"Error retrieving MinIO information: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error retrieving MinIO information: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="List Buckets", show_input=False)
async def list_user_buckets(max_buckets: int = 20) -> Dict:
    """
    List all MinIO buckets available to the user with basic statistics.
    
    Args:
        max_buckets: Maximum number of buckets to return (default: 20)
        
    Returns:
        Dict containing information about available buckets
    """
    try:
        client = await get_user_minio_client()
        
        # Get all buckets
        buckets = client.list_buckets()
        
        # Limit to max_buckets
        buckets = buckets[:max_buckets] if len(buckets) > max_buckets else buckets
        
        bucket_data = []
        for bucket in buckets:
            # Get basic stats for each bucket
            try:
                # Count objects and get total size
                objects = list(client.list_objects(bucket.name, recursive=False, prefix="", include_user_meta=False, include_version=False))
                object_count = len(objects)
                
                # Try to get a sample of object types to suggest bucket purpose
                sample_objects = list(client.list_objects(bucket.name, recursive=True, prefix="", include_user_meta=False, include_version=False))
                object_types = set()
                ml_related = False
                pipeline_related = False
                
                for obj in sample_objects:
                    # Extract extension if any
                    name = obj.object_name
                    if '.' in name:
                        ext = name.split('.')[-1].lower()
                        object_types.add(ext)
                    
                    # Check for ML patterns in paths
                    if any(term in name.lower() for term in ['model', 'metric', 'plot', 'training', 'ml', 'ai']):
                        ml_related = True
                    
                    # Check for pipeline patterns
                    if any(term in name.lower() for term in ['pipeline', 'kubeflow', 'workflow', 'run-']):
                        pipeline_related = True
                
                # Get first-level "directories"
                dirs = set()
                for obj in objects:
                    parts = obj.object_name.split('/')
                    if len(parts) > 1 and parts[0]:
                        dirs.add(parts[0])
                
                # Get kubeflow directories if present
                kubeflow_dirs = []
                if "kubeflow" in dirs:
                    kubeflow_objects = list(client.list_objects(bucket.name, recursive=False, prefix="kubeflow/"))
                    for obj in kubeflow_objects:
                        parts = obj.object_name.split('/')
                        if len(parts) > 1 and parts[1]:
                            kubeflow_dirs.append(parts[1])
                
                bucket_info = {
                    "name": bucket.name,
                    "creation_date": str(bucket.creation_date),
                    "object_count": object_count,
                    "top_level_dirs": list(dirs),
                    "file_types": list(object_types) if object_types else [],
                    "appears_ml_related": ml_related,
                    "appears_pipeline_related": pipeline_related,
                }
                
                # Add kubeflow pipeline names if found
                if kubeflow_dirs:
                    bucket_info["kubeflow_pipelines"] = list(set(kubeflow_dirs))
                
                # Add data_files organized by type for analysis tools
                data_files = {
                    "pickle": [obj.object_name for obj in sample_objects if obj.object_name.lower().endswith(('.pkl', '.pickle'))],
                    "json": [obj.object_name for obj in sample_objects if obj.object_name.lower().endswith('.json')],
                    "pdf": [obj.object_name for obj in sample_objects if obj.object_name.lower().endswith('.pdf')]
                }
                # Only include non-empty types, limit to 10 per type for readability
                filtered_data_files = {}
                for file_type, files in data_files.items():
                    if files:
                        filtered_data_files[file_type] = files[:10]
                        if len(files) > 10:
                            filtered_data_files[f"{file_type}_truncated"] = True
                            filtered_data_files[f"total_{file_type}_files"] = len(files)
                if filtered_data_files:
                    bucket_info["data_files"] = filtered_data_files
                
                bucket_data.append(bucket_info)
                
            except Exception as bucket_error:
                # If we can't get stats, just include basic info
                bucket_data.append({
                    "name": bucket.name,
                    "creation_date": str(bucket.creation_date),
                    "error": str(bucket_error)
                })
        
        return {
            "total_buckets": len(buckets),
            "buckets": bucket_data
        }
            
    except Exception as e:
        logger.error(f"Error listing buckets: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="Pipeline Artifacts", show_input=False)
async def get_pipeline_artifacts(
    bucket_name: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    run_name: Optional[str] = None,
    artifact_type: Optional[str] = None,
    max_items: int = 20,
    object_path: Optional[str] = None
) -> Dict:
    """
    Retrieve ML pipeline artifacts from MinIO storage.
    
    Args:
        bucket_name: Name of the MinIO bucket to query
        pipeline_name: Name of the pipeline to query artifacts for (e.g. 'diabetes-svm-classification')
        run_name: Optional specific run name to filter artifacts
        artifact_type: Optional artifact type to filter (e.g. 'models', 'metrics', 'plots')
        max_items: Maximum number of items to return
        object_path: Optional direct MinIO object path to retrieve (overrides other path parameters)
        
    Returns:
        Dict containing artifact information
    """
    try:
        client = await get_user_minio_client()
        
        if not bucket_name:
            return {"error": "bucket_name parameter is required. Use list_user_buckets() to discover available buckets."}
        
        # Check if bucket exists
        if not client.bucket_exists(bucket_name):
            return {"error": f"Bucket '{bucket_name}' does not exist use the list_user_buckets function to get a list of the user's buckets"}
        
        # If object_path is provided, use it directly
        if object_path:
            # Check if it's a single object or a prefix
            try:
                # Try to get the object (assuming it's a specific file)
                stat = client.stat_object(bucket_name, object_path)
                
                # It's a specific file, retrieve it
                response = client.get_object(bucket_name, object_path)
                file_content = response.read()
                
                # Determine file type and process accordingly
                if object_path.endswith('.json'):
                    try:
                        content = json.loads(file_content.decode('utf-8'))
                    except:
                        content = {"raw_content": file_content.decode('utf-8', errors='replace')}
                elif object_path.endswith(('.txt', '.md', '.py', '.yaml', '.yml')):
                    content = {"text_content": file_content.decode('utf-8', errors='replace')}
                else:
                    # For binary files, just note the size
                    content = {"binary_size": len(file_content), "format": object_path.split('.')[-1] if '.' in object_path else "unknown"}
                
                response.close()
                response.release_conn()
                
                parts = object_path.split('/')
                return {
                    "bucket": bucket_name,
                    "path": object_path,
                    "filename": parts[-1] if parts else object_path,
                    "size": stat.size,
                    "last_modified": str(stat.last_modified),
                    "content": content
                }
                
            except Exception as e:
                if "NoSuchKey" not in str(e):
                    # It's a different error than object not found
                    logger.error(f"Error accessing object {object_path}: {str(e)}")
                
                # Assume it's a prefix
                prefix = object_path
                if not prefix.endswith('/'):
                    prefix += '/'
        else:
            # Construct prefix based on provided parameters
            prefix = "kubeflow/"
            if pipeline_name:
                prefix += f"{pipeline_name}/"
                if run_name:
                    prefix += f"{run_name}/"
                    if artifact_type:
                        prefix += f"{artifact_type}/"
        
        # List objects with the constructed prefix
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
        
        # Process and categorize the artifacts
        artifacts = []
        count = 0
        
        for obj in objects:
            if count >= max_items:
                break
                
            # Extract metadata about the artifact
            object_name = obj.object_name
            parts = object_name.split('/')
            
            # Only process if we have enough path parts for standard Kubeflow artifacts
            # If using direct path, include all objects
            if object_path or len(parts) >= 4:  # kubeflow/pipeline-name/run-id/artifact-type/filename
                try:
                    # Get object statistics and tags if available
                    stat = client.stat_object(bucket_name, object_name)
                    tags = None
                    try:
                        tags = client.get_object_tags(bucket_name, object_name)
                    except:
                        # Tags might not be available for all objects
                        pass
                    
                    # Determine metadata from path parts
                    if len(parts) > 3:
                        obj_artifact_type = parts[3]
                        obj_run_id = parts[2] if len(parts) > 2 else "unknown"
                        obj_pipeline_name = parts[1] if len(parts) > 1 else "unknown"
                    else:
                        obj_artifact_type = "unknown"
                        obj_run_id = "unknown"
                        obj_pipeline_name = "unknown"
                    
                    # Get the actual filename
                    filename = parts[-1]
                    
                    # Create artifact info dictionary
                    artifact_info = {
                        "name": filename,
                        "path": object_name,
                        "pipeline_name": obj_pipeline_name,
                        "run_name": obj_run_id,
                        "type": obj_artifact_type,
                        "size": stat.size,
                        "last_modified": str(stat.last_modified),
                        "etag": stat.etag,
                    }
                    
                    # Add tags if available
                    if tags:
                        artifact_info["tags"] = dict(tags)
                    
                    artifacts.append(artifact_info)
                    count += 1
                except Exception as item_error:
                    logger.error(f"Error processing artifact {object_name}: {str(item_error)}")
        
        # Count total objects for this prefix
        total_objects = sum(1 for _ in client.list_objects(bucket_name, prefix=prefix, recursive=True))
        
        return {
            "bucket": bucket_name,
            "prefix": prefix,
            "pipeline_name": pipeline_name if not object_path else None,
            "run_name": run_name if not object_path else None,
            "artifact_type": artifact_type if not object_path else None,
            "total_artifact_count": total_objects,
            "displayed_artifacts": len(artifacts),
            "artifacts": artifacts
        }
            
    except Exception as e:
        logger.error(f"Error retrieving pipeline artifacts: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="Model Metrics", show_input=False)
async def get_model_metrics(
    bucket_name: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    run_name: Optional[str] = None,
    model_name: Optional[str] = None,
    max_items: int = 20,
    metrics_path: Optional[str] = None,
    **kwargs
) -> Dict:
    """
    Retrieve model evaluation metrics from MinIO storage.
    
    Args:
        bucket_name: Name of the MinIO bucket to query
        pipeline_name: Name of the pipeline (e.g. 'diabetes-svm-classification')
        run_name: Optional specific run name to get metrics for
        model_name: Optional model name to filter metrics (e.g. 'Support Vector Machine')
        max_items: Maximum number of items to return (default: 20)
        metrics_path: Optional direct path to metrics JSON file (overrides other parameters)
        
    Returns:
        Dict containing model metrics
    """
    try:
        client = await get_user_minio_client()
        
        if not bucket_name:
            return {"error": "bucket_name parameter is required. Use list_user_buckets() to discover available buckets."}
        
        # Check if bucket exists
        if not client.bucket_exists(bucket_name):
            return {"error": f"Bucket '{bucket_name}' does not exist use the list_user_buckets function to get a list of the user's buckets"}
        
        # Handle direct metrics file path if provided
        if metrics_path:
            try:
                # Retrieve the specified metrics file
                response = client.get_object(bucket_name, metrics_path)
                metrics_data = json.load(response)
                
                # Add path information
                metrics_data["artifact_path"] = metrics_path
                
                # Extract filename
                filename = metrics_path.split('/')[-1]
                metrics_data["filename"] = filename
                
                # Try to extract run and pipeline info from the path
                parts = metrics_path.split('/')
                if len(parts) >= 3 and parts[0] == "kubeflow":
                    metrics_data["pipeline_name"] = parts[1]
                    metrics_data["run_name"] = parts[2]
                
                response.close()
                response.release_conn()
                
                return {
                    "pipeline_name": metrics_data.get("pipeline_name", "unknown"),
                    "run_name": metrics_data.get("run_name", "unknown"),
                    "model_name": metrics_data.get("model_name", "unknown"),
                    "metrics_count": 1,
                    "metrics": [metrics_data]
                }
                
            except Exception as e:
                return {"error": f"Error retrieving metrics from '{metrics_path}': {str(e)}"}
        
        # Original functionality for finding metrics based on parameters
        # Get list of runs if run_id not specified
        runs = []
        if not run_name:
            run_prefix = f"kubeflow/{pipeline_name}/"
            objects = client.list_objects(bucket_name, prefix=run_prefix, recursive=False)
            
            # Extract unique run IDs
            for obj in objects:
                parts = obj.object_name.split('/')
                if len(parts) > 2:
                    runs.append(parts[2])
            
            # Get unique run IDs
            runs = list(set(runs))
        else:
            runs = [run_name]
        
        # Collect metrics for each run
        all_metrics = []
        
        for current_run in runs:
            # Construct prefix to find metrics files
            metrics_prefix = f"kubeflow/{pipeline_name}/{current_run}/metrics/"
            
            # List objects with metrics prefix
            metrics_objects = client.list_objects(bucket_name, prefix=metrics_prefix, recursive=True)
            
            for obj in metrics_objects:
                try:
                    # Download and parse the metrics JSON file
                    response = client.get_object(bucket_name, obj.object_name)
                    metrics_data = json.load(response)
                    
                    # Check if we need to filter by model name
                    if model_name and metrics_data.get("model_name") != model_name:
                        continue
                    
                    # Add run name to the metrics
                    metrics_data["run_name"] = current_run
                    metrics_data["artifact_path"] = obj.object_name
                    
                    # Extract filename
                    filename = obj.object_name.split('/')[-1]
                    metrics_data["filename"] = filename
                    
                    all_metrics.append(metrics_data)
                    
                    # Limit number of metrics based on max_items
                    if len(all_metrics) >= max_items:
                        break
                        
                except Exception as metrics_error:
                    logger.error(f"Error processing metrics file {obj.object_name}: {str(metrics_error)}")
                finally:
                    if 'response' in locals():
                        response.close()
                        response.release_conn()
                        
            # If we've reached the max_items limit, break out of the run loop
            if len(all_metrics) >= max_items:
                break
        
        # Sort metrics by run name and model name
        all_metrics.sort(key=lambda x: (x.get("run_name", ""), x.get("model_name", "")))
        
        return {
            "pipeline_name": pipeline_name,
            "run_name": run_name,
            "model_name": model_name,
            "metrics_count": len(all_metrics),
            "metrics": all_metrics
        }
            
    except Exception as e:
        logger.error(f"Error retrieving model metrics: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="Get Pipeline Visualization", show_input=False)
async def get_pipeline_visualization(
    bucket_name: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    run_name: Optional[str] = None,
    visualization_type: Optional[str] = None,
    model_name: Optional[str] = None,
    visualization_path: Optional[str] = None
) -> Dict:
    """
    Retrieve and return visualization artifacts (HTML plots) from pipeline runs.
    
    Args:
        bucket_name: Name of the MinIO bucket to query
        pipeline_name: Name of the pipeline (e.g. 'diabetes-svm-classification')
        run_name: Run name to get visualizations for
        visualization_type: Type of visualization to retrieve ('confusion_matrix', 'roc_curve', 'feature_importance')
        model_name: Optional model name part to filter by (e.g. 'svm')
        visualization_path: Optional direct path to a visualization HTML file (overrides other parameters)
        
    Returns:
        Dict containing HTML visualization content that can be displayed in the chat
    """
    try:
        client = await get_user_minio_client()
        
        if not bucket_name:
            return {"error": "bucket_name parameter is required. Use list_user_buckets() to discover available buckets."}
        
        # Check if bucket exists
        if not client.bucket_exists(bucket_name):
            return {"error": f"Bucket '{bucket_name}' does not exist use the list_user_buckets function to get a list of the user's buckets"}
        
        # If a direct visualization path is provided, use it
        if visualization_path:
            try:
                # Validate the path points to an HTML file
                if not visualization_path.endswith('.html'):
                    return {"error": f"Visualization path must point to an HTML file: {visualization_path}"}
                
                # Download the HTML content
                response = client.get_object(bucket_name, visualization_path)
                html_content = response.read().decode('utf-8')
                response.close()
                response.release_conn()
                
                # Extract filename and path parts
                filename = visualization_path.split('/')[-1]
                parts = visualization_path.split('/')
                
                # Try to determine visualization type from filename
                vis_type = "unknown"
                if "confusion_matrix" in filename:
                    vis_type = "confusion_matrix"
                elif "roc_curve" in filename:
                    vis_type = "roc_curve"
                elif "feature_importance" in filename:
                    vis_type = "feature_importance"
                
                # Try to extract pipeline and run info from path
                extracted_pipeline_name = None
                extracted_run_name = None
                if len(parts) >= 3 and parts[0] == "kubeflow":
                    extracted_pipeline_name = parts[1]
                    extracted_run_name = parts[2]
                
                return {
                    "pipeline_name": extracted_pipeline_name or "unknown",
                    "run_name": extracted_run_name or "unknown",
                    "visualization_type": vis_type,
                    "path": visualization_path,
                    "filename": filename,
                    "html_content": html_content
                }
                
            except Exception as e:
                return {"error": f"Error retrieving visualization from '{visualization_path}': {str(e)}"}
        
        # Original functionality when visualization_path is not provided
        if not pipeline_name or not run_name or not visualization_type:
            return {"error": "When not providing a direct visualization_path, you must specify pipeline_name, run_name, and visualization_type"}
            
        # Construct prefix for visualization files
        vis_prefix = f"kubeflow/{pipeline_name}/{run_name}/plots/"
        
        # Map visualization_type to expected filename pattern
        type_to_pattern = {
            "confusion_matrix": "_confusion_matrix.html",
            "roc_curve": "_roc_curve.html",
            "feature_importance": "_feature_importance.html"
        }
        
        if visualization_type not in type_to_pattern:
            return {"error": f"Invalid visualization type. Choose from: {', '.join(type_to_pattern.keys())}"}
        
        file_pattern = type_to_pattern[visualization_type]
        
        # List objects with visualization prefix
        vis_objects = client.list_objects(bucket_name, prefix=vis_prefix, recursive=True)
        
        # Find matching visualization files
        matching_files = []
        
        for obj in vis_objects:
            filename = obj.object_name.split('/')[-1]
            
            # Check if filename matches the pattern
            if file_pattern in filename:
                # Check model name if specified
                if model_name is None or model_name.lower() in filename.lower():
                    matching_files.append(obj.object_name)
        
        if not matching_files:
            return {
                "error": f"No {visualization_type} visualizations found for pipeline '{pipeline_name}' and run '{run_name}'"
            }
        
        # Get the first matching file
        vis_path = matching_files[0]
        
        # Download the HTML content
        response = client.get_object(bucket_name, vis_path)
        html_content = response.read().decode('utf-8')
        response.close()
        response.release_conn()
        
        return {
            "pipeline_name": pipeline_name,
            "run_name": run_name,
            "visualization_type": visualization_type,
            "model_name": model_name,
            "path": vis_path,
            "filename": vis_path.split('/')[-1],
            "html_content": html_content,
            "available_visualizations": [path.split('/')[-1] for path in matching_files]
        }
            
    except Exception as e:
        logger.error(f"Error retrieving pipeline visualization: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="Compare Pipeline Runs", show_input=False)
async def compare_pipeline_runs(
    bucket_name: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    run_names: Optional[List[str]] = None,
    metric_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compare multiple pipeline runs based on their metrics.
    
    Args:
        bucket_name: Name of the MinIO bucket to query
        pipeline_name: Name of the pipeline (e.g. 'diabetes-svm-classification')
        run_names: List of run names to compare
        metric_names: Optional list of specific metrics to compare (e.g. ['accuracy', 'precision'])
        
    Returns:
        Dict containing comparison results
    """
    try:
        client = await get_user_minio_client()
        
        if not bucket_name:
            return {"error": "bucket_name parameter is required. Use list_user_buckets() to discover available buckets."}
        
        if not pipeline_name:
            return {"error": "pipeline_name parameter is required."}
        
        # Check if bucket exists
        try:
            client.bucket_exists(bucket_name)
        except Exception as e:
            return {"error": f"Bucket '{bucket_name}' does not exist. Use list_user_buckets() to get a list of the user's buckets."}
        
        if not run_names or len(run_names) < 1:
            return {"error": "At least one run name must be provided"}
        
        # Collect metrics for each run
        run_metrics = []
        
        for run_name in run_names:
            # Construct prefix for metrics files
            metrics_prefix = f"kubeflow/{pipeline_name}/{run_name}/metrics/"
            
            # List objects with metrics prefix
            metrics_objects = client.list_objects(bucket_name, prefix=metrics_prefix, recursive=True)
            
            run_data = {
                "run_name": run_name,
                "models": []
            }
            
            for obj in metrics_objects:
                try:
                    # Download and parse the metrics JSON file
                    response = client.get_object(bucket_name, obj.object_name)
                    metrics_data = json.load(response)
                    
                    model_metrics = {
                        "model_name": metrics_data.get("model_name", "Unknown"),
                        "metrics": {}
                    }
                    
                    # Extract requested metrics or all available metrics
                    for metric_name, metric_value in metrics_data.items():
                        if metric_name not in ["model_name", "confusion_matrix"] and (metric_names is None or metric_name in metric_names):
                            model_metrics["metrics"][metric_name] = metric_value
                    
                    # Add parameters if available in MinIO
                    try:
                        params_prefix = f"kubeflow/{pipeline_name}/{run_name}/metadata/run_parameters.json"
                        params_response = client.get_object(bucket_name, params_prefix)
                        params_data = json.load(params_response)
                        
                        if "parameters" in params_data:
                            model_metrics["parameters"] = params_data["parameters"]
                        
                        params_response.close()
                        params_response.release_conn()
                    except:
                        # Parameters might not be available
                        pass
                    
                    run_data["models"].append(model_metrics)
                    
                except Exception as metrics_error:
                    logger.error(f"Error processing metrics file {obj.object_name}: {str(metrics_error)}")
                finally:
                    if 'response' in locals():
                        response.close()
                        response.release_conn()
            
            run_metrics.append(run_data)
        
        # Organize comparison by model and metric
        comparison = {}
        all_metrics = set()
        
        # First pass: collect all unique model names and metrics
        for run_data in run_metrics:
            for model_data in run_data["models"]:
                model_name = model_data["model_name"]
                
                if model_name not in comparison:
                    comparison[model_name] = {"runs": {}}
                
                for metric_name in model_data["metrics"].keys():
                    all_metrics.add(metric_name)
        
        # Second pass: organize by model and metric
        for run_data in run_metrics:
            run_name = run_data["run_name"]
            
            for model_data in run_data["models"]:
                model_name = model_data["model_name"]
                
                # Add run data to the model
                comparison[model_name]["runs"][run_name] = {
                    "metrics": model_data["metrics"],
                    "parameters": model_data.get("parameters", {})
                }
        
        # Find the best run for each model and metric
        for model_name, model_data in comparison.items():
            model_data["best_runs"] = {}
            
            for metric_name in all_metrics:
                # Determine if higher is better for this metric
                higher_is_better = metric_name in ["accuracy", "precision", "recall", "f1_score", "auc"]
                
                best_value = float('-inf') if higher_is_better else float('inf')
                best_run = None
                
                for run_name, run_data in model_data["runs"].items():
                    if metric_name in run_data["metrics"]:
                        metric_value = run_data["metrics"][metric_name]
                        
                        if (higher_is_better and metric_value > best_value) or \
                           (not higher_is_better and metric_value < best_value):
                            best_value = metric_value
                            best_run = run_name
                
                if best_run:
                    model_data["best_runs"][metric_name] = {
                        "run_name": best_run,
                        "value": best_value
                    }
        
        return {
            "pipeline_name": pipeline_name,
            "run_names": run_names,
            "metrics_compared": list(all_metrics),
            "models_compared": list(comparison.keys()),
            "comparison": comparison
        }
            
    except Exception as e:
        logger.error(f"Error comparing pipeline runs: {str(e)}")
        return {"error": str(e)}

# Kubeflow
@cl.step(type="tool", name="Kubeflow Pipelines", show_input=False)
async def get_kf_pipelines(
    search_term: Optional[str] = None, 
    page_size: int = 20, 
    page_token: str = "",
    sort_by: str = "created_at desc"
) -> Dict:
    """
    Retrieve information about the user's Kubeflow pipelines with flexible search capabilities.
    
    Args:
        search_term: Optional term to search for in pipeline names (case-insensitive partial match)
        page_size: Number of results to return per page (default: 20)
        page_token: Token for pagination (default: empty string for first page)
        sort_by: How to sort results (default: created_at desc - newest first)
        
    Returns:
        Dict containing filtered pipeline information
    """
    try:
        kf_client = await get_user_kubeflow_client()

        namespace = UserSessionManager.get_kubeflow_namespace() or os.getenv('KUBEFLOW_NAMESPACE')
        
        # Get pipelines without filter - we'll filter on the client side
        response = kf_client.list_pipelines(
            page_token=page_token,
            page_size=page_size,
            sort_by=sort_by,
            namespace=namespace
        )
        
        # Extract pipeline data
        pipelines = response.pipelines or []
        
        # Filter pipelines by search term if provided
        filtered_pipelines = pipelines
        if search_term and search_term.strip():
            search_term_lower = search_term.lower()
            filtered_pipelines = [
                p for p in pipelines 
                if (p.display_name and search_term_lower in p.display_name.lower()) or
                   (p.description and search_term_lower in p.description.lower())
            ]
        
        # Build response
        result = {
            "total_pipelines": len(filtered_pipelines),
            "total_available": len(pipelines),
            "next_page_token": response.next_page_token,
            "namespace_type": "private" if namespace else "shared",
            "namespace": namespace or "shared",
            "pipelines": [{
                "id": p.pipeline_id,
                "name": p.display_name,
                "description": p.description,
                "created_at": str(p.created_at)
            } for p in filtered_pipelines]
        }
        
        return result
            
    except Exception as e:
        logger.error(f"Error retrieving Kubeflow information: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="Kubeflow Pipeline Details", show_input=False)
async def get_pipeline_details(pipeline_id: str) -> Dict:
    """
    Retrieve detailed information about a specific Kubeflow pipeline.
    
    Args:
        pipeline_id: The unique identifier of the pipeline
        
    Returns:
        Dict containing detailed pipeline information
    """
    try:
        if not pipeline_id or not pipeline_id.strip():
            return {"error": "Pipeline ID is required"}
            
        kf_client = await get_user_kubeflow_client()
        pipeline = kf_client.get_pipeline(pipeline_id=pipeline_id)
        
        # Extract the pipeline details
        pipeline_details = {
            "id": pipeline.pipeline_id,
            "name": pipeline.display_name,
            "description": pipeline.description,
            "created_at": str(pipeline.created_at),
            "updated_at": str(pipeline.updated_at) if hasattr(pipeline, 'updated_at') else None,
            "namespace": pipeline.namespace if hasattr(pipeline, 'namespace') else "unknown",
            "default_version": {
                "id": pipeline.default_version.pipeline_version_id if hasattr(pipeline, 'default_version') and pipeline.default_version else None,
                "name": pipeline.default_version.display_name if hasattr(pipeline, 'default_version') and pipeline.default_version else None,
                "created_at": str(pipeline.default_version.created_at) if hasattr(pipeline, 'default_version') and pipeline.default_version else None
            } if hasattr(pipeline, 'default_version') and pipeline.default_version else None,
            "pipeline_versions": [
                {
                    "id": version.pipeline_version_id,
                    "name": version.display_name,
                    "created_at": str(version.created_at)
                } for version in kf_client.list_pipeline_versions(pipeline_id=pipeline_id).pipeline_versions
            ] if hasattr(kf_client, 'list_pipeline_versions') else []
        }
        
        return pipeline_details
            
    except Exception as e:
        logger.error(f"Error retrieving Kubeflow pipeline details: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="Kubeflow Pipeline Details", show_input=False)
async def get_pipeline_version_details(pipeline_id: str, pipeline_version_id: str) -> Dict:
    """
    Retrieve detailed information about a specific Kubeflow pipeline version,
    including code, components, and parameters.
    
    Args:
        pipeline_id: The unique identifier of the pipeline
        pipeline_version_id: The unique identifier of the pipeline version
        
    Returns:
        Dict containing detailed pipeline version information
    """
    try:
        if not pipeline_id or not pipeline_id.strip():
            return {"error": "Pipeline ID is required"}
            
        if not pipeline_version_id or not pipeline_version_id.strip():
            return {"error": "Pipeline Version ID is required"}
            
        kf_client = await get_user_kubeflow_client()
        pipeline_version = kf_client.get_pipeline_version(
            pipeline_id=pipeline_id,
            pipeline_version_id=pipeline_version_id
        )
        
        # Extract the pipeline version details
        pipeline_version_details = {
            "pipeline_id": pipeline_id,
            "version_id": pipeline_version.pipeline_version_id,
            "name": pipeline_version.display_name,
            "description": pipeline_version.description,
            "created_at": str(pipeline_version.created_at),
            "code_source_url": pipeline_version.code_source_url if hasattr(pipeline_version, 'code_source_url') else None
        }
        
        # Try to extract pipeline specification (YAML/JSON) if available
        if hasattr(pipeline_version, 'pipeline_spec') and pipeline_version.pipeline_spec:
            # The pipeline spec might be in various formats (YAML/JSON string or object)
            spec = pipeline_version.pipeline_spec
            
            # Try to extract components from spec
            components = []
            params = []
            
            # If it's a string, try to parse it
            if isinstance(spec, str):
                try:
                    if spec.strip().startswith('{'):
                        # JSON format
                        spec_obj = json.loads(spec)
                    else:
                        # YAML format - need to import yaml for this
                        import yaml
                        spec_obj = yaml.safe_load(spec)
                        
                    # Extract components and params from spec_obj if possible
                    if isinstance(spec_obj, dict):
                        if 'components' in spec_obj:
                            for component_id, component_info in spec_obj['components'].items():
                                components.append({
                                    "id": component_id,
                                    "name": component_info.get('name', component_id),
                                    "description": component_info.get('description', '')
                                })
                                
                        if 'inputs' in spec_obj:
                            for param_id, param_info in spec_obj['inputs'].items():
                                params.append({
                                    "name": param_id,
                                    "type": param_info.get('type', 'unknown'),
                                    "description": param_info.get('description', ''),
                                    "default": param_info.get('default', None)
                                })
                except Exception as parse_error:
                    logger.error(f"Error parsing pipeline spec: {str(parse_error)}")
                    pipeline_version_details["spec_parse_error"] = str(parse_error)
            
            # Add the original spec and parsed components/params to the response
            pipeline_version_details["pipeline_spec"] = spec if isinstance(spec, str) else json.dumps(spec)
            pipeline_version_details["components"] = components
            pipeline_version_details["parameters"] = params
            
        return pipeline_version_details
            
    except Exception as e:
        logger.error(f"Error retrieving Kubeflow pipeline version details: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="Run Kubeflow Pipeline", show_input=False)
async def run_pipeline(
    experiment_id: str,
    job_name: str,
    params: Optional[Dict[str, Any]] = None,
    pipeline_id: Optional[str] = None,
    version_id: Optional[str] = None,
    pipeline_package_path: Optional[str] = None,
    pipeline_root: Optional[str] = None,
    enable_caching: Optional[bool] = True,
    service_account: Optional[str] = None
) -> Dict:
    """
    Run a specified Kubeflow pipeline with the given parameters.
    
    Args:
        experiment_id: ID of the experiment to run the pipeline in
        job_name: Name for this pipeline run job
        params: Dictionary of parameter name-value pairs for the pipeline
        pipeline_id: ID of the pipeline to run (use with or without version_id)
        version_id: Optional specific version of the pipeline to run
        pipeline_package_path: Alternative to pipeline_id - local path to pipeline package file
        pipeline_root: Root path for pipeline outputs
        enable_caching: Whether to enable caching for the pipeline tasks
        service_account: Kubernetes service account to use for this run
        
    Returns:
        Dict containing run information and status
    """
    try:
        # Validate required inputs
        if not experiment_id or not experiment_id.strip():
            return {"error": "Experiment ID is required"}
            
        if not job_name or not job_name.strip():
            return {"error": "Job name is required"}
            
        # Validate that either pipeline_id or pipeline_package_path is provided
        if not pipeline_id and not pipeline_package_path:
            return {"error": "Either pipeline_id or pipeline_package_path must be specified"}
            
        # Get Kubeflow client
        kf_client = await get_user_kubeflow_client()
        
        # Prepare parameters
        if params is None:
            params = {}
            
        # Run the pipeline
        run = kf_client.run_pipeline(
            experiment_id=experiment_id,
            job_name=job_name,
            params=params,
            pipeline_id=pipeline_id,
            version_id=version_id,
            pipeline_package_path=pipeline_package_path,
            pipeline_root=pipeline_root,
            enable_caching=enable_caching,
            service_account=service_account
        )
        
        # Format run information as a dictionary
        run_info = {
            "run_id": run.run_id,
            "name": run.display_name,
            "status": run.state,
            "created_at": str(run.created_at),
            "pipeline_id": pipeline_id,
            "pipeline_version_id": version_id,
            "experiment_id": experiment_id,
            "service_account": service_account,
            "params": params
        }
        
        return run_info
            
    except Exception as e:
        logger.error(f"Error running Kubeflow pipeline: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="List Kubeflow Runs", show_input=False)
async def list_runs(
    search_term: Optional[str] = None,
    experiment_id: Optional[str] = None,
    namespace: Optional[str] = None, 
    page_size: int = 20, 
    page_token: str = "",
    sort_by: str = "created_at desc",
    status_filter: Optional[str] = None
) -> Dict:
    """
    Retrieve information about Kubeflow pipeline runs with flexible search capabilities.
    
    Args:
        search_term: Optional term to search for in run names (case-insensitive partial match)
        experiment_id: Optional experiment ID to filter runs
        namespace: Optional namespace to filter runs
        page_size: Number of results to return per page (default: 20)
        page_token: Token for pagination (default: empty string for first page)
        sort_by: How to sort results (default: created_at desc - newest first)
        status_filter: Optional filter for run status (e.g. 'SUCCEEDED', 'FAILED', 'RUNNING')
        
    Returns:
        Dict containing filtered run information
    """
    try:
        kf_client = await get_user_kubeflow_client()
        
        # Use user's namespace if not explicitly provided
        if not namespace:
            namespace = UserSessionManager.get_kubeflow_namespace()
        
        # Get runs with API-level filtering
        filter_str = None
        if status_filter:
            filter_dict = {
                "predicates": [{
                    "operation": "EQUALS",
                    "key": "state",
                    "stringValue": status_filter.upper(),
                }]
            }
            filter_str = json.dumps(filter_dict)
            
        response = kf_client.list_runs(
            page_token=page_token,
            page_size=page_size,
            sort_by=sort_by,
            experiment_id=experiment_id,
            namespace=namespace,
            filter=filter_str
        )
        
        # Extract run data
        runs = response.runs or []
        
        # Filter runs by search term if provided (client-side filtering)
        filtered_runs = runs
        if search_term and search_term.strip():
            search_term_lower = search_term.lower()
            filtered_runs = [
                r for r in runs 
                if (r.display_name and search_term_lower in r.display_name.lower()) or
                   (hasattr(r, 'description') and r.description and search_term_lower in r.description.lower())
            ]
        
        # Build response with detailed run information
        result = {
            "total_runs": len(filtered_runs),
            "total_available": len(runs),
            "next_page_token": response.next_page_token,
            "experiment_id": experiment_id,
            "namespace": namespace or "default",
            "runs": [{
                "id": r.run_id,
                "name": r.display_name,
                "state": r.state,
                "created_at": str(r.created_at),
                "finished_at": str(r.finished_at) if hasattr(r, 'finished_at') and r.finished_at else None,
                "pipeline_id": r.pipeline_id if hasattr(r, 'pipeline_id') else None,
                "pipeline_version_id": r.pipeline_version_id if hasattr(r, 'pipeline_version_id') else None,
                "experiment_id": r.experiment_id if hasattr(r, 'experiment_id') else experiment_id,
            } for r in filtered_runs]
        }
        
        return result
            
    except Exception as e:
        logger.error(f"Error retrieving Kubeflow run information: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="Kubeflow Run Details", show_input=False)
async def get_run_details(run_id: str) -> Dict:
    """
    Retrieve detailed information about a specific Kubeflow pipeline run.
    
    Args:
        run_id: The unique identifier of the run
        
    Returns:
        Dict containing detailed run information
    """
    try:
        if not run_id or not run_id.strip():
            return {"error": "Run ID is required"}
            
        kf_client = await get_user_kubeflow_client()
        run = kf_client.get_run(run_id=run_id)
        
        # Extract the run details
        run_details = {
            "id": run.run_id,
            "name": run.display_name,
            "state": run.state,
            "created_at": str(run.created_at),
            "finished_at": str(run.finished_at) if hasattr(run, 'finished_at') and run.finished_at else None,
            "pipeline_id": run.pipeline_id if hasattr(run, 'pipeline_id') else None,
            "pipeline_name": run.pipeline_name if hasattr(run, 'pipeline_name') else None,
            "pipeline_version_id": run.pipeline_version_id if hasattr(run, 'pipeline_version_id') else None,
            "pipeline_version_name": run.pipeline_version_name if hasattr(run, 'pipeline_version_name') else None,
            "experiment_id": run.experiment_id if hasattr(run, 'experiment_id') else None,
            "experiment_name": run.experiment_name if hasattr(run, 'experiment_name') else None,
            "service_account": run.service_account if hasattr(run, 'service_account') else None,
            "scheduled_at": str(run.scheduled_at) if hasattr(run, 'scheduled_at') and run.scheduled_at else None,
            "runtime_config": {
                "parameters": run.runtime_config.parameters if hasattr(run, 'runtime_config') and hasattr(run.runtime_config, 'parameters') else {}
            } if hasattr(run, 'runtime_config') else {},
            "error": {
                "code": run.error.code if hasattr(run.error, 'code') else None,
                "message": run.error.message if hasattr(run.error, 'message') else None
            } if hasattr(run, 'error') and run.error else None
        }
        
        return run_details
            
    except Exception as e:
        logger.error(f"Error retrieving Kubeflow run details: {str(e)}")
        return {"error": str(e)}

## Kubeflow Experiments
@cl.step(type="tool", name="List Kubeflow Experiments", show_input=False)
async def list_experiments(
    search_term: Optional[str] = None,
    namespace: Optional[str] = None, 
    page_size: int = 10, 
    page_token: str = "",
    sort_by: str = "created_at desc"
) -> Dict:
    """
    Retrieve information about Kubeflow experiments with flexible search capabilities.
    
    Args:
        search_term: Optional term to search for in experiment names (case-insensitive partial match)
        namespace: Optional namespace to filter experiments
        page_size: Number of results to return per page (default: 10)
        page_token: Token for pagination (default: empty string for first page)
        sort_by: How to sort results (default: created_at desc - newest first)
        
    Returns:
        Dict containing filtered experiment information
    """
    try:
        kf_client = await get_user_kubeflow_client()
        
        # Use user's namespace if not explicitly provided
        if not namespace:
            namespace = UserSessionManager.get_kubeflow_namespace()
        
        # Create filter if search term is provided
        filter_str = None
        if search_term and search_term.strip():
            filter_dict = {
                "predicates": [{
                    "operation": "EQUALS",
                    "key": "display_name",
                    "stringValue": search_term,
                }]
            }
            filter_str = json.dumps(filter_dict)
        
        # Get experiments with the given parameters
        response = kf_client.list_experiments(
            page_token=page_token,
            page_size=page_size,
            sort_by=sort_by,
            namespace=namespace,
            filter=filter_str
        )
        
        # Extract experiment data
        experiments = response.experiments or []
        
        # If we used an exact match filter but want partial match, we need to filter client-side
        filtered_experiments = experiments
        if search_term and search_term.strip() and not filter_str:
            search_term_lower = search_term.lower()
            filtered_experiments = [
                e for e in experiments 
                if (e.display_name and search_term_lower in e.display_name.lower()) or
                   (e.description and search_term_lower in e.description.lower())
            ]
        
        # Build response
        result = {
            "total_experiments": len(filtered_experiments),
            "total_available": len(experiments),
            "next_page_token": response.next_page_token,
            "namespace": namespace or "default",
            "experiments": [{
                "id": e.experiment_id,
                "name": e.display_name,
                "description": e.description,
                "created_at": str(e.created_at)
            } for e in filtered_experiments]
        }
        
        return result
            
    except Exception as e:
        logger.error(f"Error retrieving Kubeflow experiment information: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="Get Experiment Details", show_input=False)
async def get_experiment_details(
    experiment_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
    namespace: Optional[str] = None
) -> Dict:
    """
    Retrieve detailed information about a specific Kubeflow experiment.
    
    Args:
        experiment_id: Optional ID of the experiment to retrieve details for
        experiment_name: Optional name of the experiment to retrieve details for
        namespace: Optional namespace where the experiment is located
        
    Returns:
        Dict containing detailed experiment information
    """
    try:
        if not experiment_id and not experiment_name:
            return {"error": "Either experiment_id or experiment_name must be provided"}
            
        kf_client = await get_user_kubeflow_client()
        
        # Use user's namespace if not explicitly provided
        if not namespace:
            namespace = UserSessionManager.get_kubeflow_namespace()
        
        # Get experiment details
        experiment = kf_client.get_experiment(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            namespace=namespace
        )
        
        # Extract the experiment details
        experiment_details = {
            "id": experiment.experiment_id,
            "name": experiment.display_name,
            "description": experiment.description,
            "created_at": str(experiment.created_at),
            "namespace": namespace or "default" if namespace else experiment.namespace if hasattr(experiment, 'namespace') else "default",
            "storage_state": experiment.storage_state if hasattr(experiment, 'storage_state') else None
        }
        
        # Try to get associated runs if possible
        try:
            runs_response = kf_client.list_runs(
                experiment_id=experiment.experiment_id,
                page_size=10,  # Limit to 10 most recent runs
                sort_by="created_at desc"
            )
            
            if runs_response.runs:
                experiment_details["recent_runs"] = [{
                    "id": r.run_id,
                    "name": r.display_name,
                    "state": r.state,
                    "created_at": str(r.created_at)
                } for r in runs_response.runs]
                experiment_details["run_count"] = len(runs_response.runs)
        except Exception as runs_error:
            logger.error(f"Error retrieving runs for experiment: {str(runs_error)}")
            # We'll continue without run information
        
        return experiment_details
            
    except Exception as e:
        logger.error(f"Error retrieving Kubeflow experiment details: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="Get User Namespace", show_input=False)
async def get_user_namespace() -> Dict:
    """
    Gets user namespace in context config.
    
    Returns:
        Dict containing the Kubernetes namespace from the local context file or empty if it wasn't set
    """
    try:
        kf_client = await get_user_kubeflow_client()
        namespace = kf_client.get_user_namespace()
        
        return {
            "namespace": namespace if namespace else "",
            "message": "Namespace retrieved successfully" if namespace else "No namespace found in context"
        }
            
    except Exception as e:
        logger.error(f"Error retrieving user namespace: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="Create Experiment", show_input=False)
async def create_experiment(
    name: str,
    description: Optional[str] = None,
    namespace: Optional[str] = None
) -> Dict:
    """
    Creates a new experiment.
    
    Args:
        name: Name of the experiment
        description: Optional description of the experiment
        namespace: Optional Kubernetes namespace to use
        
    Returns:
        Dict containing the created experiment details
    """
    try:
        if not name or not name.strip():
            return {"error": "Experiment name is required"}
            
        kf_client = await get_user_kubeflow_client()
        
        # Use user's namespace if not explicitly provided
        if not namespace:
            namespace = UserSessionManager.get_kubeflow_namespace()
        experiment = kf_client.create_experiment(
            name=name,
            description=description,
            namespace=namespace
        )
        
        # Extract the experiment details
        experiment_details = {
            "id": experiment.experiment_id,
            "name": experiment.display_name,
            "description": experiment.description,
            "namespace": namespace or experiment.namespace if hasattr(experiment, 'namespace') else "default",
            "created_at": str(experiment.created_at) if hasattr(experiment, 'created_at') else None
        }
        
        return experiment_details
            
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="Get Pipeline ID", show_input=False)
async def get_pipeline_id(name: str) -> Dict:
    """
    Gets the ID of a pipeline by its name.
    
    Args:
        name: Pipeline name
        
    Returns:
        Dict containing the pipeline ID if a pipeline with the name exists
    """
    try:
        if not name or not name.strip():
            return {"error": "Pipeline name is required"}
            
        kf_client = await get_user_kubeflow_client()
        pipeline_id = kf_client.get_pipeline_id(name=name)
        
        if not pipeline_id:
            return {
                "message": f"No pipeline found with name '{name}'",
                "pipeline_id": None
            }
            
        return {
            "pipeline_id": pipeline_id,
            "pipeline_name": name,
            "message": "Pipeline ID retrieved successfully"
        }
            
    except Exception as e:
        logger.error(f"Error retrieving pipeline ID: {str(e)}")
        return {"error": str(e)}

@cl.step(type="tool", name="PDF Parser", show_input=False)
async def parse_pdf_from_minio(
    bucket_name: str,
    object_path: str,
    summarize: bool = False,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    summary_type: str = "map_reduce"
) -> Dict:
    """
    Download and parse text from a PDF file stored in MinIO.
    Extracts text content, chunks it for processing, and optionally provides summarization.
    
    Args:
        bucket_name: Name of the MinIO bucket containing the PDF file
        object_path: Full path to the PDF file in MinIO
        summarize: Whether to generate a summary of the PDF content
        chunk_size: Character size for text chunks when splitting the document
        chunk_overlap: Number of characters to overlap between chunks
        summary_type: Type of summarization chain ('map_reduce', 'refine', or 'stuff')
        
    Returns:
        Dict containing extracted text, chunks, optional summary, and metadata
    """
    try:
        # Validate PDF file extension
        if not object_path.lower().endswith('.pdf'):
            return {"error": f"File '{object_path}' is not a PDF file. Only PDF files are supported."}
        
        # Get MinIO client
        client = await get_user_minio_client()
        
        # Validate bucket exists
        if not bucket_name:
            return {"error": "bucket_name parameter is required. Use list_user_buckets() to discover available buckets."}
        
        if not client.bucket_exists(bucket_name):
            return {"error": f"Bucket '{bucket_name}' does not exist. Use list_user_buckets() to get a list of available buckets."}
        
        # Download PDF from MinIO
        try:
            stat = client.stat_object(bucket_name, object_path)
            response = client.get_object(bucket_name, object_path)
            pdf_bytes = response.read()
            response.close()
            response.release_conn()
        except S3Error as e:
            if "NoSuchKey" in str(e):
                return {"error": f"PDF file '{object_path}' not found in bucket '{bucket_name}'."}
            else:
                logger.error(f"Error downloading PDF from MinIO: {str(e)}")
                return {"error": f"Error downloading PDF: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error downloading PDF: {str(e)}")
            return {"error": f"Unexpected error downloading PDF: {str(e)}"}
        
        # Display PDF in the UI
        try:
            pdf_filename = object_path.split('/')[-1]
            pdf_element = cl.Pdf(
                name=pdf_filename,
                display="side",
                content=pdf_bytes,
                page=1
            )
            # Send message with PDF element (name must be in content for side/page display)
            await cl.Message(
                content=f"Source PDF: {pdf_filename}",
                elements=[pdf_element]
            ).send()
        except Exception as e:
            logger.warning(f"Error displaying PDF element: {str(e)}")
        
        # Write PDF bytes to temporary file for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Load PDF using PyPDFLoader
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            
            if not documents:
                return {"error": "PDF file appears to be empty or could not be parsed."}
            
            # Extract full text
            full_text = "\n\n".join([doc.page_content for doc in documents])
            page_count = len(documents)
            
            # Chunk the text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            
            chunks = text_splitter.split_documents(documents)
            chunk_count = len(chunks)
            
            # Prepare result structure
            result = {
                "bucket": bucket_name,
                "object_path": object_path,
                "filename": object_path.split('/')[-1],
                "file_size": stat.size,
                "page_count": page_count,
                "chunk_count": chunk_count,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "full_text": full_text,
                "chunks": [{"chunk_index": i, "text": chunk.page_content, "page": chunk.metadata.get("page", None)} for i, chunk in enumerate(chunks)]
            }
            
            # Optional summarization
            if summarize:
                try:
                    # Initialize LLM for summarization
                    # Fix for Pydantic 2.11+ compatibility - rebuild model before instantiation
                    try:
                        ChatOpenAI.model_rebuild()
                    except Exception:
                        # If rebuild fails, continue anyway - may work without it
                        pass
                    
                    llm = ChatOpenAI(
                        model=LLM_MODEL,
                        temperature=LLM_TEMPERATURE,
                        max_tokens=LLM_MAX_TOKENS
                    )
                    
                    # Choose summarization chain based on summary_type
                    if summary_type == "stuff":
                        # Stuff chain - good for small documents
                        chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)
                        summary = await asyncio.to_thread(chain.run, chunks)
                    elif summary_type == "refine":
                        # Refine chain - iteratively refines summary
                        chain = load_summarize_chain(llm, chain_type="refine", verbose=False)
                        summary = await asyncio.to_thread(chain.run, chunks)
                    else:  # map_reduce (default)
                        # Map-Reduce chain - summarizes each chunk then combines
                        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
                        summary = await asyncio.to_thread(chain.run, chunks)
                    
                    result["summary"] = summary
                    result["summary_type"] = summary_type
                except Exception as e:
                    logger.error(f"Error generating summary: {str(e)}")
                    result["summary_error"] = f"Failed to generate summary: {str(e)}"
                    result["summary"] = None
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {str(e)}")
            return {"error": f"Error parsing PDF file: {str(e)}. The file may be corrupted or in an unsupported format."}
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Error deleting temporary file: {str(e)}")
                
    except Exception as e:
        logger.error(f"Unexpected error in parse_pdf_from_minio: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

@cl.step(type="tool", name="Plot Data", show_input=False)
async def plot_data(
    data: Any,
    chart_type: Optional[str] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    color_column: Optional[str] = None,
    display: str = "inline",
    size: str = "medium"
) -> Dict:
    """
    Create an interactive Plotly visualization from data. The agent should intelligently 
    decide on the chart type (line, bar, pie, scatter) based on the data structure if 
    chart_type is not specified.
    
    Args:
        data: The data to plot. Can be:
            - A list of numbers (single series)
            - A list of lists (multiple series)
            - A dict with keys as labels and values as data
            - A list of dicts (tabular data)
        chart_type: Optional chart type ('line', 'bar', 'pie', 'scatter', 'histogram').
                    If not provided, will be automatically determined.
        title: Optional title for the chart
        x_label: Optional label for x-axis
        y_label: Optional label for y-axis
        x_column: Optional column name for x-axis (for dict/list of dicts data)
        y_column: Optional column name for y-axis (for dict/list of dicts data)
        color_column: Optional column name for color grouping (for dict/list of dicts data)
        display: How to display the chart ('inline', 'side', or 'page'). Default: 'inline'
        size: Size of the chart ('small', 'medium', or 'large'). Default: 'medium'
        
    Returns:
        Dict containing the Plotly figure as JSON and metadata
    """
    try:
        # Determine chart type if not provided
        if not chart_type:
            chart_type = _determine_chart_type(data, x_column, y_column)
            logger.info(f"Auto-determined chart type: {chart_type}")
        
        # Create the appropriate Plotly figure
        fig = _create_plotly_figure(
            data, chart_type, title, x_label, y_label, 
            x_column, y_column, color_column
        )
        
        # Convert figure to JSON for serialization
        fig_json = fig.to_json()
        
        return {
            "figure_json": fig_json,
            "chart_type": chart_type,
            "title": title or "Data Visualization",
            "display": display,
            "size": size,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error creating plot: {str(e)}")
        return {"error": f"Error creating plot: {str(e)}", "success": False}


def _determine_chart_type(data: Any, x_column: Optional[str] = None, y_column: Optional[str] = None) -> str:
    """
    Intelligently determine the best chart type based on data structure.
    """
    # If data is a list of numbers
    if isinstance(data, list) and len(data) > 0:
        if all(isinstance(x, (int, float)) for x in data):
            # Single series of numbers - use line chart
            return "line"
        elif all(isinstance(x, list) and len(x) == 2 for x in data):
            # List of [x, y] pairs - use scatter
            return "scatter"
        elif all(isinstance(x, dict) for x in data):
            # List of dicts - check structure
            if x_column and y_column:
                # Has explicit x and y columns
                y_values = [d.get(y_column) for d in data if y_column in d]
                if all(isinstance(v, (int, float)) for v in y_values):
                    # Check if x is categorical
                    x_values = [d.get(x_column) for d in data if x_column in d]
                    if len(set(x_values)) < len(x_values) * 0.5:
                        # Many repeated x values - use bar chart
                        return "bar"
                    else:
                        return "line"
            # Try to infer
            if len(data) > 0:
                keys = list(data[0].keys())
                if len(keys) == 2:
                    # Two columns - likely x and y
                    return "scatter"
                elif len(keys) > 2:
                    # Multiple columns - default to line
                    return "line"
    
    # If data is a dict
    elif isinstance(data, dict):
        values = list(data.values())
        if all(isinstance(v, (int, float)) for v in values):
            # Dict with numeric values
            if len(data) <= 10:
                # Small number of categories - use pie or bar
                return "pie"
            else:
                return "bar"
        elif all(isinstance(v, list) for v in values):
            # Dict with lists as values - multiple series
            return "line"
    
    # Default to line chart
    return "line"


def _create_plotly_figure(
    data: Any,
    chart_type: str,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    color_column: Optional[str] = None
) -> go.Figure:
    """
    Create a Plotly figure based on the chart type and data.
    """
    title = title or "Data Visualization"
    
    # Handle different data types
    if isinstance(data, list) and len(data) > 0:
        # List of numbers
        if all(isinstance(x, (int, float)) for x in data):
            if chart_type == "line":
                fig = go.Figure(data=go.Scatter(y=data, mode='lines+markers'))
            elif chart_type == "bar":
                fig = go.Figure(data=go.Bar(y=data))
            elif chart_type == "pie":
                fig = go.Figure(data=go.Pie(values=data, labels=[f"Item {i+1}" for i in range(len(data))]))
            else:
                fig = go.Figure(data=go.Scatter(y=data, mode='lines+markers'))
        
        # List of [x, y] pairs
        elif all(isinstance(x, list) and len(x) == 2 for x in data):
            x_vals = [item[0] for item in data]
            y_vals = [item[1] for item in data]
            if chart_type == "scatter":
                fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='markers'))
            elif chart_type == "line":
                fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='lines+markers'))
            else:
                fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='lines+markers'))
        
        # List of dicts (tabular data)
        elif all(isinstance(x, dict) for x in data):
            if x_column and y_column:
                x_vals = [d.get(x_column) for d in data]
                y_vals = [d.get(y_column) for d in data]
                
                if chart_type == "bar":
                    fig = go.Figure(data=go.Bar(x=x_vals, y=y_vals))
                elif chart_type == "scatter":
                    fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='markers'))
                elif chart_type == "line":
                    fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='lines+markers'))
                elif chart_type == "pie":
                    fig = go.Figure(data=go.Pie(labels=x_vals, values=y_vals))
                else:
                    fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='lines+markers'))
            else:
                # Try to infer columns from first dict
                if len(data) > 0:
                    keys = list(data[0].keys())
                    if len(keys) >= 2:
                        x_col = keys[0]
                        y_col = keys[1]
                        x_vals = [d.get(x_col) for d in data]
                        y_vals = [d.get(y_col) for d in data]
                        if chart_type == "bar":
                            fig = go.Figure(data=go.Bar(x=x_vals, y=y_vals))
                        elif chart_type == "pie":
                            fig = go.Figure(data=go.Pie(labels=x_vals, values=y_vals))
                        else:
                            fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='lines+markers'))
                    else:
                        # Single column - use index as x
                        y_col = keys[0]
                        y_vals = [d.get(y_col) for d in data]
                        if chart_type == "bar":
                            fig = go.Figure(data=go.Bar(y=y_vals))
                        elif chart_type == "pie":
                            fig = go.Figure(data=go.Pie(values=y_vals, labels=[f"Item {i+1}" for i in range(len(y_vals))]))
                        else:
                            fig = go.Figure(data=go.Scatter(y=y_vals, mode='lines+markers'))
                else:
                    raise ValueError("Empty data list")
        else:
            raise ValueError(f"Unsupported data format: {type(data[0])}")
    
    # Dict data
    elif isinstance(data, dict):
        labels = list(data.keys())
        values = list(data.values())
        
        if chart_type == "pie":
            fig = go.Figure(data=go.Pie(labels=labels, values=values))
        elif chart_type == "bar":
            fig = go.Figure(data=go.Bar(x=labels, y=values))
        elif chart_type == "line":
            fig = go.Figure(data=go.Scatter(x=labels, y=values, mode='lines+markers'))
        else:
            # Default to bar for dict data
            fig = go.Figure(data=go.Bar(x=labels, y=values))
    
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label or "",
        yaxis_title=y_label or "",
        template="plotly_white",
        hovermode='closest'
    )
    
    return fig


# Smart Cities Pilot Analysis Tool

# Broadened patterns for detecting time-related columns
TIME_PATTERNS = ['time', 'duration', 'processing', 'elapsed', 'seconds', 'ms', 'latency', 'response']

# Pilot access configuration - maps email patterns to pilots
PILOT_CONFIG = {
    "smart_cities": {
        "allowed_emails": ["cities-pilot@humaine.com"],
        "email_patterns": ["@cities", "smart-cities", "cities-pilot"],
        "default_bucket": "smart-cities-data"
    },
    # Future pilots can be added here
    # "healthcare": {
    #     "allowed_emails": ["health-pilot@humaine.com"],
    #     "email_patterns": ["@health", "healthcare"],
    #     "default_bucket": "healthcare-data"
    # }
}


def _get_user_pilot_context() -> Dict:
    """
    Get the pilot context for current user based on their email/identity.
    Uses UserSessionManager to retrieve user information and matches against PILOT_CONFIG.
    
    Returns:
        Dict containing:
        - pilot: Name of the matched pilot (or None)
        - default_bucket: Default bucket for the pilot (or None)
        - has_access: Boolean indicating if user has pilot access
        - user_email: The user's email address
    """
    user_email = UserSessionManager.get_user_id() or ""
    user_policies = UserSessionManager.get_user_policies() or []
    
    for pilot_name, config in PILOT_CONFIG.items():
        # Check explicit email match
        if user_email in config["allowed_emails"]:
            return {
                "pilot": pilot_name,
                "default_bucket": config["default_bucket"],
                "has_access": True,
                "user_email": user_email
            }
        # Check email pattern match
        for pattern in config.get("email_patterns", []):
            if pattern.lower() in user_email.lower():
                return {
                    "pilot": pilot_name,
                    "default_bucket": config["default_bucket"],
                    "has_access": True,
                    "user_email": user_email
                }
        # Check policy-based access (MinIO policies may indicate pilot access)
        for policy in user_policies:
            if pilot_name.replace("_", "-") in policy.lower():
                return {
                    "pilot": pilot_name,
                    "default_bucket": config["default_bucket"],
                    "has_access": True,
                    "user_email": user_email
                }
    
    return {
        "pilot": None,
        "default_bucket": None,
        "has_access": False,
        "user_email": user_email
    }

@cl.step(type="tool", name="Smart Cities Analysis", show_input=False)
async def analyze_smart_cities_data(
    bucket_name: str,
    object_path: str,
    query_type: str,
    filter_value: Optional[str] = None
) -> Dict:
    """
    Analyze Smart Cities pilot application data from pickle files stored in MinIO.
    Provides predefined analysis functions for error types, AI decisions, operator decisions,
    and processing performance metrics.
    
    Args:
        bucket_name: Name of the MinIO bucket containing the pickle file
        object_path: Full path to the pickle file in MinIO
        query_type: Type of analysis to perform. Options:
            - 'overview': Dataset summary (row count, column groups, value distributions)
            - 'error_distribution': Count of applications by error type
            - 'ai_decisions': Distribution of AI decisions (Accepted/Rejected/Flagged)
            - 'operator_decisions': Distribution of operator review outcomes
            - 'processing_time': AI and operator processing time statistics
            - 'field_errors': Which fields have most discrepancies between GT_ and APP_
            - 'decision_flow': Flow from AI decision to final operator outcome
        filter_value: Optional filter value for specific queries
        
    Returns:
        Dict containing analysis results based on the query_type
    """
    try:
        # Check user access using dynamic pilot context
        pilot_context = _get_user_pilot_context()
        if pilot_context["pilot"] != "smart_cities" or not pilot_context["has_access"]:
            logger.warning(f"Access denied for user {pilot_context['user_email']} to Smart Cities tool")
            return {
                "error": "Access denied. This tool is only available to Smart Cities pilot users.",
                "user_email": pilot_context["user_email"],
                "detected_pilot": pilot_context["pilot"]
            }
        
        # Validate pickle file extension
        if not object_path.lower().endswith('.pkl') and not object_path.lower().endswith('.pickle'):
            return {"error": f"File '{object_path}' is not a pickle file. Only .pkl or .pickle files are supported."}
        
        # Validate query_type
        valid_query_types = [
            'overview', 'error_distribution', 'ai_decisions', 
            'operator_decisions', 'processing_time', 'field_errors', 'decision_flow',
            'confusion_matrix', 'ai_accuracy_metrics'
        ]
        if query_type not in valid_query_types:
            return {"error": f"Invalid query_type '{query_type}'. Valid options: {valid_query_types}"}
        
        # Get MinIO client
        client = await get_user_minio_client()
        
        # Validate bucket exists
        if not bucket_name:
            return {"error": "bucket_name parameter is required. Use list_user_buckets() to discover available buckets."}
        
        if not client.bucket_exists(bucket_name):
            return {"error": f"Bucket '{bucket_name}' does not exist. Use list_user_buckets() to get a list of available buckets."}
        
        # Download pickle from MinIO
        try:
            stat = client.stat_object(bucket_name, object_path)
            response = client.get_object(bucket_name, object_path)
            pickle_bytes = response.read()
            response.close()
            response.release_conn()
        except S3Error as e:
            if "NoSuchKey" in str(e):
                # Auto-discover available files to help user
                discovered = _discover_files(client, bucket_name)
                return {
                    "error": f"File '{object_path}' not found in bucket '{bucket_name}'.",
                    "available_files": discovered if discovered else {},
                    "suggestion": f"Use one of these exact paths: {discovered.get('pickle', [])}" if discovered.get('pickle') else "No pickle files found in bucket",
                    "help": "Use list_user_buckets() to find available buckets and their data_files"
                }
            else:
                logger.error(f"Error downloading pickle from MinIO: {str(e)}")
                return {"error": f"Error downloading pickle: {str(e)}"}
        
        # Write to temporary file and load with pandas
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            temp_file.write(pickle_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Load the dataframe
            df = pd.read_pickle(temp_file_path)
            
            if not isinstance(df, pd.DataFrame):
                return {"error": "The pickle file does not contain a pandas DataFrame."}
            
            # Identify column groups
            gt_cols = [c for c in df.columns if c.startswith('GT_')]
            app_cols = [c for c in df.columns if c.startswith('APP_')]
            ai_cols = [c for c in df.columns if c.startswith('AI_')]
            op_cols = [c for c in df.columns if c.startswith('OP_')]
            other_cols = [c for c in df.columns if not any(c.startswith(p) for p in ['GT_', 'APP_', 'AI_', 'OP_'])]
            
            # Execute the requested query
            if query_type == 'overview':
                result = _smart_cities_overview(df, gt_cols, app_cols, ai_cols, op_cols, other_cols)
            elif query_type == 'error_distribution':
                result = _smart_cities_error_distribution(df)
            elif query_type == 'ai_decisions':
                result = _smart_cities_ai_decisions(df, ai_cols)
            elif query_type == 'operator_decisions':
                result = _smart_cities_operator_decisions(df, op_cols)
            elif query_type == 'processing_time':
                result = _smart_cities_processing_time(df, ai_cols, op_cols)
            elif query_type == 'field_errors':
                result = _smart_cities_field_errors(df, gt_cols, app_cols)
            elif query_type == 'decision_flow':
                result = _smart_cities_decision_flow(df, ai_cols, op_cols)
            elif query_type == 'confusion_matrix':
                result = _smart_cities_confusion_matrix(df, gt_cols, app_cols, ai_cols)
            elif query_type == 'ai_accuracy_metrics':
                result = _smart_cities_accuracy_metrics(df, gt_cols, app_cols, ai_cols)
            else:
                result = {"error": f"Query type '{query_type}' not implemented"}
            
            # Add metadata to result
            result["bucket"] = bucket_name
            result["object_path"] = object_path
            result["query_type"] = query_type
            result["total_records"] = len(df)
            result["user_context"] = {
                "pilot": pilot_context["pilot"],
                "default_bucket": pilot_context["default_bucket"]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing pickle data: {str(e)}")
            return {"error": f"Error analyzing data: {str(e)}"}
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Error deleting temporary file: {str(e)}")
                
    except Exception as e:
        logger.error(f"Unexpected error in analyze_smart_cities_data: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}


def _smart_cities_overview(df: pd.DataFrame, gt_cols: List, app_cols: List, ai_cols: List, op_cols: List, other_cols: List) -> Dict:
    """Generate overview statistics for the Smart Cities dataset."""
    result = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "column_groups": {
            "ground_truth_GT": {"count": len(gt_cols), "columns": gt_cols},
            "application_APP": {"count": len(app_cols), "columns": app_cols},
            "ai_processing_AI": {"count": len(ai_cols), "columns": ai_cols},
            "operator_OP": {"count": len(op_cols), "columns": op_cols},
            "other": {"count": len(other_cols), "columns": other_cols}
        }
    }
    
    # Add error type distribution if available
    if 'error_type' in df.columns:
        result["error_type_distribution"] = df['error_type'].value_counts().to_dict()
    
    # Add AI decision distribution if available
    ai_decision_col = next((c for c in df.columns if 'decision' in c.lower() and c.startswith('AI_')), None)
    if ai_decision_col:
        result["ai_decision_distribution"] = df[ai_decision_col].value_counts().to_dict()
    
    # Add operator decision distribution if available
    op_decision_col = next((c for c in df.columns if 'decision' in c.lower() and c.startswith('OP_')), None)
    if op_decision_col:
        result["operator_decision_distribution"] = df[op_decision_col].value_counts().to_dict()
    
    return result


def _smart_cities_error_distribution(df: pd.DataFrame) -> Dict:
    """Analyze error type distribution in applications."""
    result = {"analysis": "error_distribution"}
    
    # Look for error_type column (case-insensitive)
    error_col = next((c for c in df.columns if 'error_type' in c.lower()), None)
    
    if error_col:
        distribution = df[error_col].value_counts().to_dict()
        total = len(df)
        
        result["distribution"] = distribution
        result["percentages"] = {k: round(v / total * 100, 2) for k, v in distribution.items()}
        result["total_applications"] = total
        
        # Categorize by error severity
        no_errors = sum(v for k, v in distribution.items() if 'none' in str(k).lower() or 'no error' in str(k).lower())
        obvious_errors = sum(v for k, v in distribution.items() if 'obvious' in str(k).lower())
        actual_errors = sum(v for k, v in distribution.items() if 'actual' in str(k).lower() or 'real' in str(k).lower())
        other_errors = total - no_errors - obvious_errors - actual_errors
        
        result["summary"] = {
            "no_errors": no_errors,
            "obvious_errors": obvious_errors,
            "actual_errors": actual_errors,
            "other": other_errors
        }
    else:
        result["error"] = "No error_type column found in the dataset"
        result["available_columns"] = list(df.columns)
    
    return result


def _smart_cities_ai_decisions(df: pd.DataFrame, ai_cols: List) -> Dict:
    """Analyze AI decision distribution."""
    result = {"analysis": "ai_decisions"}
    
    # Look for AI decision column
    decision_col = next((c for c in ai_cols if 'decision' in c.lower()), None)
    
    if decision_col:
        distribution = df[decision_col].value_counts().to_dict()
        total = len(df)
        
        result["column_used"] = decision_col
        result["distribution"] = distribution
        result["percentages"] = {k: round(v / total * 100, 2) for k, v in distribution.items()}
        
        # Categorize decisions
        accepted = sum(v for k, v in distribution.items() if 'accept' in str(k).lower())
        rejected = sum(v for k, v in distribution.items() if 'reject' in str(k).lower())
        flagged = sum(v for k, v in distribution.items() if 'flag' in str(k).lower() or 'verif' in str(k).lower())
        
        result["summary"] = {
            "accepted": accepted,
            "rejected": rejected,
            "flagged_for_verification": flagged,
            "acceptance_rate": round(accepted / total * 100, 2) if total > 0 else 0
        }
        
        # Add confidence statistics if available
        confidence_col = next((c for c in ai_cols if 'confidence' in c.lower()), None)
        if confidence_col and pd.api.types.is_numeric_dtype(df[confidence_col]):
            result["confidence_stats"] = {
                "mean": round(df[confidence_col].mean(), 4),
                "median": round(df[confidence_col].median(), 4),
                "min": round(df[confidence_col].min(), 4),
                "max": round(df[confidence_col].max(), 4)
            }
    else:
        result["error"] = "No AI decision column found"
        result["available_ai_columns"] = ai_cols
    
    return result


def _smart_cities_operator_decisions(df: pd.DataFrame, op_cols: List) -> Dict:
    """Analyze operator decision distribution."""
    result = {"analysis": "operator_decisions"}
    
    # Look for operator decision column
    decision_col = next((c for c in op_cols if 'decision' in c.lower()), None)
    
    if decision_col:
        distribution = df[decision_col].value_counts().to_dict()
        total = len(df)
        
        result["column_used"] = decision_col
        result["distribution"] = distribution
        result["percentages"] = {k: round(v / total * 100, 2) for k, v in distribution.items()}
        
        # Categorize decisions
        accepted = sum(v for k, v in distribution.items() if 'accept' in str(k).lower() and 'fix' not in str(k).lower() and 'verif' not in str(k).lower())
        rejected = sum(v for k, v in distribution.items() if 'reject' in str(k).lower())
        fixed_accepted = sum(v for k, v in distribution.items() if 'fix' in str(k).lower())
        verified_accepted = sum(v for k, v in distribution.items() if 'verif' in str(k).lower())
        
        result["summary"] = {
            "directly_accepted": accepted,
            "rejected": rejected,
            "fixed_and_accepted": fixed_accepted,
            "accepted_after_verification": verified_accepted,
            "final_acceptance_rate": round((accepted + fixed_accepted + verified_accepted) / total * 100, 2) if total > 0 else 0
        }
    else:
        result["error"] = "No operator decision column found"
        result["available_op_columns"] = op_cols
    
    return result


def _smart_cities_processing_time(df: pd.DataFrame, ai_cols: List, op_cols: List) -> Dict:
    """Analyze processing time statistics with per-application averages."""
    result = {"analysis": "processing_time"}
    
    total_applications = len(df)
    result["total_applications"] = total_applications
    
    # Use exact column names for processing time analysis
    ai_time_col = 'AI_PROCESSING_TIME' if 'AI_PROCESSING_TIME' in ai_cols else None
    op_time_col = 'OP_PROCESSING_TIME' if 'OP_PROCESSING_TIME' in op_cols else None
    
    if ai_time_col and pd.api.types.is_numeric_dtype(df[ai_time_col]):
        # Drop NaN values for accurate calculations
        ai_times = df[ai_time_col].dropna()
        valid_count = len(ai_times)
        
        result["ai_processing"] = {
            "column": ai_time_col,
            "average_per_application": round(ai_times.mean(), 4) if valid_count > 0 else None,
            "median_per_application": round(ai_times.median(), 4) if valid_count > 0 else None,
            "std_deviation": round(ai_times.std(), 4) if valid_count > 0 else None,
            "min_time": round(ai_times.min(), 4) if valid_count > 0 else None,
            "max_time": round(ai_times.max(), 4) if valid_count > 0 else None,
            "total_time": round(ai_times.sum(), 4) if valid_count > 0 else None,
            "applications_with_data": valid_count,
            "applications_missing_data": total_applications - valid_count
        }
    else:
        result["ai_processing"] = {
            "error": "No AI processing time column found or column is not numeric",
            "available_ai_columns": ai_cols,
            "searched_patterns": TIME_PATTERNS
        }
    
    if op_time_col and pd.api.types.is_numeric_dtype(df[op_time_col]):
        # Drop NaN values for accurate calculations
        op_times = df[op_time_col].dropna()
        valid_count = len(op_times)
        
        result["operator_processing"] = {
            "column": op_time_col,
            "average_per_application": round(op_times.mean(), 4) if valid_count > 0 else None,
            "median_per_application": round(op_times.median(), 4) if valid_count > 0 else None,
            "std_deviation": round(op_times.std(), 4) if valid_count > 0 else None,
            "min_time": round(op_times.min(), 4) if valid_count > 0 else None,
            "max_time": round(op_times.max(), 4) if valid_count > 0 else None,
            "total_time": round(op_times.sum(), 4) if valid_count > 0 else None,
            "applications_with_data": valid_count,
            "applications_missing_data": total_applications - valid_count
        }
    else:
        result["operator_processing"] = {
            "error": "No operator processing time column found or column is not numeric",
            "available_op_columns": op_cols,
            "searched_patterns": TIME_PATTERNS
        }
    
    # Calculate efficiency comparison if both are available
    ai_proc = result.get("ai_processing", {})
    op_proc = result.get("operator_processing", {})
    
    if "error" not in ai_proc and "error" not in op_proc:
        ai_avg = ai_proc.get("average_per_application")
        op_avg = op_proc.get("average_per_application")
        
        if ai_avg is not None and op_avg is not None and op_avg > 0:
            result["comparison"] = {
                "ai_avg_per_application": ai_avg,
                "operator_avg_per_application": op_avg,
                "ai_vs_operator_ratio": round(ai_avg / op_avg, 4),
                "faster_by": "AI" if ai_avg < op_avg else "Operator",
                "time_difference_per_application": round(abs(ai_avg - op_avg), 4),
                "speedup_factor": round(op_avg / ai_avg, 2) if ai_avg > 0 else None
            }
    
    # Summary for easy reading
    result["summary"] = {
        "ai_average_time_per_application": ai_proc.get("average_per_application") if "error" not in ai_proc else "N/A",
        "operator_average_time_per_application": op_proc.get("average_per_application") if "error" not in op_proc else "N/A"
    }
    
    return result


def _smart_cities_field_errors(df: pd.DataFrame, gt_cols: List, app_cols: List) -> Dict:
    """Analyze which fields have most discrepancies between ground truth and application data."""
    result = {"analysis": "field_errors"}
    
    # Match GT_ columns with APP_ columns
    field_errors = {}
    matched_fields = []
    
    for gt_col in gt_cols:
        # Extract field name (remove GT_ prefix)
        field_name = gt_col[3:]  # Remove 'GT_'
        app_col = f"APP_{field_name}"
        
        if app_col in app_cols:
            matched_fields.append(field_name)
            # Count mismatches
            mismatches = (df[gt_col] != df[app_col]).sum()
            total = len(df)
            field_errors[field_name] = {
                "mismatches": int(mismatches),
                "total": total,
                "error_rate": round(mismatches / total * 100, 2) if total > 0 else 0
            }
    
    if field_errors:
        # Sort by error rate
        sorted_fields = sorted(field_errors.items(), key=lambda x: x[1]['error_rate'], reverse=True)
        
        result["field_error_rates"] = dict(sorted_fields)
        result["most_error_prone_field"] = sorted_fields[0][0] if sorted_fields else None
        result["least_error_prone_field"] = sorted_fields[-1][0] if sorted_fields else None
        result["matched_fields_count"] = len(matched_fields)
        result["average_error_rate"] = round(sum(f['error_rate'] for f in field_errors.values()) / len(field_errors), 2)
    else:
        result["error"] = "Could not match GT_ columns with APP_ columns"
        result["gt_columns"] = gt_cols
        result["app_columns"] = app_cols
    
    return result


def _smart_cities_decision_flow(df: pd.DataFrame, ai_cols: List, op_cols: List) -> Dict:
    """Analyze the flow from AI decision to final operator outcome."""
    result = {"analysis": "decision_flow"}
    
    # Find decision columns
    ai_decision_col = next((c for c in ai_cols if 'decision' in c.lower()), None)
    op_decision_col = next((c for c in op_cols if 'decision' in c.lower()), None)
    
    if ai_decision_col and op_decision_col:
        # Create cross-tabulation
        crosstab = pd.crosstab(df[ai_decision_col], df[op_decision_col])
        
        result["ai_column"] = ai_decision_col
        result["operator_column"] = op_decision_col
        result["flow_matrix"] = crosstab.to_dict()
        
        # Calculate flow statistics
        flows = []
        for ai_dec in crosstab.index:
            for op_dec in crosstab.columns:
                count = crosstab.loc[ai_dec, op_dec]
                if count > 0:
                    flows.append({
                        "from_ai": str(ai_dec),
                        "to_operator": str(op_dec),
                        "count": int(count),
                        "percentage": round(count / len(df) * 100, 2)
                    })
        
        # Sort by count
        flows.sort(key=lambda x: x['count'], reverse=True)
        result["flow_details"] = flows
        
        # Calculate agreement rate (AI decision matches operator decision concept)
        # This is a simplified check - you may need to adjust based on actual decision values
        agreement_count = 0
        for _, row in df.iterrows():
            ai_dec = str(row[ai_decision_col]).lower()
            op_dec = str(row[op_decision_col]).lower()
            # Check if both accepted or both rejected
            if ('accept' in ai_dec and 'accept' in op_dec) or ('reject' in ai_dec and 'reject' in op_dec):
                agreement_count += 1
        
        result["ai_operator_agreement_rate"] = round(agreement_count / len(df) * 100, 2) if len(df) > 0 else 0
    else:
        result["error"] = "Could not find both AI and operator decision columns"
        result["ai_columns"] = ai_cols
        result["op_columns"] = op_cols
    
    return result


def _smart_cities_confusion_matrix(df: pd.DataFrame, gt_cols: List, app_cols: List, ai_cols: List) -> Dict:
    """
    Calculate confusion matrix for AI decisions based on GT vs APP field comparison.
    
    Logic:
    - Ground truth: If ANY GT_* field != corresponding APP_* field -> should be REJECTED
    - If ALL GT_* fields == corresponding APP_* fields -> should be ACCEPTED
    - Compare AI decision with this ground truth to get TP/FP/TN/FN
    """
    result = {"analysis": "confusion_matrix"}
    
    # Find AI decision column
    ai_decision_col = next((c for c in ai_cols if 'decision' in c.lower()), None)
    
    if not ai_decision_col:
        result["error"] = "No AI decision column found"
        result["available_ai_columns"] = ai_cols
        return result
    
    # Match GT_ columns with APP_ columns
    matched_fields = []
    for gt_col in gt_cols:
        field_name = gt_col[3:]  # Remove 'GT_' prefix
        app_col = f"APP_{field_name}"
        if app_col in app_cols:
            matched_fields.append((gt_col, app_col, field_name))
    
    if not matched_fields:
        result["error"] = "Could not match any GT_ columns with APP_ columns"
        result["gt_columns"] = gt_cols
        result["app_columns"] = app_cols
        return result
    
    # Calculate ground truth for each row: should_be_rejected = any mismatch
    def should_be_rejected(row):
        for gt_col, app_col, _ in matched_fields:
            if row[gt_col] != row[app_col]:
                return True
        return False
    
    df_analysis = df.copy()
    df_analysis['_should_reject'] = df_analysis.apply(should_be_rejected, axis=1)
    df_analysis['_ai_accepted'] = df_analysis[ai_decision_col].apply(
        lambda x: 'accept' in str(x).lower()
    )
    
    # Calculate confusion matrix
    # TP: AI accepted AND should be accepted (not rejected)
    # FP: AI accepted BUT should be rejected
    # TN: AI rejected AND should be rejected
    # FN: AI rejected BUT should be accepted (not rejected)
    
    tp = len(df_analysis[(df_analysis['_ai_accepted'] == True) & (df_analysis['_should_reject'] == False)])
    fp = len(df_analysis[(df_analysis['_ai_accepted'] == True) & (df_analysis['_should_reject'] == True)])
    tn = len(df_analysis[(df_analysis['_ai_accepted'] == False) & (df_analysis['_should_reject'] == True)])
    fn = len(df_analysis[(df_analysis['_ai_accepted'] == False) & (df_analysis['_should_reject'] == False)])
    
    total = len(df)
    
    result["confusion_matrix"] = {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn
    }
    
    result["matrix_table"] = {
        "headers": ["", "Should Accept (GT=APP)", "Should Reject (GT!=APP)"],
        "rows": [
            ["AI Accepted", tp, fp],
            ["AI Rejected", fn, tn]
        ]
    }
    
    # Calculate metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    result["metrics"] = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4)
    }
    
    result["fields_compared"] = [f[2] for f in matched_fields]
    result["ai_decision_column"] = ai_decision_col
    result["total_applications"] = total
    
    return result


def _smart_cities_accuracy_metrics(df: pd.DataFrame, gt_cols: List, app_cols: List, ai_cols: List) -> Dict:
    """
    Calculate detailed accuracy metrics: true positives, false positives, 
    true negatives, false negatives with per-field breakdown.
    
    An AI accepted that should be accepted is a true positive.
    An AI accepted that should be rejected is a false positive.
    An AI rejected that should be rejected is a true negative.
    An AI rejected that should be accepted is a false negative.
    """
    result = {"analysis": "ai_accuracy_metrics"}
    
    # First get the confusion matrix data
    cm_result = _smart_cities_confusion_matrix(df, gt_cols, app_cols, ai_cols)
    
    if "error" in cm_result:
        return cm_result
    
    # Copy main metrics
    result["confusion_matrix"] = cm_result["confusion_matrix"]
    result["metrics"] = cm_result["metrics"]
    result["fields_compared"] = cm_result["fields_compared"]
    result["ai_decision_column"] = cm_result["ai_decision_column"]
    result["total_applications"] = cm_result["total_applications"]
    
    # Add detailed breakdown
    tp = cm_result["confusion_matrix"]["true_positives"]
    fp = cm_result["confusion_matrix"]["false_positives"]
    tn = cm_result["confusion_matrix"]["true_negatives"]
    fn = cm_result["confusion_matrix"]["false_negatives"]
    total = cm_result["total_applications"]
    
    result["detailed_breakdown"] = {
        "true_positives": {
            "count": tp,
            "percentage": round(tp / total * 100, 2) if total > 0 else 0,
            "description": "AI correctly accepted applications with matching GT/APP data"
        },
        "false_positives": {
            "count": fp,
            "percentage": round(fp / total * 100, 2) if total > 0 else 0,
            "description": "AI incorrectly accepted applications with mismatched GT/APP data"
        },
        "true_negatives": {
            "count": tn,
            "percentage": round(tn / total * 100, 2) if total > 0 else 0,
            "description": "AI correctly rejected applications with mismatched GT/APP data"
        },
        "false_negatives": {
            "count": fn,
            "percentage": round(fn / total * 100, 2) if total > 0 else 0,
            "description": "AI incorrectly rejected applications with matching GT/APP data"
        }
    }
    
    # Per-field error analysis for false positives
    ai_decision_col = cm_result["ai_decision_column"]
    matched_fields = []
    for gt_col in gt_cols:
        field_name = gt_col[3:]
        app_col = f"APP_{field_name}"
        if app_col in app_cols:
            matched_fields.append((gt_col, app_col, field_name))
    
    # Find which fields caused false positives (AI accepted but should have rejected)
    df_fp = df[df[ai_decision_col].apply(lambda x: 'accept' in str(x).lower())]
    field_fp_counts = {}
    
    for gt_col, app_col, field_name in matched_fields:
        mismatch_count = (df_fp[gt_col] != df_fp[app_col]).sum()
        if mismatch_count > 0:
            field_fp_counts[field_name] = int(mismatch_count)
    
    if field_fp_counts:
        sorted_fields = sorted(field_fp_counts.items(), key=lambda x: x[1], reverse=True)
        result["false_positive_field_breakdown"] = dict(sorted_fields)
        result["most_problematic_field"] = sorted_fields[0][0] if sorted_fields else None
    
    return result


def _discover_files(client: Minio, bucket_name: str, file_extensions: List[str] = None) -> Dict[str, List[str]]:
    """
    Discover files in a bucket, optionally filtered by extension.
    Used for auto-discovery when specified paths fail.
    
    Args:
        client: MinIO client instance
        bucket_name: Name of the bucket to search
        file_extensions: List of extensions to filter (e.g., ['.pkl', '.json', '.pdf'])
                        If None, returns all files organized by type
    
    Returns:
        Dict with keys 'pickle', 'json', 'pdf', 'other' containing file paths
    """
    files_by_type = {
        "pickle": [],
        "json": [],
        "pdf": [],
        "other": []
    }
    try:
        objects = client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            name = obj.object_name.lower()
            if name.endswith(('.pkl', '.pickle')):
                files_by_type["pickle"].append(obj.object_name)
            elif name.endswith('.json'):
                files_by_type["json"].append(obj.object_name)
            elif name.endswith('.pdf'):
                files_by_type["pdf"].append(obj.object_name)
            else:
                files_by_type["other"].append(obj.object_name)
    except Exception as e:
        logger.warning(f"Error listing objects for auto-discovery: {e}")
    
    # Filter by requested extensions if specified
    if file_extensions:
        filtered = []
        for ext in file_extensions:
            ext_lower = ext.lower().lstrip('.')
            if ext_lower in ['pkl', 'pickle']:
                filtered.extend(files_by_type["pickle"])
            elif ext_lower == 'json':
                filtered.extend(files_by_type["json"])
            elif ext_lower == 'pdf':
                filtered.extend(files_by_type["pdf"])
        return {"filtered_files": filtered}
    
    # Return non-empty categories only
    return {k: v for k, v in files_by_type.items() if v}


@cl.step(type="tool", name="Smart Cities File Comparison", show_input=False)
async def compare_smart_cities_files(
    bucket_name: str,
    object_paths: List[str],
    comparison_type: str = "full_summary"
) -> Dict:
    """
    Compare Smart Cities pilot data across multiple pickle files.
    
    Args:
        bucket_name: Name of the MinIO bucket containing the pickle files
        object_paths: List of paths to pickle files to compare
        comparison_type: Type of comparison to perform:
            - 'decisions': Compare AI and operator decision distributions
            - 'processing_time': Compare processing times across files
            - 'full_summary': Complete comparison with all metrics (default)
        
    Returns:
        Dict containing comparison results across all files
    """
    try:
        # Check user access using dynamic pilot context
        pilot_context = _get_user_pilot_context()
        if pilot_context["pilot"] != "smart_cities" or not pilot_context["has_access"]:
            logger.warning(f"Access denied for user {pilot_context['user_email']} to Smart Cities comparison tool")
            return {
                "error": "Access denied. This tool is only available to Smart Cities pilot users.",
                "user_email": pilot_context["user_email"],
                "detected_pilot": pilot_context["pilot"]
            }

        # Validate inputs
        if not object_paths or len(object_paths) < 1:
            return {"error": "At least one object_path is required for comparison."}
        
        valid_comparison_types = ['decisions', 'processing_time', 'full_summary']
        if comparison_type not in valid_comparison_types:
            return {"error": f"Invalid comparison_type '{comparison_type}'. Valid options: {valid_comparison_types}"}
        
        # Get MinIO client
        client = await get_user_minio_client()
        
        # Validate bucket exists
        if not bucket_name:
            return {"error": "bucket_name parameter is required. Use list_user_buckets() to discover available buckets."}
        
        if not client.bucket_exists(bucket_name):
            return {"error": f"Bucket '{bucket_name}' does not exist. Use list_user_buckets() to get a list of available buckets."}
        
        # Load and analyze each file
        file_results = []
        errors = []
        
        for object_path in object_paths:
            try:
                # Validate pickle file extension
                if not object_path.lower().endswith('.pkl') and not object_path.lower().endswith('.pickle'):
                    errors.append({"file": object_path, "error": "Not a pickle file"})
                    continue
                
                # Download pickle from MinIO
                try:
                    response = client.get_object(bucket_name, object_path)
                    pickle_bytes = response.read()
                    response.close()
                    response.release_conn()
                except S3Error as e:
                    errors.append({
                        "file": object_path, 
                        "error": f"File not found or access error: {str(e)}",
                        "suggestion": f"Use get_minio_info(bucket_name='{bucket_name}') to list available files"
                    })
                    continue
                
                # Write to temporary file and load with pandas
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                    temp_file.write(pickle_bytes)
                    temp_file_path = temp_file.name
                
                try:
                    df = pd.read_pickle(temp_file_path)
                    
                    if not isinstance(df, pd.DataFrame):
                        errors.append({"file": object_path, "error": "File does not contain a pandas DataFrame"})
                        continue
                    
                    # Extract metrics from this file
                    file_data = _extract_file_metrics(df, object_path)
                    file_results.append(file_data)
                    
                finally:
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                        
            except Exception as e:
                errors.append({"file": object_path, "error": str(e)})
        
        if not file_results:
            # Auto-discover available files to help user
            discovered = _discover_files(client, bucket_name)
            return {
                "error": "Could not load any pickle files successfully.",
                "file_errors": errors,
                "available_files": discovered if discovered else {},
                "suggestion": f"Try using these exact paths: {discovered.get('pickle', [])}" if discovered.get('pickle') else "No pickle files found in bucket",
                "help": "Use list_user_buckets() to find available buckets and their data_files"
            }
        
        # Build comparison result based on comparison_type
        result = {
            "comparison_type": comparison_type,
            "bucket": bucket_name,
            "files_analyzed": len(file_results),
            "files_requested": len(object_paths)
        }
        
        if errors:
            result["file_errors"] = errors
        
        if comparison_type == 'decisions' or comparison_type == 'full_summary':
            result["decisions_comparison"] = _build_decisions_comparison(file_results)
        
        if comparison_type == 'processing_time' or comparison_type == 'full_summary':
            result["processing_time_comparison"] = _build_processing_time_comparison(file_results)
        
        if comparison_type == 'full_summary':
            result["summary_table"] = _build_summary_table(file_results)
        
        # Add user context
        result["user_context"] = {
            "pilot": pilot_context["pilot"],
            "default_bucket": pilot_context["default_bucket"]
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error in compare_smart_cities_files: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}


def _extract_file_metrics(df: pd.DataFrame, object_path: str) -> Dict:
    """Extract key metrics from a single DataFrame for comparison."""
    file_name = object_path.split('/')[-1]
    total_rows = len(df)
    
    # Identify column groups
    gt_cols = [c for c in df.columns if c.startswith('GT_')]
    app_cols = [c for c in df.columns if c.startswith('APP_')]
    ai_cols = [c for c in df.columns if c.startswith('AI_')]
    op_cols = [c for c in df.columns if c.startswith('OP_')]
    
    result = {
        "file": file_name,
        "object_path": object_path,
        "total_rows": total_rows
    }
    
    # AI decision metrics
    ai_decision_col = next((c for c in ai_cols if 'decision' in c.lower()), None)
    if ai_decision_col:
        ai_decisions = df[ai_decision_col].value_counts().to_dict()
        result["ai_decisions"] = ai_decisions
        
        # Count AI decisions that lead to operator intervention (flagged/verification)
        flagged = sum(v for k, v in ai_decisions.items() 
                     if 'flag' in str(k).lower() or 'verif' in str(k).lower())
        result["ai_flagged_count"] = flagged
        result["ai_flagged_rate"] = round(flagged / total_rows, 4) if total_rows > 0 else 0
    
    # Operator decision metrics
    op_decision_col = next((c for c in op_cols if 'decision' in c.lower()), None)
    if op_decision_col:
        op_decisions = df[op_decision_col].value_counts().to_dict()
        result["operator_decisions"] = op_decisions
        
        # Count operator interventions (any decision that's not direct accept)
        interventions = sum(v for k, v in op_decisions.items() 
                          if 'fix' in str(k).lower() or 'reject' in str(k).lower() or 'verif' in str(k).lower())
        result["operator_interventions"] = interventions
        result["operator_intervention_rate"] = round(interventions / total_rows, 4) if total_rows > 0 else 0
    
    # Processing time metrics using broadened patterns
    ai_time_col = next((c for c in ai_cols if any(p in c.lower() for p in TIME_PATTERNS)), None)
    if ai_time_col and pd.api.types.is_numeric_dtype(df[ai_time_col]):
        ai_times = df[ai_time_col].dropna()
        result["ai_avg_time_per_app"] = round(ai_times.mean(), 4) if len(ai_times) > 0 else None
        result["ai_total_time"] = round(ai_times.sum(), 4) if len(ai_times) > 0 else None
        result["ai_time_column_found"] = ai_time_col
    
    op_time_col = next((c for c in op_cols if any(p in c.lower() for p in TIME_PATTERNS)), None)
    if op_time_col and pd.api.types.is_numeric_dtype(df[op_time_col]):
        op_times = df[op_time_col].dropna()
        result["op_avg_time_per_app"] = round(op_times.mean(), 4) if len(op_times) > 0 else None
        result["op_total_time"] = round(op_times.sum(), 4) if len(op_times) > 0 else None
        result["op_time_column_found"] = op_time_col
    
    return result


def _build_decisions_comparison(file_results: List[Dict]) -> Dict:
    """Build decisions comparison across files."""
    comparison = {
        "files": []
    }
    
    for fr in file_results:
        file_data = {
            "file": fr["file"],
            "total_rows": fr["total_rows"],
            "ai_decisions": fr.get("ai_decisions", {}),
            "operator_decisions": fr.get("operator_decisions", {}),
            "ai_flagged_rate": fr.get("ai_flagged_rate"),
            "operator_intervention_rate": fr.get("operator_intervention_rate")
        }
        comparison["files"].append(file_data)
    
    # Calculate aggregate stats
    if len(file_results) > 1:
        avg_ai_flagged = sum(fr.get("ai_flagged_rate", 0) for fr in file_results) / len(file_results)
        avg_op_intervention = sum(fr.get("operator_intervention_rate", 0) for fr in file_results) / len(file_results)
        comparison["aggregate"] = {
            "average_ai_flagged_rate": round(avg_ai_flagged, 4),
            "average_operator_intervention_rate": round(avg_op_intervention, 4)
        }
    
    return comparison


def _build_processing_time_comparison(file_results: List[Dict]) -> Dict:
    """Build processing time comparison across files."""
    comparison = {
        "files": []
    }
    
    for fr in file_results:
        file_data = {
            "file": fr["file"],
            "total_rows": fr["total_rows"],
            "ai_avg_time_per_app": fr.get("ai_avg_time_per_app"),
            "op_avg_time_per_app": fr.get("op_avg_time_per_app"),
            "ai_total_time": fr.get("ai_total_time"),
            "op_total_time": fr.get("op_total_time")
        }
        comparison["files"].append(file_data)
    
    # Calculate aggregate stats
    if len(file_results) > 1:
        ai_times = [fr.get("ai_avg_time_per_app") for fr in file_results if fr.get("ai_avg_time_per_app") is not None]
        op_times = [fr.get("op_avg_time_per_app") for fr in file_results if fr.get("op_avg_time_per_app") is not None]
        
        comparison["aggregate"] = {}
        if ai_times:
            comparison["aggregate"]["average_ai_time_across_files"] = round(sum(ai_times) / len(ai_times), 4)
        if op_times:
            comparison["aggregate"]["average_op_time_across_files"] = round(sum(op_times) / len(op_times), 4)
    
    return comparison


def _build_summary_table(file_results: List[Dict]) -> List[Dict]:
    """Build a summary table for all files."""
    table = []
    
    for fr in file_results:
        row = {
            "file": fr["file"],
            "total_rows": fr["total_rows"],
            "avg_ai_time_per_app": fr.get("ai_avg_time_per_app"),
            "avg_op_time_per_app": fr.get("op_avg_time_per_app"),
            "operator_interventions": fr.get("operator_interventions"),
            "intervention_rate": fr.get("operator_intervention_rate")
        }
        table.append(row)
    
    return table


function_map = {
    "get_docs": get_docs,
    "get_minio_info": get_minio_info,
    "get_kf_pipelines": get_kf_pipelines,
    "get_pipeline_details": get_pipeline_details,
    "get_pipeline_version_details": get_pipeline_version_details,
    "run_pipeline": run_pipeline,
    "list_runs": list_runs,
    "get_run_details": get_run_details,
    "get_pipeline_artifacts_from_MinIO": get_pipeline_artifacts,
    "get_model_metrics": get_model_metrics,
    "get_pipeline_visualization": get_pipeline_visualization,
    "compare_pipeline_runs": compare_pipeline_runs,
    "list_user_buckets": list_user_buckets,
    "list_experiments": list_experiments,
    "get_experiment_details": get_experiment_details,
    "get_user_kubeflow_namespace": get_user_namespace,
    "create_experiment": create_experiment,
    "get_pipeline_id": get_pipeline_id,
    "parse_pdf_from_minio": parse_pdf_from_minio,
    "plot_data": plot_data,
    "analyze_smart_cities_data": analyze_smart_cities_data,
    "compare_smart_cities_files": compare_smart_cities_files
}