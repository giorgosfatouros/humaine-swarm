import os
from datetime import datetime
import json
import chainlit as cl
from literalai import AsyncLiteralClient
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from pinecone.grpc import PineconeGRPC as Pinecone
from llama_index.core import Settings
from llama_index.llms.openai import AsyncOpenAI, OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from utils.helper_functions import setup_logging, rag_extract_deliverables, get_minio_client
from utils.config import *  # Import configuration settings
import logging
import requests
import kfp
from typing import Dict, List, Optional, Any
from minio import Minio
from minio.error import S3Error
from utils.helper_functions import get_kubeflow_client

# Set up the LLM
Settings.llm  = OpenAI(model=LLM_MODEL, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE)
Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
logger = setup_logging('TOOLS', level=logging.INFO)
client = AsyncOpenAI()
lai = AsyncLiteralClient()
lai.instrument_openai()


# Set up the vector store
pc = Pinecone(api_key=PINECONE_API_KEY)
vector_store = PineconeVectorStore(pinecone_index=pc.Index(PINECONE_INDEX))
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)




def read_prompt(type: str):
    current_date = datetime.now().strftime("%Y-%m-%d")

    if type == "system":
        with open('prompts/system.md', 'r') as file:
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
async def get_minio_info(prefix: Optional[str] = None, max_items: int = 10) -> Dict:
    """
    Retrieve information about objects in a specific MinIO bucket.
    
    Args:
        prefix: Optional prefix to filter objects (like a folder path)
        max_items: Maximum number of items to return (default: 10)
        
    Returns:
        Dict containing object information from the specified bucket
    """
    try:
        client = get_minio_client()
        bucket_name = os.getenv('MINIO_BUCKET')
        
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

@cl.step(type="tool", name="List User Buckets", show_input=False)
async def list_user_buckets(max_buckets: int = 20) -> Dict:
    """
    List all MinIO buckets available to the user with basic statistics.
    
    Args:
        max_buckets: Maximum number of buckets to return (default: 20)
        
    Returns:
        Dict containing information about available buckets
    """
    try:
        client = get_minio_client()
        
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
    pipeline_name: Optional[str] = None,
    run_name: Optional[str] = None,
    artifact_type: Optional[str] = None,
    max_items: int = 20,
    object_path: Optional[str] = None
) -> Dict:
    """
    Retrieve ML pipeline artifacts from MinIO storage.
    
    Args:
        pipeline_name: Name of the pipeline to query artifacts for (e.g. 'diabetes-svm-classification')
        run_name: Optional specific run name to filter artifacts
        artifact_type: Optional artifact type to filter (e.g. 'models', 'metrics', 'plots')
        max_items: Maximum number of items to return
        object_path: Optional direct MinIO object path to retrieve (overrides other path parameters)
        
    Returns:
        Dict containing artifact information
    """
    try:
        client = get_minio_client()
        bucket_name = os.getenv('MINIO_BUCKET')
        
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
        pipeline_name: Name of the pipeline (e.g. 'diabetes-svm-classification')
        run_name: Optional specific run name to get metrics for
        model_name: Optional model name to filter metrics (e.g. 'Support Vector Machine')
        max_items: Maximum number of items to return (default: 20)
        metrics_path: Optional direct path to metrics JSON file (overrides other parameters)
        
    Returns:
        Dict containing model metrics
    """
    try:
        client = get_minio_client()
        bucket_name = os.getenv('MINIO_BUCKET')
        
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
    pipeline_name: Optional[str] = None,
    run_name: Optional[str] = None,
    visualization_type: Optional[str] = None,
    model_name: Optional[str] = None,
    visualization_path: Optional[str] = None
) -> Dict:
    """
    Retrieve and return visualization artifacts (HTML plots) from pipeline runs.
    
    Args:
        pipeline_name: Name of the pipeline (e.g. 'diabetes-svm-classification')
        run_name: Run name to get visualizations for
        visualization_type: Type of visualization to retrieve ('confusion_matrix', 'roc_curve', 'feature_importance')
        model_name: Optional model name part to filter by (e.g. 'svm')
        visualization_path: Optional direct path to a visualization HTML file (overrides other parameters)
        
    Returns:
        Dict containing HTML visualization content that can be displayed in the chat
    """
    try:
        client = get_minio_client()
        bucket_name = os.getenv('MINIO_BUCKET')
        
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
    pipeline_name: str,
    run_names: List[str],
    metric_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compare multiple pipeline runs based on their metrics.
    
    Args:
        pipeline_name: Name of the pipeline (e.g. 'diabetes-svm-classification')
        run_names: List of run names to compare
        metric_names: Optional list of specific metrics to compare (e.g. ['accuracy', 'precision'])
        
    Returns:
        Dict containing comparison results
    """
    try:
        client = get_minio_client()
        bucket_name = os.getenv('MINIO_BUCKET')
        # Check if bucket exists

        try:
            client.bucket_exists(bucket_name)
        except Exception as e:
            return {"error": f"Bucket '{bucket_name}' does not exist use the list_user_buckets function to get a list of the user's buckets"}
        
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
        kf_client = get_kubeflow_client()

        namespace = os.getenv('KUBEFLOW_NAMESPACE')
        
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
            
        kf_client = get_kubeflow_client()
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
            
        kf_client = get_kubeflow_client()
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
        kf_client = get_kubeflow_client()
        
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
        kf_client = get_kubeflow_client()
        
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
            
        kf_client = get_kubeflow_client()
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
        kf_client = get_kubeflow_client()
        
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
            
        kf_client = get_kubeflow_client()
        
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
        kf_client = get_kubeflow_client()
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
            
        kf_client = get_kubeflow_client()
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
            
        kf_client = get_kubeflow_client()
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
    "get_pipeline_id": get_pipeline_id
}