import json
import os
from datetime import datetime
import chainlit as cl
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from pinecone.grpc import PineconeGRPC as Pinecone
from llama_index.core import Settings
from llama_index.llms.openai import AsyncOpenAI, OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from utils.helper_functions import setup_logging, rag_extract_deliverables
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
# lai = AsyncLiteralClient()
# lai.instrument_openai()


# Set up the vector store
pc = Pinecone(api_key=PINECONE_API_KEY)
vector_store = PineconeVectorStore(pinecone_index=pc.Index(PINECONE_INDEX))
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

def get_minio_client() -> Minio:
    """Initialize and return a MinIO client."""
    return Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        session_token=os.getenv("MINIO_SESSION_TOKEN"),
        secure=MINIO_SECURE
    )



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


@cl.step(type="tool", name="Documentation", show_input=False)
async def get_docs(query: str):
    # optimized_query = await optimize_query(query)
    retriever_humaine = VectorIndexRetriever(index=index, similarity_top_k=10)
    retrieved_documents = await retriever_humaine.aretrieve(query)
    logger.info(f"Retrieved documents: {retrieved_documents}")
    return rag_extract_deliverables(retrieved_documents)

@cl.step(type="tool", name="Kubeflow Pipelines", show_input=False)
async def get_kf_pipelines(
    search_term: Optional[str] = None, 
    namespace: Optional[str] = None, 
    page_size: int = 20, 
    page_token: str = "",
    sort_by: str = "created_at desc"
) -> Dict:
    """
    Retrieve information about Kubeflow pipelines with flexible search capabilities.
    
    Args:
        search_term: Optional term to search for in pipeline names (case-insensitive partial match)
        namespace: Optional namespace to filter pipelines (None returns shared pipelines)
        page_size: Number of results to return per page (default: 20)
        page_token: Token for pagination (default: empty string for first page)
        sort_by: How to sort results (default: created_at desc - newest first)
        
    Returns:
        Dict containing filtered pipeline information
    """
    try:
        kf_client = get_kubeflow_client()
        
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

@cl.step(type="tool", name="MinIO Bucket Info", show_input=False)
async def get_minio_info(bucket_name: Optional[str] = None, prefix: Optional[str] = None, max_items: int = 10) -> Dict:
    """
    Retrieve information about MinIO buckets and objects.
    
    Args:
        bucket_name: Optional name of the MinIO bucket to query. If None, lists all buckets.
        prefix: Optional prefix to filter objects (like a folder path)
        max_items: Maximum number of items to return (default: 10)
        
    Returns:
        Dict containing bucket information or object information from a specific bucket
    """
    try:
        client = get_minio_client()
        
        # If no bucket specified, list all buckets
        if bucket_name is None:
            buckets = client.list_buckets()
            bucket_list = [{
                "name": bucket.name,
                "creation_date": str(bucket.creation_date)
            } for bucket in buckets]
            
            return {
                "total_buckets": len(bucket_list),
                "buckets": bucket_list
            }
        
        # Check if specified bucket exists
        if not client.bucket_exists(bucket_name):
            return {"error": f"Bucket '{bucket_name}' does not exist"}
        
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
                "metrics": [
                    {"name": m.name, "value": m.number_value}
                    for m in r.metrics
                ] if hasattr(r, 'metrics') and r.metrics else []
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
            "metrics": [
                {"name": m.name, "value": m.number_value}
                for m in run.metrics
            ] if hasattr(run, 'metrics') and run.metrics else [],
            "error": {
                "code": run.error.code if hasattr(run.error, 'code') else None,
                "message": run.error.message if hasattr(run.error, 'message') else None
            } if hasattr(run, 'error') and run.error else None
        }
        
        return run_details
            
    except Exception as e:
        logger.error(f"Error retrieving Kubeflow run details: {str(e)}")
        return {"error": str(e)}

function_map = {
    "get_docs": get_docs,
    "get_minio_info": get_minio_info,
    "get_kf_pipelines": get_kf_pipelines,
    "get_pipeline_details": get_pipeline_details,
    "get_pipeline_version_details": get_pipeline_version_details,
    "run_pipeline": run_pipeline,
    "list_runs": list_runs,
    "get_run_details": get_run_details
}