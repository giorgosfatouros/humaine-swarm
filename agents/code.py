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
import logging
import requests
import kfp
from typing import Dict, List, Optional

# Set up the LLM
Settings.llm  = OpenAI(model="gpt-4o-mini", max_tokens=4096, temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
logger = setup_logging('TOOLS', level=logging.INFO)
client = AsyncOpenAI()
# lai = AsyncLiteralClient()
# lai.instrument_openai()


# Set up the vector store
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
vector_store = PineconeVectorStore(pinecone_index=pc.Index("humaine"))
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Kubeflow connection settings
KUBEFLOW_HOST = 'http://hua-kubeflow.ddns.net/'
KUBEFLOW_USERNAME = "user@example.com"
KUBEFLOW_PASSWORD = "2LZHseTdrLFFvx"
KUBEFLOW_NAMESPACE = "kubeflow-user-example-com"

def get_kubeflow_client() -> kfp.Client:
    """Initialize and return a Kubeflow Pipelines client."""
    session = requests.Session()
    response = session.get(KUBEFLOW_HOST)
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    data = {"login": KUBEFLOW_USERNAME, "password": KUBEFLOW_PASSWORD}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()["authservice_session"]
    
    return kfp.Client(
        host=f"{KUBEFLOW_HOST}/pipeline",
        cookies=f"authservice_session={session_cookie}",
        namespace=KUBEFLOW_NAMESPACE,
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


@cl.step(type="tool", name="HumAIne Insights", show_input=False)
async def get_humaine_info(query: str):
    optimized_query = await optimize_query(query)
    retriever_humaine = VectorIndexRetriever(index=index, similarity_top_k=10)
    retrieved_documents = await retriever_humaine.aretrieve(optimized_query)
    logger.info(f"Retrieved documents: {retrieved_documents}")
    return rag_extract_deliverables(retrieved_documents)

@cl.step(type="tool", name="Kubeflow Pipeline Info", show_input=False)
async def get_kubeflow_info() -> Dict:
    """
    Retrieve information about all Kubeflow pipelines.
    
    Returns:
        Dict containing pipeline information
    """
    try:
        kf_client = get_kubeflow_client()
        pipelines = kf_client.list_pipelines().pipelines
        
        return {
            "pipelines": [{
                "name": p.display_name,
                "description": p.description,
                "created_at": str(p.created_at)
            } for p in (pipelines or [])]
        }
            
    except Exception as e:
        logger.error(f"Error retrieving Kubeflow information: {str(e)}")
        return {"error": str(e)}

function_map = {
    "get_humaine_info": get_humaine_info,
    "get_kubeflow_info": get_kubeflow_info
}