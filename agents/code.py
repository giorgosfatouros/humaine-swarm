import requests
import json
import sqlite3
import os
from datetime import datetime
import chainlit as cl

from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
# from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from pinecone.grpc import PineconeGRPC as Pinecone
from llama_index.core import Settings
from llama_index.llms.openai import AsyncOpenAI, OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo


# Set up the LLM
Settings.llm  = OpenAI(model="gpt-4o-mini", max_tokens=4096, temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
# hyde = HyDEQueryTransform(include_original=True)

# for rag 
vector_store = PineconeVectorStore(pinecone_index=pc.Index("sec-filings"))
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever_filings = VectorIndexRetriever(index=index, similarity_top_k=15)

function_map = {
    # "query_router": query_router,
    # "classify_mri": classify_mri,
    # "analyze_exam": analyze_exam,
    "retrieve_pubmed_research": retrieve_pubmed_articles,
    # "generate_follow_up_questions": generate_follow_up_questions
  

}