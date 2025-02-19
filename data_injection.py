import logging, os, argparse
import pandas as pd
from utils.helper_functions import setup_logging
from datetime import datetime, timedelta
from typing import List
import time
from pinecone import ServerlessSpec, Pinecone
from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

logger = setup_logging('Data Injection', level=logging.INFO)

EMBED_MODEL = "text-embedding-3-small"
INDEX_NAME = 'humaine'
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
embedding_model = OpenAIEmbeddings(model=EMBED_MODEL)

def process_data():
    """
    Main function to process data through the pipeline
    """
    logger.info("Starting data processing")
    
    # Get and process reports
    df = get_deliverables()
    if df.empty:
        logger.warning("No data to process")
        return
    
    # Save to Pinecone
    save_to_pinecone(df)
    logger.info("Data processing completed successfully")

def get_embeddings(text_list) -> List[List[float]]:
    try:
        embeddings = embedding_model.embed_documents(text_list)
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return []

def get_deliverables():
    """
    Load and process documents from local ./docs directory
    """
    try:
        docs_dir = '.docs'
        if not os.path.exists(docs_dir):
            logger.error(f"Directory {docs_dir} does not exist")
            return pd.DataFrame()

        data = []
        # Walk through all files in the docs directory
        for root, _, files in os.walk(docs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    from langchain_community.document_loaders import PyPDFLoader

                    loader = PyPDFLoader(file_path)
                    pages = loader.load()  # Load all pages at once
                    content = " ".join([page.page_content for page in pages])  # Use page_content instead of extract_text
                    
                    # Use filename as title (without extension)
                    title = os.path.splitext(file)[0]
                    
                    data.append({'title': title, 'text': content})
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    continue
        
        if not data:
            logger.warning("No documents found in ./docs directory")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Clean title to ensure ASCII compatibility
        df['id'] = df['title'].apply(lambda x: ''.join(char for char in x if ord(char) < 128))
        df['id'] = df['id'].str.replace('[^a-zA-Z0-9-]', '_', regex=True)  # Replace non-alphanumeric chars with underscore

        # Chunk the documents
        chunked_data = chunk_documents(df)
        if not chunked_data:
            return pd.DataFrame()

        chunked_df = pd.DataFrame(chunked_data)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = get_embeddings(chunked_df['chunk_text'].to_list())
        chunked_df['embeddings'] = embeddings

        return chunked_df

    except Exception as e:
        logger.error(f"Error in get_reports: {str(e)}")
        return pd.DataFrame()

def chunk_documents(df):
    """
    Chunk documents using semantic chunking
    """
    chunked_data = []
    text_splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="gradient",
        min_chunk_size=100,
        breakpoint_threshold_amount=0.8

    )

    for idx, row in df.iterrows():
        text = row['text']
        if not text:
            continue
        
        try:
            docs = text_splitter.create_documents([text])
            
            for i, doc in enumerate(docs):
                # Create ASCII-safe chunk ID
                chunk_id = f"{row['id']}_chunk_{i}".replace(' ', '_')
                chunked_data.append({
                    'id': chunk_id,
                    'chunk_text': doc.page_content,
                    'title': row['title']
                })
        except Exception as e:
            logger.error(f"Error chunking document {row['id']}: {str(e)}")
            continue
    
    return chunked_data

def save_to_pinecone(df):
    """
    Save processed data to Pinecone
    """
    try:
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        check_and_create_index(INDEX_NAME, len(df['embeddings'][0]), spec)

        index = pc.Index(INDEX_NAME)
        time.sleep(1)
        logger.info(index.describe_index_stats())

        batch_size = 32
        for i in tqdm(range(0, len(df), batch_size)):
            i_end = min(i+batch_size, len(df))
            ids_batch = df['id'][i: i_end]
            
            # Include title and chunk text in metadata
            metadata = [
                {
                    "title": df.loc[j, "title"],
                    "text": df.loc[j, "chunk_text"]
                }
                for j in range(i, i_end)
            ]
            
            to_upsert = zip(ids_batch, df['embeddings'][i:i_end], metadata)
            index.upsert(vectors=list(to_upsert))
            
    except Exception as e:
        logger.error(f"Error saving to Pinecone: {str(e)}")

def check_and_create_index(index_name, dimension, spec):
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                index_name,
                dimension=dimension,
                metric='dotproduct',
                spec=spec
            )
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Injection Script")
    args = parser.parse_args()
    
    process_data()
   