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
INDEX_NAME = 'humaine-test'
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
embedding_model = OpenAIEmbeddings(model=EMBED_MODEL)

def process_data():
    """
    Main function to process data through the pipeline
    """
    logger.info("Starting data processing")
    
    # Get and process reports
    df = get_reports()
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

def get_reports():
    """
    Load and process documents from local ./docs directory
    """
    try:
        docs_dir = './docs'
        if not os.path.exists(docs_dir):
            logger.error(f"Directory {docs_dir} does not exist")
            return pd.DataFrame()

        data = []
        # Walk through all files in the docs directory
        for root, _, files in os.walk(docs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Read the file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract organization from directory structure (optional)
                    org = os.path.basename(os.path.dirname(file_path))
                    if org == 'docs':  # If file is directly in docs folder
                        org = 'Default'
                    
                    # Use filename as title (without extension)
                    title = os.path.splitext(file)[0]
                    
                    data.append({
                        'Organization': org,
                        'Title': title,
                        'cleaned_text': content,
                        'Link': file_path,  # Using file path as link for reference
                        'summary': ''  # Empty summary by default
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    continue
        
        if not data:
            logger.warning("No documents found in ./docs directory")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Clean and process text
        df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x.replace('\n', ' '))
        
        # Generate unique IDs using Title and Organization
        df['id'] = df['Organization'].str.replace(' ', '') + '_' + df['Title'].str.replace(' ', '').str.encode('ascii', 'ignore').str.decode('ascii')

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
        breakpoint_threshold_type="gradient"
    )

    for idx, row in df.iterrows():
        text = row['cleaned_text']
        if not text:
            continue
        
        try:
            docs = text_splitter.create_documents([text])
            
            for i, doc in enumerate(docs):
                chunk_id = f"{row['id']}_chunk_{i}"
                chunked_data.append({
                    'id': chunk_id,
                    'chunk_text': doc.page_content,
                    'Organization': row['Organization'],
                    'Title': row['Title'],
                    'Link': row['Link'],
                    'summary': row.get('summary', ''),
                    'source_id': row['id']
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
            
            metadata = [
                {
                    "Organization": df.loc[j, "Organization"],
                    "Title": df.loc[j, "Title"],
                    "Link": df.loc[j, "Link"],
                    "summary": df.loc[j, "summary"],
                    "text": df.loc[j, "chunk_text"],
                    "source_id": df.loc[j, "source_id"]
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
   