import requests
import json
import sqlite3
from datetime import datetime
from langchain_community.retrievers import PubMedRetriever
import logging
import os
import requests
from bs4 import BeautifulSoup
from pinecone import Pinecone
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_openai import OpenAIEmbeddings
from typing import Dict, Any




# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),  # Log to a file
        logging.StreamHandler()  # Also log to the console
    ]
)

# Create a logger for your module
logger = logging.getLogger(__name__)


# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_REGION"))

# Check if the index exists; create if not
index_name = "humaine-test"

# Connect to the index
vectorstore = pc.Index(index_name)  # Use the correct method to connect

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)
def read_prompt(type: str):
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    if type == "research":
        with open('prompts/research_agent.md', 'r') as file:
            prompt = file.read()
    elif type == "mri":
        with open('prompts/mri_agent.md', 'r') as file:
            prompt = file.read()
    elif type == "exam":
        with open('prompts/exams_agent.md', 'r') as file:
            return file.read()
    else:
        with open('prompts/system.md', 'r') as file:
            prompt = file.read()
    
    update_info = f"\n\n*Note: Today's date is {current_date}.*\n\nWhen you receive function results, incorporate them into your responses to provide accurate and helpful information to the user."
    
    prompt += update_info
    return prompt

# This is the MRI classification function that the model can call
def classify_mri(image: bytes) -> dict:
    """
    Classify MRI images by sending them to an external model API for breast cancer diagnosis.
    Returns the classification result in JSON format.
    """
    # Example external API call to classify MRI images
    try:
        # Assuming the external service requires a POST request
        files = {'image': image}
        response = requests.post("https://example.com/classify_mri", files=files)
        
        # Parse the response
        if response.status_code == 200:
            classification_result = response.json()
            return {"status": "success", "result": classification_result}
        else:
            return {"status": "error", "message": f"Failed to classify MRI: {response.status_code}"}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


# This is the function to analyze a breast cancer exam PDF
def analyze_exam(pdf_url: str) -> dict:
    """
    Analyze the content of a breast cancer exam PDF by parsing the PDF and extracting useful data.
    """
    try:
        # Example API to analyze the PDF
        response = requests.post("https://example.com/analyze_pdf", json={"pdf_url": pdf_url})
        
        # Parse the response
        if response.status_code == 200:
            exam_analysis = response.json()
            return {"status": "success", "result": exam_analysis}
        else:
            return {"status": "error", "message": f"Failed to analyze PDF: {response.status_code}"}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

    


# Function to retrieve articles from PubMed
def retrieve_pubmed_articles(query: str, num_articles: int = 3) -> Dict[str, Any]:
    """
    Retrieve article metadata from PubMed.
    """
    if not query.strip():
        return {"status": "error", "message": "Query cannot be empty."}
    
    logger.debug(f"Starting retrieve_pubmed_articles with query: {query} and num_articles: {num_articles}")
    try:
        retriever = PubMedRetriever()
        articles = retriever.invoke(query)
        
        if not articles:
            return {"status": "success", "urls": [], "message": "No articles found."}

        article_list = []
        for article in articles[:num_articles]:
            uid = article.metadata.get("uid", None)
            if uid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
                article_list.append(url)

        return {"status": "success", "urls": article_list}
    
    except Exception as e:
        logger.error(f"Error in retrieve_pubmed_articles: {e}")
        return {"status": "error", "message": str(e)}

#This is a function to get the text from the pdfs
def get_quotes_and_authors(page_contents):
    soup = BeautifulSoup(page_contents, 'html.parser')
    text = soup.find_all('p', class_='text')
    return text



async def process_and_store_articles(article_urls: list) -> Dict[str, Any]:
    """
    Fetches article content using AsyncHtmlLoader, extracts text from all <p> tags,
    and uploads it to Pinecone.

    Args:
        article_urls (list): List of article URLs to process.

    Returns:
        dict: Status and details of stored articles.
    """
    processed_articles = []

    try:
        # Step 1: Load HTML content
        loader = AsyncHtmlLoader(article_urls)
        docs = loader.load()  # Remove `await` if `load` is synchronous

        # Step 2: Process each document
        for i, doc in enumerate(docs):
            try:
                # Parse the HTML content with BeautifulSoup
                soup = BeautifulSoup(doc.page_content, 'html.parser')

                # Extract all text content from <p> tags
                paragraphs = soup.find_all("p")
                text = [p.get_text(strip=True) for p in paragraphs]  # Get the text from each <p> tag

                # Combine all extracted text into a single string
                article_text = " ".join(text).strip()

                # Skip if no content is extracted
                if not article_text:
                    print(f"No text content found for URL: {article_urls[i]}")
                    continue

                # Generate embeddings for the combined text
                embedding = embeddings.embed_query(article_text)

                # Add the document to Pinecone
                vectorstore.upsert(vectors=[
                    {"id": article_urls[i], "values": embedding, "metadata": {"text": article_text, "source": article_urls[i]}}
                ])
                processed_articles.append({"url": article_urls[i], "status": "success"})
            except Exception as e:
                processed_articles.append({"url": article_urls[i], "status": "error", "error": str(e)})

    except Exception as e:
        print(f"Error during processing: {e}")
        return {"status": "error", "message": str(e)}

    return {"status": "completed", "processed_articles": processed_articles}

#This is a function that retrieves the text needed for the query from Pinecone
def retrieve_from_pinecone(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Retrieves the most relevant text from Pinecone based on the query.

    Args:
        query (str): The search query.
        top_k (int): The number of most relevant results to retrieve.

    Returns:
        dict: A dictionary containing the retrieved text and metadata.
    """
    try:
        # Generate embedding for the query
        query_embedding = embeddings.embed_query(query)

        # Perform a similarity search in Pinecone
        search_results = vectorstore.query(
            namespace=None,  # Specify a namespace if used during upserts
            top_k=top_k,
            include_metadata=True,
            vector=query_embedding
        )

        # Extract and format results
        retrieved_texts = []
        for match in search_results.matches:
            retrieved_texts.append({
                "source": match.metadata.get("source", ""),
                "text": match.metadata.get("text", "")  # Ensure text was stored in metadata
            })

        return {"status": "success", "results": retrieved_texts}

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return {"status": "error", "message": str(e)}



function_map = {
    # "query_router": query_router,
    "classify_mri": classify_mri,
    "analyze_exam": analyze_exam,
    "retrieve_pubmed_articles": retrieve_pubmed_articles,
    # "generate_follow_up_questions": generate_follow_up_questions
  

}