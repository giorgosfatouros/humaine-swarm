import requests
import json
import sqlite3
from langchain_community.retrievers import PubMedRetriever

# This is the MRI classification function that the model can call
def classify_mri(image_url: str) -> dict:
    """
    Classify MRI images by sending them to an external model API for breast cancer diagnosis.
    Returns the classification result in JSON format.
    """
    # Example external API call to classify MRI images
    try:
        # Assuming the external service requires a POST request
        response = requests.post("https://example.com/classify_mri", json={"image_url": image_url})
        
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


# This is the function to retrieve breast cancer research from PubMed using LangChain's PubMedRetriever
def retrieve_pubmed_articles(query: str) -> dict:
    """
    Use PubMedRetriever (from LangChain) to fetch relevant breast cancer research articles.
    Returns a list of articles in JSON format.
    """
    try:
        retriever = PubMedRetriever()
        # Fetch research articles
        articles = retriever.get_relevant_documents(query)
        
        # Process the articles into a readable format
        article_list = [{"title": article.metadata["title"], "url": article.metadata["url"]} for article in articles]
        
        return {"status": "success", "articles": article_list}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


function_map = {
    # "query_router": query_router,
    "classify_mri": classify_mri,
    "analyze_exam": analyze_exam,
    "retrieve_pubmed_research": retrieve_pubmed_articles,
    # "generate_follow_up_questions": generate_follow_up_questions
}