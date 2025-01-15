# Define a tool for classifying MRI images
functions = [ 
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "classify_mri",
    #         "description": "Classify MRI breast images by interacting with an external model.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "image_url": {
    #                     "type": "string",
    #                     "description": "URL to the MRI image that needs classification."
    #                 },
    #             },
    #             "required": ["image_url"],
    #             "additionalProperties": False,
    #         },
    #     }
    # },

    # {
    #     "type": "function",
    #     "function": {
    #         "name": "analyze_exam",
    #         "description": "Analyze breast cancer patient's exam PDFs and extract relevant medical data.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "pdf_url": {
    #                     "type": "string",
    #                     "description": "Extracted text from the exam PDF that needs to be analyzed."
    #                 },
    #             },
    #             "required": ["exam_text"],
    #             "additionalProperties": False,
    #         },
    #     }
    # },
    
<<<<<<< HEAD
    {
        "type": "function",
        "function": {
            "name": "retrieve_pubmed_research",
            "description": "Retrieve relevant research articles on breast cancer from PubMed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for retrieving breast cancer research articles."
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }
    }
]
=======
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "retrieve_pubmed_research",
    #         "description": "Retrieve relevant research articles on breast cancer from PubMed.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "query": {
    #                     "type": "string",
    #                     "description": "Search query for retrieving breast cancer research articles."
    #                 },
    #             },
    #             "required": ["query"],
    #             "additionalProperties": False,
    #         },
    #     }
    # }

    # {
    #     "type": "function",
    #     "function": {
    #         "name": "process_and_store_articles",
    #         "description": "process the articles retrieved ftom PubMed (text file) and store them in Pinecone",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "pdf_url": {
    #                     "type": "string",
    #                     "description": "Extracted text from the pdf and store it in Pinecone"
    #                 },
    #             },
    #             "required": ["query"],
    #             "additionalProperties": False,
    #         },
    #     }
    # }

    # {
    #     "type": "function",
    #     "function": {
    #         "name": "retrieve_from_pinecone",
    #         "description": "Retrieve relevant text from pinecone.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "pdf_url": {
    #                     "type": "string",
    #                     "description": "Retrieved text about breast cancer based on query from pubmed."
    #                 },
    #             },
    #             "required": ["query"],
    #             "additionalProperties": False,
    #         },
    #     }
    # }
]
{
    "status": "success",
    "articles": [
        {"title": "Article Title 1", "url": "https://example.com/1"},
        {"title": "Article Title 2", "url": "https://example.com/2"},
        # More articles
    ]
}
>>>>>>> origin/main

