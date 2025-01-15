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

