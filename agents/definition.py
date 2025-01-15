# Define a tool for classifying MRI images
functions = [ 
    {
        "type": "function",
        "function": {
            "name": "get_humaine_info",  # Ensure this is present
            "description": "Retrieve relevant information about HumAIne EU-funded research project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for retrieving information about HumAIne EU-funded research project such project objectives, vision, consortium, tasks, technologies, etc."
                    },
                },
                "required": ["search_query"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_kubeflow_info",
            "description": "Retrieve information about all Kubeflow pipelines",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
    }
]
