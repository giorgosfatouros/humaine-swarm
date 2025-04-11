# Define a tool for classifying MRI images
functions = [ 
    {
        "type": "function",
        "function": {
            "name": "get_docs", 
            "description": "Retrieve information about HumAIne EU-funded research solutions and Kubeflow pipelines.",
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
            "name": "get_kf_pipelines",
            "description": "Retrieve information about Kubeflow pipelines with flexible search capabilities",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Optional term to search for in pipeline names and descriptions (case-insensitive partial match)"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace to filter pipelines (None returns shared pipelines)"
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of results to return per page (default: 20)",
                        "default": 20
                    },
                    "page_token": {
                        "type": "string",
                        "description": "Token for pagination (default: empty string for first page)",
                        "default": ""
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "How to sort results (default: created_at desc - newest first)",
                        "default": "created_at desc"
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_minio_info",
            "description": "Retrieve information about MinIO buckets and objects",
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket_name": {
                        "type": "string",
                        "description": "Optional name of the MinIO bucket to query. If not provided, lists all buckets."
                    },
                    "prefix": {
                        "type": "string",
                        "description": "Optional prefix to filter objects (like a folder path)",
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "Maximum number of items to return (default: 10)",
                        "default": 10
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_details",
            "description": "Retrieve detailed information about a specific Kubeflow pipeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {
                        "type": "string",
                        "description": "The unique identifier of the pipeline to retrieve details for"
                    }
                },
                "required": ["pipeline_id"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_version_details",
            "description": "Retrieve detailed information about a specific Kubeflow pipeline version, including code and components",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {
                        "type": "string",
                        "description": "The unique identifier of the pipeline"
                    },
                    "pipeline_version_id": {
                        "type": "string",
                        "description": "The unique identifier of the pipeline version"
                    }
                },
                "required": ["pipeline_id", "pipeline_version_id"],
                "additionalProperties": False,
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_pipeline",
            "description": "Run a Kubeflow pipeline with specified parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {
                        "type": "string",
                        "description": "ID of the experiment to run the pipeline in"
                    },
                    "job_name": {
                        "type": "string",
                        "description": "Name for this pipeline run job"
                    },
                    "params": {
                        "type": "object",
                        "description": "Dictionary of parameter name-value pairs for the pipeline"
                    },
                    "pipeline_id": {
                        "type": "string",
                        "description": "ID of the pipeline to run (use with or without version_id)"
                    },
                    "version_id": {
                        "type": "string",
                        "description": "Optional specific version of the pipeline to run"
                    },
                    "pipeline_package_path": {
                        "type": "string",
                        "description": "Alternative to pipeline_id - local path to pipeline package file"
                    },
                    "pipeline_root": {
                        "type": "string",
                        "description": "Root path for pipeline outputs"
                    },
                    "enable_caching": {
                        "type": "boolean",
                        "description": "Whether to enable caching for pipeline tasks (default: true)",
                        "default": True
                    },
                    "service_account": {
                        "type": "string",
                        "description": "Kubernetes service account to use for this run"
                    }
                },
                "required": ["experiment_id", "job_name"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_runs",
            "description": "Retrieve information about Kubeflow pipeline runs with flexible filtering options",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Optional term to search for in run names (case-insensitive partial match)"
                    },
                    "experiment_id": {
                        "type": "string",
                        "description": "Optional experiment ID to filter runs"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace to filter runs"
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of results to return per page (default: 20)",
                        "default": 20
                    },
                    "page_token": {
                        "type": "string",
                        "description": "Token for pagination (default: empty string for first page)",
                        "default": ""
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "How to sort results (default: created_at desc - newest first)",
                        "default": "created_at desc"
                    },
                    "status_filter": {
                        "type": "string",
                        "description": "Optional filter for run status (e.g. 'SUCCEEDED', 'FAILED', 'RUNNING')"
                    }
                },
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_run_details",
            "description": "Retrieve detailed information about a specific Kubeflow pipeline run",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "The unique identifier of the run to retrieve details for"
                    }
                },
                "required": ["run_id"],
                "additionalProperties": False
            }
        }
    }
]
