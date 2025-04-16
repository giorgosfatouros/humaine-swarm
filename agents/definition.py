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
            "description": "Retrieve basic Kubeflow pipeline listings (pipeline registry info only, NOT artifacts or actual results)",
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
            "description": "Retrieve basic Kubeflow pipeline definition details (NOT actual outputs or metrics - use get_pipeline_artifacts for those)",
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
            "description": "List Kubeflow pipeline runs with basic status information (does NOT include actual metrics or visualizations)",
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
            "description": "Retrieve basic run status details from Kubeflow (does NOT include actual metrics or visualizations - use MinIO functions for those)",
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_artifacts",
            "description": "Retrieve actual ML pipeline artifacts (models, metrics, visualizations) stored in MinIO. Use for exploring pipeline outputs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_name": {
                        "type": "string",
                        "description": "Name of the pipeline to query artifacts for (e.g. 'diabetes-svm-classification')"
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Optional specific run ID to filter artifacts"
                    },
                    "artifact_type": {
                        "type": "string",
                        "description": "Optional artifact type to filter (e.g. 'models', 'metrics', 'plots')"
                    },
                    "bucket_name": {
                        "type": "string",
                        "description": "Storage bucket name where artifacts are stored (use list_user_buckets first if unsure)"
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "Maximum number of items to return",
                        "default": 20
                    }
                },
                "required": ["pipeline_name", "bucket_name"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_metrics",
            "description": "Retrieve detailed model performance metrics (accuracy, precision, etc.) from pipeline runs. Best for analyzing model results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_name": {
                        "type": "string",
                        "description": "Name of the pipeline (e.g. 'diabetes-svm-classification')"
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Optional specific run ID to get metrics for"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Optional model name to filter metrics (e.g. 'Support Vector Machine')"
                    },
                    "bucket_name": {
                        "type": "string",
                        "description": "Storage bucket name where metrics are stored (use list_user_buckets first if unsure)"
                    }
                },
                "required": ["pipeline_name", "bucket_name"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_visualization",
            "description": "Retrieve visual model results (confusion matrices, ROC curves, feature importance) as HTML content for display",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_name": {
                        "type": "string",
                        "description": "Name of the pipeline (e.g. 'diabetes-svm-classification')"
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Run ID to get visualizations for"
                    },
                    "visualization_type": {
                        "type": "string",
                        "description": "Type of visualization to retrieve ('confusion_matrix', 'roc_curve', 'feature_importance')"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Optional model name part to filter by (e.g. 'svm')"
                    },
                    "bucket_name": {
                        "type": "string",
                        "description": "Storage bucket name where visualizations are stored (use list_user_buckets first if unsure)"
                    }
                },
                "required": ["pipeline_name", "run_id", "visualization_type", "bucket_name"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_pipeline_runs",
            "description": "Compare performance metrics across multiple pipeline runs. Best for analyzing which model/parameters performed best.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_name": {
                        "type": "string",
                        "description": "Name of the pipeline (e.g. 'diabetes-svm-classification')"
                    },
                    "run_ids": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of run IDs to compare"
                    },
                    "metric_names": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional list of specific metrics to compare (e.g. ['accuracy', 'precision'])"
                    },
                    "bucket_name": {
                        "type": "string",
                        "description": "Storage bucket name where metrics are stored (use list_user_buckets first if unsure)"
                    }
                },
                "required": ["pipeline_name", "run_ids", "bucket_name"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_user_buckets",
            "description": "List all available storage buckets with information about what data, ML pipelines and artifacts they contain",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_buckets": {
                        "type": "integer",
                        "description": "Maximum number of buckets to return",
                        "default": 20
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
            "name": "list_experiments",
            "description": "List Kubeflow experiments with options for filtering and pagination",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Optional term to search for in experiment names (case-insensitive partial match)"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace to filter experiments"
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of results to return per page (default: 10)",
                        "default": 10
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
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_experiment_details",
            "description": "Retrieve detailed information about a specific Kubeflow experiment",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {
                        "type": "string",
                        "description": "Optional ID of the experiment to retrieve details for"
                    },
                    "experiment_name": {
                        "type": "string",
                        "description": "Optional name of the experiment to retrieve details for"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace where the experiment is located"
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
            "name": "get_user_namespace",
            "description": "Gets user namespace in Kubeflow",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_experiment",
            "description": "Creates a new Kubeflow experiment",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the experiment"
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of the experiment"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional Kubernetes namespace to use"
                    }
                },
                "required": ["name"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_id",
            "description": "Gets the ID of a Kubeflow pipeline by its name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Pipeline name"
                    }
                },
                "required": ["name"],
                "additionalProperties": False
            }
        }
    }
]
