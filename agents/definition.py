# Define a tool for classifying MRI images
functions = [ 
    {
        "type": "function",
        "function": {
            "name": "get_docs", 
            "description": "Retrieve information from the HumAIne RAG knowledge base: platform integration and Training Centre deliverables, Active Learning (modAL, HumAL ticketing pilot), XAI (humaine-explainerdashboard, SHAP/LIME), HumAIne Swarm API and usage, Kubeflow/project documentation, and HAIC evaluation framework content when indexed (HAIC Benchmark Suite, logging schema haic.decisions.v1, metrics F/D/HCL/Tr/A/S/EL/EfficiencyScore and interpretation).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for indexed HumAIne documentation. Use for project deliverables, Active Learning, XAI and explainer dashboard, Swarm assistant capabilities, Kubeflow/MLOps project docs, HAIC metrics/logging/benchmark suite, and hackathon AL+XAI technical questions."
                    },
                },
                "required": ["query"],
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
            "description": "Retrieve information about objects in a specific MinIO bucket",
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket_name": {
                        "type": "string",
                        "description": "Name of the MinIO bucket to query. Use list_user_buckets() to discover available buckets."
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
                "required": ["bucket_name"],
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
            "name": "get_pipeline_artifacts_from_MinIO",
            "description": "Retrieve actual ML pipeline artifacts (models, metrics, visualizations) stored in MinIO. Use for exploring pipeline outputs or fetching a specific file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket_name": {
                        "type": "string",
                        "description": "Name of the MinIO bucket to query. Use list_user_buckets() to discover available buckets."
                    },
                    "pipeline_name": {
                        "type": "string",
                        "description": "Name of the pipeline to query artifacts for (e.g. 'diabetes-svm-classification')"
                    },
                    "run_name": {
                        "type": "string",
                        "description": "Optional specific run name to filter artifacts. It is MinIO object name that contains artifacts for a specific run of a pipeline"
                    },
                    "artifact_type": {
                        "type": "string",
                        "description": "Optional artifact type to filter (e.g. 'models', 'metrics', 'plots')"
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "Maximum number of items to return",
                        "default": 20
                    },
                    "object_path": {
                        "type": "string",
                        "description": "Optional direct MinIO object path to retrieve a specific file (overrides other parameters). Use after discovering paths with list_user_buckets or get_minio_info."
                    }
                },
                "required": ["bucket_name"],
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
                    "bucket_name": {
                        "type": "string",
                        "description": "Name of the MinIO bucket to query. Use list_user_buckets() to discover available buckets."
                    },
                    "pipeline_name": {
                        "type": "string",
                        "description": "Name of the pipeline (e.g. 'diabetes-svm-classification')"
                    },
                    "run_name": {
                        "type": "string",
                        "description": "Optional specific run name to get metrics for. It is MinIO object name that contains artifacts for a specific run of a pipeline"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Optional model name to filter metrics (e.g. 'Support Vector Machine')"
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "Maximum number of items to return",
                        "default": 20
                    },
                    "metrics_path": {
                        "type": "string",
                        "description": "Optional direct path to a specific metrics JSON file (overrides other parameters). Use after discovering paths with list_user_buckets or get_minio_info."
                    }
                },
                "required": ["bucket_name"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_visualization",
            "description": "Retrieve PRE-EXISTING visual model results (confusion matrices, ROC curves, feature importance) as HTML content from MinIO storage. These are visualizations that were generated during Kubeflow ML pipeline execution. NOTE: This fetches existing files from ML pipeline runs - to CREATE new visualizations from any data, use plot_data instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket_name": {
                        "type": "string",
                        "description": "Name of the MinIO bucket to query. Use list_user_buckets() to discover available buckets."
                    },
                    "pipeline_name": {
                        "type": "string",
                        "description": "Name of the pipeline (e.g. 'diabetes-svm-classification')"
                    },
                    "run_name": {
                        "type": "string",
                        "description": "Run name to get visualizations for. It is MinIO object name that contains artifacts for a specific run of a pipeline"
                    },
                    "visualization_type": {
                        "type": "string",
                        "description": "Type of visualization to retrieve ('confusion_matrix', 'roc_curve', 'feature_importance')"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Optional model name part to filter by (e.g. 'svm')"
                    },
                    "visualization_path": {
                        "type": "string",
                        "description": "Optional direct path to a specific HTML visualization file (overrides other parameters). Use after discovering paths with list_user_buckets or get_minio_info."
                    }
                },
                "required": ["bucket_name"],
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
                    "bucket_name": {
                        "type": "string",
                        "description": "Name of the MinIO bucket to query. Use list_user_buckets() to discover available buckets."
                    },
                    "pipeline_name": {
                        "type": "string",
                        "description": "Name of the pipeline (e.g. 'diabetes-svm-classification')"
                    },
                    "run_names": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of run names to compare. It is MinIO object name that contains artifacts for a specific run of a pipeline"
                    },
                    "metric_names": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional list of specific metrics to compare (e.g. ['accuracy', 'precision'])"
                    }
                },
                "required": ["bucket_name", "pipeline_name", "run_names"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_user_buckets",
            "description": "List all available storage buckets with information about what data, ML pipelines and artifacts they contain. Returns data_files dict organized by type (pickle, json, pdf) with EXACT file paths - USE THESE EXACT PATHS when calling other tools like analyze_smart_cities_data or compare_smart_cities_files. Do not modify or guess file paths.",
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
            "name": "get_user_kubeflow_namespace",
            "description": "Gets user namespace in kubeflow. It is unrelated to MinIO buckets",
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
    },
    {
        "type": "function",
        "function": {
            "name": "parse_pdf_from_minio",
            "description": "Download and parse text from a PDF file stored in MinIO. Extracts text content, chunks it for processing, and optionally provides summarization using LangChain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket_name": {
                        "type": "string",
                        "description": "Name of the MinIO bucket containing the PDF file. Use list_user_buckets() to discover available buckets."
                    },
                    "object_path": {
                        "type": "string",
                        "description": "Full path to the PDF file in MinIO (e.g. 'documents/report.pdf' or 'kubeflow/pipeline-name/run-name/document.pdf')"
                    },
                    "summarize": {
                        "type": "boolean",
                        "description": "Whether to generate a summary of the PDF content (default: false)",
                        "default": False
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Character size for text chunks when splitting the document (default: 2000)",
                        "default": 2000
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "description": "Number of characters to overlap between chunks (default: 200)",
                        "default": 200
                    },
                    "summary_type": {
                        "type": "string",
                        "description": "Type of summarization chain to use: 'map_reduce' (for large documents), 'refine' (iterative refinement), or 'stuff' (for small documents). Default: 'map_reduce'",
                        "enum": ["map_reduce", "refine", "stuff"],
                        "default": "map_reduce"
                    }
                },
                "required": ["bucket_name", "object_path"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Create an interactive Plotly visualization from ANY data. This is a generic tool for all users and pilots - use it to visualize analysis results, metrics, comparisons, confusion matrices, decision distributions, or any structured data. The agent should intelligently decide on the chart type (line, bar, pie, scatter) based on the data structure if chart_type is not specified. NOTE: This creates NEW charts from data you provide. For fetching pre-existing ML pipeline visualizations stored in MinIO, use get_pipeline_visualization instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "description": "The data to plot. Can be: a list of numbers (single series), a list of [x, y] pairs (as arrays of 2 numbers), a dict with keys as labels and values as numeric data, or a list of dicts (tabular data). For tabular data, use x_column and y_column to specify which columns to plot. The function will automatically detect the data structure and choose the appropriate chart type. Examples: [1, 2, 3, 4] for a simple series, [[1, 2], [3, 4], [5, 6]] for x-y pairs, {'A': 10, 'B': 20} for categorical data, or [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}] for tabular data.",
                        "oneOf": [
                            {
                                "type": "array",
                                "items": {}
                            },
                            {"type": "object"}
                        ]
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": ["line", "bar", "pie", "scatter", "histogram"],
                        "description": "Optional chart type. If not provided, will be automatically determined based on data structure. Use 'line' for time series or continuous data, 'bar' for categorical comparisons, 'pie' for proportions/percentages, 'scatter' for relationships between two variables, 'histogram' for distributions."
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the chart"
                    },
                    "x_label": {
                        "type": "string",
                        "description": "Optional label for x-axis"
                    },
                    "y_label": {
                        "type": "string",
                        "description": "Optional label for y-axis"
                    },
                    "x_column": {
                        "type": "string",
                        "description": "Optional column name for x-axis (required for tabular data - list of dicts)"
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Optional column name for y-axis (required for tabular data - list of dicts)"
                    },
                    "color_column": {
                        "type": "string",
                        "description": "Optional column name for color grouping (for multi-series visualizations)"
                    },
                    "display": {
                        "type": "string",
                        "enum": ["inline", "side", "page"],
                        "description": "How to display the chart in the UI. 'inline' shows it within the message, 'side' shows it in a sidebar, 'page' shows it on a separate page. Default: 'inline'",
                        "default": "inline"
                    },
                    "size": {
                        "type": "string",
                        "enum": ["small", "medium", "large"],
                        "description": "Size of the chart. Only works with display='inline'. Default: 'medium'",
                        "default": "medium"
                    }
                },
                "required": ["data"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_smart_cities_data",
            "description": "Analyze Smart Cities pilot application data from pickle files stored in MinIO. Provides insights on error types, AI decisions, operator decisions, and processing performance. This tool is only available to authorized Smart Cities pilot users. If file path is incorrect, returns available_files with correct paths organized by type. IMPORTANT: Use EXACT paths from list_user_buckets() data_files - do not guess or modify paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket_name": {
                        "type": "string",
                        "description": "Name of the MinIO bucket containing the pickle file. Use list_user_buckets() to discover available buckets."
                    },
                    "object_path": {
                        "type": "string",
                        "description": "Full path to the pickle file in MinIO (e.g., 'pilot-data/sim-pilot-apps-v0.pkl')"
                    },
                    "query_type": {
                        "type": "string",
                        "enum": ["overview", "error_distribution", "ai_decisions", "operator_decisions", "processing_time", "field_errors", "decision_flow", "confusion_matrix", "ai_accuracy_metrics"],
                        "description": "Type of analysis to perform: 'overview' for dataset summary, 'error_distribution' for error type counts, 'ai_decisions' for AI decision distribution (Accepted/Rejected/Flagged), 'operator_decisions' for operator review outcomes, 'processing_time' for time statistics per application, 'field_errors' for GT vs APP field discrepancies, 'decision_flow' for AI to operator decision flow, 'confusion_matrix' for AI decision accuracy matrix (TP/FP/TN/FN based on GT vs APP comparison), 'ai_accuracy_metrics' for detailed accuracy metrics with field breakdown"
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "Optional filter value for specific queries"
                    }
                },
                "required": ["bucket_name", "object_path", "query_type"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_smart_cities_files",
            "description": "Compare Smart Cities pilot data across multiple pickle files. Use this tool to compare decisions, processing times, and operator interventions across different data files. Returns aggregated statistics normalized by row count. Only available to Smart Cities pilot users.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket_name": {
                        "type": "string",
                        "description": "Name of the MinIO bucket containing the pickle files. Use list_user_buckets() to discover available buckets."
                    },
                    "object_paths": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of paths to pickle files in MinIO to compare (e.g., ['pilot-data/sim-pilot-apps-v0.pkl', 'pilot-data/sim-pilot-apps-v1.pkl'])"
                    },
                    "comparison_type": {
                        "type": "string",
                        "enum": ["decisions", "processing_time", "full_summary"],
                        "description": "Type of comparison: 'decisions' to compare AI and operator decision distributions, 'processing_time' to compare AI and operator processing times per application, 'full_summary' for complete comparison including a summary table with all metrics normalized by total rows"
                    }
                },
                "required": ["bucket_name", "object_paths"],
                "additionalProperties": False
            }
        }
    }
]
