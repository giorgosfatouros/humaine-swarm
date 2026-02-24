# Capabilities

Structured overview of what you can ask and what the assistant uses under the hood. No implementation details — only user-facing behaviour.

## run_id vs run_name

- **run_id**: Unique ID assigned by **Kubeflow** to a pipeline run. Used when you ask for “run details” or “status of run [id]”.
- **run_name**: String used in **MinIO** paths to organize artifacts for a run (often includes pipeline name and run identifier). Used when you ask for “metrics for run X”, “artifacts for run X”, or “compare runs X and Y”.

The assistant will use the correct one depending on the question; when you have a run_id from Kubeflow, you may need to discover the corresponding run_name in MinIO (e.g. by listing artifacts or runs) for metrics/compare.

---

## Documentation (RAG)

| What you can ask | Example prompts |
|------------------|------------------|
| Project and Kubeflow docs | “Tell me about the HumAIne project”; “What are HumAIne’s AI paradigms?”; “How do Kubeflow pipelines work in this project?” |

The assistant searches a knowledge base built from project documentation (e.g. PDFs in `.docs`) and returns relevant excerpts in its answer.

---

## Kubeflow

| Capability | Example prompts |
|------------|------------------|
| List pipelines | “List pipelines”; “Show me diabetes-related pipelines” |
| Pipeline details | “Details for pipeline [pipeline_id]”; “What does pipeline [id] do?” |
| Pipeline versions | “What versions does pipeline [id] have?”; “Pipeline version details for [id] version [version_id]” |
| List runs | “List my runs”; “Runs for experiment [experiment_id]” |
| Run details | “Status of run [run_id]”; “Details for run [run_id]” |
| List experiments | “List experiments”; “What experiments exist?” |
| Experiment details | “Details for experiment [experiment_id]” or “experiment named [name]” |
| Create experiment | “Create an experiment called [name]” |
| Run pipeline | “Run pipeline [id] in experiment [exp_id] with job name [name]” (and parameters as needed) |
| Pipeline ID by name | “What is the pipeline ID for [pipeline name]?” |
| Your namespace | “What’s my Kubeflow namespace?” |

---

## MinIO

All MinIO actions require knowing a **bucket name**. Start with: **“What buckets do I have access to?”** or **“List my buckets.”**

| Capability | Example prompts |
|------------|------------------|
| Bucket contents | “What’s in bucket [bucket_name]?”; “List objects in [bucket] with prefix [path]” |
| Pipeline artifacts | “Artifacts for pipeline [pipeline_name] in bucket [bucket]”; “Artifacts for run [run_name]” |
| Model metrics | “Metrics for pipeline [name] in bucket [bucket]”; “Metrics for run [run_name]” |
| Visualizations | “Confusion matrix for pipeline [name] run [run_name]”; “ROC curve for run [run_name]”; “Feature importance for run [run_name]” |
| Compare runs | “Compare runs [run1], [run2] for pipeline [name] in bucket [bucket]”; optionally “compare accuracy and F1” |

**run_name** here is the MinIO path segment for that run (as in artifact paths), not the Kubeflow run_id.

---

## PDF from MinIO

| Capability | Example prompts |
|------------|------------------|
| Parse PDF | “Parse the PDF at [object_path] in bucket [bucket]” |
| Summarize | “Parse and summarize the PDF at [path] in bucket [bucket]” |

You can optionally specify chunk size, overlap, and summary type (e.g. map_reduce). The assistant can show the PDF in the sidebar and return extracted text and optional summary.

---

## Visualization (plot data)

| Capability | Example prompts |
|------------|------------------|
| Chart from data | “Plot this data as a bar chart: …”; “Create a line chart of [data]”; “Show a pie chart for [data]” |

The assistant can build line, bar, pie, scatter, or histogram charts from structured data you provide (or from metrics it has retrieved) and display them inline.
