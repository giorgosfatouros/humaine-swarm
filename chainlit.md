# Welcome to HumAIne Swarm Assistant! 🤖

Welcome to the **HumAIne Swarm Assistant**, an AI-powered assistant developed as part of the [HumAIne](https://humaine-horizon.eu/) EU-funded research project. This assistant helps researchers and developers interact with Kubeflow infrastructure, ML pipelines, and related tools to facilitate AI/ML development workflows.

## What Can This Assistant Do?

The HumAIne Swarm Assistant is designed to help you with:

- **Kubeflow Pipeline Management**: Discover, explore, and execute ML pipelines
- **MinIO Storage Access**: Browse and retrieve ML artifacts, models, metrics, and visualizations
- **Experiment Management**: Create and manage Kubeflow experiments
- **Model Analysis**: Compare pipeline runs, analyze metrics, and view visualizations
- **Project Documentation**: Access information about the HumAIne project and Kubeflow documentation

## Getting Started

### 1. Exploring Available Resources

Start by discovering what's available:

- **"What buckets do I have access to?"** - Lists your available MinIO storage buckets
- **"Show me available pipelines"** - Lists all Kubeflow pipelines you can access
- **"What experiments are available?"** - Lists your Kubeflow experiments
- **"Tell me about the HumAIne project"** - Retrieves project information and documentation

### 2. Working with Pipelines

#### Discovering Pipelines
- **"List all pipelines"** or **"Show me diabetes-related pipelines"** - Search and browse available pipelines
- **"Get details for pipeline [pipeline_id]"** - View detailed pipeline definitions
- **"What versions does pipeline [pipeline_id] have?"** - Explore pipeline versions

#### Running Pipelines
- **"Run pipeline [pipeline_id] with parameters..."** - Execute a pipeline with specific parameters
- You'll need to provide:
  - Experiment ID (where to run the pipeline)
  - Job name (a name for this run)
  - Pipeline parameters (as needed by the pipeline)

#### Monitoring Pipeline Runs
- **"List my pipeline runs"** - View all your pipeline executions
- **"Show me runs for experiment [experiment_id]"** - Filter runs by experiment
- **"Get details for run [run_id]"** - Check the status and details of a specific run

### 3. Accessing ML Artifacts and Results

#### Browsing Storage
- **"What's in bucket [bucket_name]?"** - Explore bucket contents
- **"Show me artifacts for pipeline [pipeline_name]"** - Find artifacts stored in MinIO
- **"What artifacts are in run [run_name]?"** - View artifacts for a specific pipeline run

#### Analyzing Model Performance
- **"Get metrics for pipeline [pipeline_name]"** - Retrieve model performance metrics (accuracy, precision, recall, etc.)
- **"Show metrics for run [run_name]"** - Get metrics for a specific run
- **"Compare runs [run1, run2, run3] for pipeline [pipeline_name]"** - Compare performance across multiple runs

#### Viewing Visualizations
- **"Show me the confusion matrix for run [run_name]"** - Get confusion matrix visualization
- **"Display ROC curve for pipeline [pipeline_name] run [run_name]"** - View ROC curve
- **"Show feature importance for [run_name]"** - View feature importance plots

### 4. Managing Experiments

- **"Create a new experiment called [name]"** - Create a new Kubeflow experiment
- **"Show me experiment [experiment_id] details"** - Get detailed experiment information
- **"What's my Kubeflow namespace?"** - Check your assigned namespace

## Important Notes

### MinIO Buckets
- Your access to MinIO buckets depends on your user policies
- Always start by listing your available buckets: **"What buckets can I access?"**
- All MinIO operations require specifying a `bucket_name` parameter

### Understanding Run IDs vs Run Names
- **Run ID**: A unique identifier assigned by Kubeflow (used with `get_run_details`)
- **Run Name**: A string used in MinIO paths to organize artifacts (used with metrics and artifact functions)
- The assistant will help you navigate between these when needed

### Getting Help
- Ask the assistant directly: **"How do I...?"** or **"What can you help me with?"**
- Request documentation: **"Show me documentation about [topic]"**
- The assistant can guide you through any workflow step-by-step

## Example Workflows

### Complete Pipeline Workflow
1. **"List available pipelines"** - Find a pipeline to work with
2. **"Show me details for pipeline [id]"** - Understand what the pipeline does
3. **"List my experiments"** - Find or create an experiment
4. **"Run pipeline [id] in experiment [exp_id] with name [job_name]"** - Execute the pipeline
5. **"Show me runs for experiment [exp_id]"** - Monitor your run
6. **"Get metrics for pipeline [name] run [run_name]"** - Analyze results
7. **"Compare these runs: [run1, run2]"** - Compare different configurations

### Analyzing Previous Results
1. **"What buckets do I have access to?"** - Discover available storage
2. **"Show me artifacts for pipeline [name]"** - Find stored results
3. **"Get metrics for pipeline [name]"** - Review performance metrics
4. **"Compare runs [run1, run2, run3]"** - Identify best performing configuration

## Need Help?

The assistant is designed to be conversational and helpful. Simply ask questions in natural language, and it will:
- Understand your intent
- Call the appropriate tools
- Present results in a clear, structured format
- Guide you through any process step-by-step

For more information about the HumAIne project, visit: https://humaine-horizon.eu/

Happy researching! 🚀
