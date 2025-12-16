# HumAIne Swarm Project

## Overview

The HumAIne Swarm project provides an intelligent assistant, the "HumAIne Swarm Assistant," designed to support researchers and developers within the [HumAIne EU-funded research project](https://humaine-horizon.eu/). This assistant facilitates AI/ML development by enabling conversational interaction with the project's MLOps infrastructure, primarily Kubeflow and MinIO, as well as providing access to project-specific knowledge.

The project leverages Large Language Models (LLMs) through libraries such as OpenAI, LangChain, and LlamaIndex, with a user interface built using Chainlit. The core idea is to enable "Swarm Learning" principles, fostering collaboration between humans and AI models, orchestrated by LLM agents.

## Architecture and Key Technologies

*   **Conversational AI:**
    *   **LLM:** Utilizes OpenAI models for understanding queries and generating responses.
    *   **Agent Framework:** Leverages LangChain and LlamaIndex for building advanced agentic RAG (Retrieval Augmented Generation) applications.
    *   **User Interface:** Employs Chainlit to provide an interactive chat interface.
*   **MLOps & Infrastructure Interaction:**
    *   **Kubeflow:** Agents can interact with Kubeflow to manage pipelines (list, get details, run, manage versions), experiments, and runs.
    *   **MinIO:** Agents can access and manage data in MinIO object storage, including listing buckets, retrieving pipeline artifacts (datasets, models, metrics, visualizations), and comparing run outputs.
*   **Knowledge Management:**
    *   **Vector Database:** Uses Pinecone for storing and retrieving information via semantic search (RAG), allowing users to query project documentation and related knowledge.
*   **Core Backend:**
    *   **Language:** Python
    *   **Dependency Management:** Poetry
    *   **Observability:** Instrumented with Literal AI for monitoring and debugging.
*   **Swarm Intelligence:** Integrates the `openai/swarm` library, suggesting a focus on multi-agent collaboration or advanced human-AI interaction patterns.
*   **Data Science Libraries:** Includes standard Python libraries for machine learning and data analysis such as TensorFlow, scikit-learn, Matplotlib, and Seaborn.

## Capabilities

The HumAIne Swarm Assistant can perform a wide range of tasks, including:

*   **Documentation & Information Retrieval:**
    *   Answer questions about the HumAIne project.
    *   Retrieve relevant documents and information using RAG from a Pinecone vector store.
    *   Optimize user queries for better information retrieval.
*   **Kubeflow Management:**
    *   List available Kubeflow pipelines, experiments, and runs.
    *   Provide detailed information about specific pipelines, versions, runs, and experiments.
    *   Trigger and monitor Kubeflow pipeline executions.
    *   Create and manage Kubeflow experiments.
    *   Retrieve user-specific Kubeflow namespace information.
*   **MinIO Storage Interaction:**
    *   List MinIO buckets and their contents.
    *   Retrieve and inspect pipeline artifacts (models, datasets, metrics files, HTML visualizations) stored in MinIO.
    *   Compare metrics from different pipeline runs.
*   **ML Workflow Support:**
    *   Guide users in setting up and running ML pipelines.
    *   Provide information about available ML components, models, and datasets.
    *   Assist in troubleshooting common ML pipeline issues.

## Getting Started

### Local Development

1.  **Environment Setup:**
    *   Copy `env.sh.example` to `env.sh`.
    *   Fill in the required environment variables in `env.sh` (OpenAI API key, Pinecone API key, MinIO credentials, Kubeflow details, etc.).
    *   Source the environment: `source env.sh`
2.  **Install Dependencies:**
    *   Ensure you have Python (>=3.11, <3.13) and Poetry installed.
    *   Run `poetry install` to install project dependencies.
3.  **Run the Application:**
    *   Execute `chainlit run app.py -w` to start the Chainlit application.


### Docker Deployment

1.  **Build the Docker image:**
    ```bash
    docker build -t humaine-swarm:latest .
    ```

2.  **Create environment file:**
    ```bash
    # Copy and edit .env-example
    cp .env-example .env
    # Edit .env with your actual values
    ```
3.  **Run the container:**
    ```bash
    docker run -d \
      --name humaine-swarm-assistant \
      -p 8000:8000 \
      --env-file .env \
      --restart unless-stopped \
      humaine-swarm:latest
    ```

## Usage

Once the application is running, open your browser to the address provided by Chainlit (usually `http://localhost:8000`). You can then interact with the "HumAIne Swarm Assistant" through the chat interface.

Ask questions or give commands related to your ML workflows, Kubeflow pipelines, MinIO artifacts, or the HumAIne project. For example:

*   "List all Kubeflow pipelines."
*   "Show me the details for pipeline ID 'xyz'."
*   "Run the 'training-pipeline' with parameter 'epochs=10'."
*   "What artifacts are available for the latest run of the 'data-processing' pipeline?"
*   "Compare the accuracy of run 'abc' and run 'def' for the 'classification-model'."
*   "Tell me about the goals of the HumAIne project."

The assistant will use its configured tools to fetch information and execute tasks, providing responses and results in the chat.



## License

Apache License 2.0 (as per `pyproject.toml` and `LICENSE` file)

