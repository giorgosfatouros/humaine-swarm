# HumAIne Kubeflow Guide 

This guide provides a overview of how to use Kubeflow for machine learning pipelines within HumAIne. It covers connecting to Kubeflow, creating components and pipelines, adding metadata for traceability, deploying pipelines, and running experiments.

## Table of Contents

1. [Connecting to Kubeflow](#1-connecting-to-kubeflow)
2. [Creating Components](#2-creating-components)
3. [Building Pipelines](#3-building-pipelines)
4. [Adding Metadata for Traceability](#4-adding-metadata-for-traceability)
5. [Deploying Pipelines to Kubeflow](#5-deploying-pipelines-to-kubeflow)
6. [Setting Input Parameters for Experiments](#6-setting-input-parameters-for-experiments)
7. [Visualizing Results with HTML Artifacts](#7-visualizing-results-with-html-artifacts)
8. [Common Metadata Standards](#8-common-metadata-standards)

## 1. Connecting to Kubeflow

Kubeflow provides a client interface that allows you to interact with the platform programmatically. You need to use the client manager (`KFPClientManager`) that handles authentication with Dex.

### Kubeflow Client Code

First, you'll need the following client code to authenticate with Kubeflow:

```python
import os
import re
import requests
import urllib3
import kfp
from urllib.parse import urlsplit, urlencode

class KFPClientManager:
    """
    A class that creates `kfp.Client` instances with Dex authentication.
    """

    def __init__(
        self,
        api_url: str,
        dex_username: str,
        dex_password: str,
        dex_auth_type: str = "local",
        skip_tls_verify: bool = False,
    ):
        """
        Initialize the KfpClient

        :param api_url: the Kubeflow Pipelines API URL
        :param skip_tls_verify: if True, skip TLS verification
        :param dex_username: the Dex username
        :param dex_password: the Dex password
        :param dex_auth_type: the auth type to use if Dex has multiple enabled, one of: ['ldap', 'local']
        """
        self._api_url = api_url
        self._skip_tls_verify = skip_tls_verify
        self._dex_username = dex_username
        self._dex_password = dex_password
        self._dex_auth_type = dex_auth_type
        self._client = None

        # disable SSL verification, if requested
        if self._skip_tls_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # ensure `dex_default_auth_type` is valid
        if self._dex_auth_type not in ["ldap", "local"]:
            raise ValueError(
                f"Invalid `dex_auth_type` '{self._dex_auth_type}', must be one of: ['ldap', 'local']"
            )

    def _get_session_cookies(self) -> str:
        """
        Get the session cookies by authenticating against Dex
        :return: a string of session cookies in the form "key1=value1; key2=value2"
        """

        # use a persistent session (for cookies)
        s = requests.Session()

        # GET the api_url, which should redirect to Dex
        resp = s.get(
            self._api_url, allow_redirects=True, verify=not self._skip_tls_verify
        )
        if resp.status_code == 200:
            pass
        elif resp.status_code == 403:
            url_obj = urlsplit(resp.url)
            url_obj = url_obj._replace(
                path="/oauth2/start", query=urlencode({"rd": url_obj.path})
            )
            resp = s.get(
                url_obj.geturl(), allow_redirects=True, verify=not self._skip_tls_verify
            )
        else:
            raise RuntimeError(
                f"HTTP status code '{resp.status_code}' for GET against: {self._api_url}"
            )

        # if we were NOT redirected, then the endpoint is unsecured
        if len(resp.history) == 0:
            # no cookies are needed
            return ""

        # if we are at `../auth` path, we need to select an auth type
        url_obj = urlsplit(resp.url)
        if re.search(r"/auth$", url_obj.path):
            url_obj = url_obj._replace(
                path=re.sub(r"/auth$", f"/auth/{self._dex_auth_type}", url_obj.path)
            )

        # if we are at `../auth/xxxx/login` path, then we are at the login page
        if re.search(r"/auth/.*/login$", url_obj.path):
            dex_login_url = url_obj.geturl()
        else:
            # otherwise, we need to follow a redirect to the login page
            resp = s.get(
                url_obj.geturl(), allow_redirects=True, verify=not self._skip_tls_verify
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP status code '{resp.status_code}' for GET against: {url_obj.geturl()}"
                )
            dex_login_url = resp.url

        # attempt Dex login
        resp = s.post(
            dex_login_url,
            data={"login": self._dex_username, "password": self._dex_password},
            allow_redirects=True,
            verify=not self._skip_tls_verify,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"HTTP status code '{resp.status_code}' for POST against: {dex_login_url}"
            )

        # if we were NOT redirected, then the login credentials were probably invalid
        if len(resp.history) == 0:
            raise RuntimeError(
                f"Login credentials are probably invalid - "
                f"No redirect after POST to: {dex_login_url}"
            )

        # if we are at `../approval` path, we need to approve the login
        url_obj = urlsplit(resp.url)
        if re.search(r"/approval$", url_obj.path):
            dex_approval_url = url_obj.geturl()

            # approve the login
            resp = s.post(
                dex_approval_url,
                data={"approval": "approve"},
                allow_redirects=True,
                verify=not self._skip_tls_verify,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP status code '{resp.status_code}' for POST against: {url_obj.geturl()}"
                )

        return "; ".join([f"{c.name}={c.value}" for c in s.cookies])

    def _create_kfp_client(self) -> kfp.Client:
        try:
            session_cookies = self._get_session_cookies()
        except Exception as ex:
            raise RuntimeError(f"Failed to get Dex session cookies") from ex

        # monkey patch the kfp.Client to support disabling SSL verification
        # kfp only added support in v2: https://github.com/kubeflow/pipelines/pull/7174
        original_load_config = kfp.Client._load_config

        def patched_load_config(client_self, *args, **kwargs):
            config = original_load_config(client_self, *args, **kwargs)
            config.verify_ssl = not self._skip_tls_verify
            return config

        patched_kfp_client = kfp.Client
        patched_kfp_client._load_config = patched_load_config

        return patched_kfp_client(
            host=self._api_url,
            cookies=session_cookies,
            namespace=os.getenv("KUBEFLOW_NAMESPACE")
            )

    def create_kfp_client(self) -> kfp.Client:
        """Get a newly authenticated Kubeflow Pipelines client."""
        return self._create_kfp_client()
            
def get_kubeflow_client() -> kfp.Client:
    kfp_client_manager = KFPClientManager(
        api_url=os.getenv("KUBEFLOW_HOST") + "/pipeline",
        skip_tls_verify=True,

        dex_username=os.getenv("KUBEFLOW_USERNAME"),
        dex_password=os.getenv("KUBEFLOW_PASSWORD"),

        # can be 'ldap' or 'local' depending on your Dex configuration
        dex_auth_type="local",
        )

    kfp_client = kfp_client_manager.create_kfp_client()
    return kfp_client
```

### Environment Setup

Before connecting, ensure you have the following environment variables set:

```python
import os

# Set these environment variables or use a .env file with python-dotenv
os.environ["KUBEFLOW_HOST"] = "http://huanew-kubeflow.ddns.net/"  # Or GFT's Kubeflow URL
os.environ["KUBEFLOW_USERNAME"] = "your-username"                # Will be provided by HUA
os.environ["KUBEFLOW_PASSWORD"] = "your-password"                # Will be provided by HUA
os.environ["KUBEFLOW_NAMESPACE"] = "your-namespace"              # Will be provided by HUA
```

### Creating a Kubeflow Client

Use the `get_kubeflow_client` function to create an authenticated client:

```python
# Create an authenticated client
client = get_kubeflow_client()
```

This client can now be used to interact with the Kubeflow API, including creating runs, checking statuses, and more.

## 2. Creating Components

Components are the building blocks of Kubeflow Pipelines. Each component is a self-contained piece of code that performs a specific ML task.

### Component Definition

In our code, we use the Kubeflow Pipelines SDK's component decorator to define components:

```python
import kfp
import kfp.dsl as dsl
from kfp.dsl import Dataset, Input, Output, Model, Metrics, ClassificationMetrics, HTML

@dsl.component(
    packages_to_install=["numpy", "scikit-learn", "pandas"]  # Specify required packages
)
def my_component(
    input_data: Input[Dataset],           # Input artifact
    output_data: Output[Dataset],         # Output artifact
    parameter_1: int = 42,                # Parameter with default value
    parameter_2: str = "default_value"    # Parameter with default value
):
    # Component implementation
    import numpy as np
    import pandas as pd
    
    # Your component logic here
    # ...
    
    # Save output data
    # ...
```

### Component Types

Based on the diabetes example, components typically fall into these categories:

1. **Data Loading Components**: Load and prepare datasets
2. **Data Preprocessing Components**: Clean, transform, and prepare data
3. **Model Training Components**: Train ML models with different algorithms
4. **Evaluation Components**: Assess model performance
5. **Comparison Components**: Compare multiple models

### Component Best Practices

1. **Clear Input/Output Definitions**: Clearly define all inputs, outputs, and parameters
2. **Package Dependencies**: List all required packages in `packages_to_install`
3. **Logging**: Include appropriate logging for debugging


## 3. Building Pipelines

Pipelines connect components into a directed acyclic graph (DAG) that represents your ML workflow.

### Pipeline Definition

Define pipelines using the `@dsl.pipeline` decorator:

```python
@dsl.pipeline(
    name='My ML Pipeline',
    description='A pipeline that processes data and trains a model' # ALWAYS ADD A COMPLDESCRIPTION
)
def my_pipeline(
    param1: int = 10,
    param2: float = 0.1,
    param3: str = 'default'
):
    # Component tasks
    data_task = load_data()
    
    process_task = process_data(
        input_data=data_task.outputs["output_data"],
        param1=param1
    )
    
    train_task = train_model(
        train_data=process_task.outputs["processed_data"],
        learning_rate=param2
    )
    
    evaluate_task = evaluate_model(
        model=train_task.outputs["model"],
        test_data=process_task.outputs["test_data"]
    )
```

### Pipeline Organization

Organize your pipeline to follow a logical flow:

1. **Data Acquisition & Loading**: Load or generate data
2. **Data Preprocessing**: Clean, transform, split data
3. **Model Training**: Train one or more models
4. **Model Evaluation**: Evaluate model performance
5. **Model Comparison**: Compare models (if applicable)
6. **Result Visualization**: Create visualizations of results

## 4. Adding Metadata for Traceability

Metadata is crucial for experiment tracking, reproducibility, and comparison. Kubeflow Pipelines allows you to attach metadata to artifacts.

### Adding Metadata to Artifacts

```python
# Example from diabetes pipeline
dataset.metadata.update({
    'creation_time': str(datetime.datetime.now()),
    'format': 'joblib',
    'size': X.shape[0],
    'sample_count': X.shape[0],
    'description': 'Diabetes dataset converted to binary classification',
    'source': 'sklearn.datasets',
    'feature_count': X.shape[1],
    'positive_class_samples': int(np.sum(y)),
    'negative_class_samples': int(np.sum(1-y))
})
```

### Standard Metadata Fields

We suggest using these standard metadata fields for consistency:

**Dataset Metadata:**
- `creation_time`: When the artifact was created
- `description`: Description of the dataset
- `format`: Format of the data (e.g., joblib, csv)
- `size`: Number of samples
- `feature_count`: Number of features
- `sample_count`: Number of samples
- `source`: Where the data came from

**Model Metadata:**
- `framework`: ML framework used (e.g., scikit-learn)
- `model_type`: Type of model (e.g., RandomForestClassifier)
- `creation_time`: When the model was created
- `version`: Model version
- `hyperparameters`: JSON string of hyperparameters
- `training_dataset_size`: Size of training dataset
- `description`: Description of the model

**Metrics Metadata:**
- `accuracy`, `precision`, `recall`, `f1_score`, etc.: Performance metrics
- `creation_time`: When the metrics were calculated
- `model_name`: Name of the model

## 5. Deploying Pipelines to Kubeflow

Once you've defined your pipeline, you can compile it to a YAML specification and upload it to Kubeflow.

### Compiling the Pipeline

```python
# Compile the pipeline to a YAML file
kfp.compiler.Compiler().compile(
    pipeline_func=my_pipeline,
    package_path='my_pipeline.yaml'
)
```

### Uploading and Running the Pipeline

```python
# Method 1: Upload the compiled YAML through the Kubeflow UI
# Navigate to Pipelines -> Upload pipeline -> Select your YAML file

# Method 2: Use the KFP client to create a run directly
client = get_kubeflow_client()

client.create_run_from_pipeline_func(
    my_pipeline,
    arguments={
        'param1': 20,
        'param2': 0.01,
        'param3': 'custom_value'
    },
    experiment_name='My Experiment'  # Group related runs together
)
```

## 6. Setting Input Parameters for Experiments

Kubeflow Pipelines allow you to parameterize your workflows, making it easy to run experiments with different configurations.

### Pipeline Parameters

Define parameters in your pipeline function signature with default values:

```python
@dsl.pipeline(
    name='Diabetes Classification Pipeline',
    description='A demonstration pipeline for diabetes classification'
)
def diabetes_classification_pipeline(
    test_size: float = 0.3,          # Fraction of data to use for testing
    random_state: int = 42,          # Random seed for reproducibility
    dt_max_depth: int = 5,           # Decision Tree max depth
    rf_n_estimators: int = 100,      # Random Forest number of trees
    svm_c: float = 1.0,              # SVM regularization parameter
    svm_kernel: str = 'rbf'          # SVM kernel type
):
    # Pipeline implementation
    # ...
```

### Running Experiments with Different Parameters

```python
# Run multiple experiments with different parameters
experiments = [
    {
        'name': 'Baseline',
        'params': {
            'test_size': 0.3,
            'random_state': 42,
            'dt_max_depth': 5,
            'rf_n_estimators': 100,
            'svm_c': 1.0,
            'svm_kernel': 'rbf'
        }
    },
    {
        'name': 'Deep Trees',
        'params': {
            'test_size': 0.3,
            'random_state': 42,
            'dt_max_depth': 10,
            'rf_n_estimators': 100,
            'svm_c': 1.0,
            'svm_kernel': 'rbf'
        }
    },
    # Add more experiment configurations
]

# Run all experiments
client = get_kubeflow_client()
for experiment in experiments:
    client.create_run_from_pipeline_func(
        diabetes_classification_pipeline,
        arguments=experiment['params'],
        experiment_name=experiment['name']
    )
```

### Experiment Tracking

After running experiments, you can:

1. View all experiments in the Kubeflow UI
2. Compare metrics between different runs
3. Visualize the results using the HTML artifacts
4. Download models and other artifacts for further analysis

## 7. Visualizing Results with HTML Artifacts

Kubeflow allows you to create HTML artifacts to visualize your results directly in the Kubeflow UI. This is particularly useful for displaying plots, charts, and other visual representations of your model's performance.

### Creating HTML Artifacts for Visualizations

Here's an example from our diabetes demo that shows how to create HTML artifacts with embedded plots:

```python
@dsl.component(
    packages_to_install=["numpy", "scikit-learn", "joblib", "matplotlib"]
)
def evaluate_model(model: Input[Model], 
                  test_data: Input[Dataset], 
                  feature_names: Input[Dataset], 
                  model_name: str,
                  metrics: Output[Metrics],
                  confusion_matrix: Output[ClassificationMetrics],
                  evaluation_plots: Output[HTML]):  # HTML output for visualizations
    import joblib
    import numpy as np
    import json
    import os
    import datetime
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as cm
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    
    # Load model, test data, and feature names
    model_obj = joblib.load(model.path)
    test_data_combined = joblib.load(test_data.path)
    feature_names_list = joblib.load(feature_names.path)
    
    # Extract features and target
    X_test = test_data_combined[:, :-1]
    y_test = test_data_combined[:, -1]
    
    # Make predictions
    y_pred = model_obj.predict(X_test)
    y_pred_proba = model_obj.predict_proba(X_test)[:, 1] if hasattr(model_obj, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = cm(y_test, y_pred).tolist()
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to confusion matrix cells
    thresh = np.array(conf_matrix).max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, conf_matrix[i][j],
                     horizontalalignment="center",
                     color="white" if conf_matrix[i][j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the confusion matrix plot to a base64 encoded string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    confusion_matrix_img = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # Generate HTML with embedded plots
    html_content = f"""
    <html>
    <head>
        <title>Model Evaluation: {model_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metrics-container {{ display: flex; flex-wrap: wrap; }}
            .metric-box {{ 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                padding: 15px; 
                margin: 10px; 
                background-color: #f9f9f9; 
                width: 300px;
            }}
            .metric-value {{ 
                font-size: 24px; 
                font-weight: bold; 
                color: #2a5885; 
            }}
            .plot-container {{ margin-top: 20px; }}
            h2 {{ color: #444; }}
        </style>
    </head>
    <body>
        <h1>Evaluation Results for {model_name}</h1>
        
        <div class="metrics-container">
            <div class="metric-box">
                <h3>Accuracy</h3>
                <div class="metric-value">{accuracy:.4f}</div>
            </div>
        </div>
        
        <div class="plot-container">
            <h2>Confusion Matrix</h2>
            <img src="data:image/png;base64,{confusion_matrix_img}" alt="Confusion Matrix">
        </div>
    """
    
    # Add ROC curve if available
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        roc_image = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        html_content += f"""
        <div class="plot-container">
            <h2>ROC Curve</h2>
            <img src="data:image/png;base64,{roc_image}" alt="ROC Curve">
        </div>
        """
    
    # Feature importance plot (for tree-based models)
    if hasattr(model_obj, 'feature_importances_'):
        importances = model_obj.feature_importances_
        
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[indices])
        plt.title(f'Feature Importance - {model_name}')
        plt.xticks(range(len(importances)), [feature_names_list[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        importance_image = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        html_content += f"""
        <div class="plot-container">
            <h2>Feature Importance</h2>
            <img src="data:image/png;base64,{importance_image}" alt="Feature Importance">
        </div>
        """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML to output
    with open(evaluation_plots.path, 'w') as f:
        f.write(html_content)
```

### Viewing HTML Artifacts in Kubeflow

The HTML artifacts can be viewed directly in the Kubeflow UI:

1. Navigate to the pipeline run in the Kubeflow UI
2. Click on the component that generated the HTML artifact
3. In the right panel, under "Outputs," click on the HTML artifact
4. The visualization will be displayed in a new tab

### Key Techniques for HTML Visualization

1. **Base64 Encoding Images**: Convert Matplotlib plots to base64-encoded strings to embed them in HTML
2. **Responsive Design**: Use CSS to make visualizations responsive and readable
3. **Interactive Elements**: Consider adding interactive elements using JavaScript
4. **Consistent Styling**: Maintain consistent styling across all visualizations
5. **Multiple Plots**: Combine related plots in a single HTML artifact for easier comparison

## 8. Common Metadata Standards

To ensure consistency across all use cases, we've established the following metadata standards:

### Dataset Metadata

```python
dataset.metadata.update({
    'creation_time': str(datetime.datetime.now()),
    'description': 'Brief description of the dataset',
    'format': 'Format of the data (csv, joblib, etc.)',
    'size': data.shape[0],
    'feature_count': data.shape[1],
    'source': 'Where the data came from',
    'preprocessing': 'Any preprocessing applied',
    # For classification datasets
    'class_distribution': {class_labels: counts},
    # Add project-specific metadata
    'project': 'Project name',
    'version': 'Dataset version'
})
```

### Model Metadata

```python
model.metadata.update({
    'framework': 'scikit-learn/pytorch/tensorflow/etc.',
    'model_type': 'Type of model',
    'creation_time': str(datetime.datetime.now()),
    'version': '1.0',
    'hyperparameters': json.dumps({
        # All hyperparameters used
        'param1': value1,
        'param2': value2
    }),
    'training_dataset_size': train_data.shape[0],
    'feature_count': train_data.shape[1],
    'description': 'Description of the model',
    # Add pilot-specific metadata
    'pilot': 'Pilot name',
    'author': 'Your name',
    'contact': 'Your email'
})
```

### Metrics Metadata

```python
metrics.metadata.update({
    'model_name': 'Name of the model',
    'creation_time': str(datetime.datetime.now()),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1_score),
    # For regression models
    'mae': float(mae),
    'mse': float(mse),
    'rmse': float(rmse),
    'r2': float(r2),
    # Add pilot-specific metrics
    'pilot': 'Pilot name',
    'dataset_version': 'Version of dataset used'
})
```

## Example: Complete Model Comparison Component

Below is a complete example of a component that compares multiple models and generates a comprehensive HTML visualization. This demonstrates many of the techniques discussed in this guide:


```python
@dsl.component(
    packages_to_install=["matplotlib"]
)
def compare_models(dt_metrics: Input[Metrics], 
                  rf_metrics: Input[Metrics], 
                  svm_metrics: Input[Metrics],
                  comparison_result: Output[HTML],
                  best_model_info: Output[Metrics]):
    import json
    import matplotlib.pyplot as plt
    import base64
    import datetime
    from io import BytesIO
    
    # Extract metrics
    model_names = ["Decision Tree", "Random Forest", "Support Vector Machine"]
    metrics_artifacts = [dt_metrics, rf_metrics, svm_metrics]
    
    # Collect metrics
    accuracies = [metrics.metadata["accuracy"] for metrics in metrics_artifacts]
    if "roc_auc" in dt_metrics.metadata:
        aucs = [metrics.metadata.get("roc_auc", 0) for metrics in metrics_artifacts]
    else:
        aucs = None
    
    # Find the best model based on accuracy
    best_model_idx = accuracies.index(max(accuracies))
    best_model = model_names[best_model_idx]
    
    # Create comparison plot for accuracy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['blue', 'green', 'red'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison - Diabetes Classification')
    plt.ylim([0, 1])
    
    # Highlight the best model
    bars[best_model_idx].set_color('gold')
    
    # Add text annotations
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    
    # Save the comparison plot
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    accuracy_comparison_image = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # Create AUC comparison if available
    auc_comparison_image = None
    if aucs:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, aucs, color=['blue', 'green', 'red'])
        plt.xlabel('Model')
        plt.ylabel('ROC AUC')
        plt.title('ROC AUC Comparison - Diabetes Classification')
        plt.ylim([0, 1])
        
        # Highlight the best model by AUC
        best_auc_idx = aucs.index(max(aucs))
        bars[best_auc_idx].set_color('gold')
        
        # Add text annotations
        for i, auc_val in enumerate(aucs):
            plt.text(i, auc_val + 0.02, f'{auc_val:.4f}', ha='center')
        
        plt.tight_layout()
        
        # Save the AUC comparison plot
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        auc_comparison_image = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
    
    # Create HTML for visual comparison
    html_content = """
    <html>
    <head>
        <title>Model Comparison Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .comparison-container { margin: 20px 0; }
            .winner { 
                background-color: #fcf8e3; 
                border: 1px solid #faebcc; 
                padding: 15px; 
                border-radius: 5px;
                margin: 20px 0;
            }
            table { 
                border-collapse: collapse; 
                width: 100%; 
                margin: 20px 0;
            }
            th, td { 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: center; 
            }
            th { 
                background-color: #f2f2f2; 
                font-weight: bold; 
            }
            .best { 
                font-weight: bold; 
                color: #5cb85c;
            }
        </style>
    </head>
    <body>
        <h1>Model Comparison Results</h1>
        
        <div class="winner">
            <h2>Best Model: """ + best_model + """</h2>
            <p>Accuracy: """ + f"{accuracies[best_model_idx]:.4f}" + """</p>
    """
    
    if aucs:
        html_content += "<p>ROC AUC: " + f"{aucs[best_model_idx]:.4f}" + "</p>"
    
    html_content += """
        </div>
        
        <h2>Performance Metrics Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
    """
    
    if aucs:
        html_content += "<th>ROC AUC</th>"
    
    html_content += "</tr>"
    
    # Add rows for each model
    for i, model in enumerate(model_names):
        is_best = (i == best_model_idx)
        html_content += f"""
        <tr>
            <td>{"<strong>" if is_best else ""}{model}{"</strong>" if is_best else ""}</td>
            <td class="{'best' if is_best else ''}">{accuracies[i]:.4f}</td>
        """
        
        if aucs:
            is_best_auc = (i == aucs.index(max(aucs)))
            html_content += f'<td class="{"best" if is_best_auc else ""}">{aucs[i]:.4f}</td>'
        
        html_content += "</tr>"
    
    html_content += """
        </table>
        
        <div class="comparison-container">
            <h2>Accuracy Comparison</h2>
            <img src="data:image/png;base64,""" + accuracy_comparison_image + """" alt="Accuracy Comparison">
        </div>
    """
    
    if auc_comparison_image:
        html_content += """
        <div class="comparison-container">
            <h2>ROC AUC Comparison</h2>
            <img src="data:image/png;base64,""" + auc_comparison_image + """" alt="AUC Comparison">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML to output
    with open(comparison_result.path, 'w') as f:
        f.write(html_content)
    
    # Save best model info
    best_model_info.metadata.update({
        'best_model': best_model,
        'accuracy': accuracies[best_model_idx],
        'creation_time': str(datetime.datetime.now()),
    })
    
    if aucs:
        best_model_info.metadata['roc_auc'] = aucs[best_model_idx]
    
    # Add comparison data to metadata
    for i, model_name in enumerate(model_names):
        best_model_info.metadata[f"{model_name.lower().replace(' ', '_')}_accuracy"] = accuracies[i]
        if aucs:
            best_model_info.metadata[f"{model_name.lower().replace(' ', '_')}_auc"] = aucs[i]
```



## Conclusion

This guide provides the foundation for using Kubeflow in HumAIne. By following this guide, we ensure consistency, reproducibility, and traceability across all our machine learning workflows.

### Key Takeaways

1. **Client Authentication**: Use the provided `KFPClientManager` to authenticate with Kubeflow
2. **Component Design**: Create modular, well-documented components with clear input/output definitions
3. **Pipeline Organization**: Structure pipelines with a logical flow from data loading to model evaluation
4. **Metadata Tracking**: Add consistent metadata to all artifacts for traceability and reproducibility
5. **Results Visualization**: Use HTML artifacts to create rich visualizations of model performance
6. **Experiment Management**: Parameterize pipelines to run and compare different experiments
7. **Standardization**: Follow the common metadata standards for consistency across pilots

### Next Steps

After following this guide, data scientists should be able to:

1. Connect to the project's Kubeflow instance
2. Create custom components for their specific tasks
3. Build pipelines that orchestrate these components
4. Add appropriate metadata to all artifacts
5. Generate visualizations of their results
6. Run experiments with different parameters
7. Compare and analyze model performance

For specific implementation details, refer to the `diabetes_demo.py` example which demonstrates a complete pipeline with multiple components, proper metadata handling, and comprehensive evaluation.

---

**Note**: Kubeflow credentials will be provided by HUA. Please keep these credentials secure and do not share them outside.

