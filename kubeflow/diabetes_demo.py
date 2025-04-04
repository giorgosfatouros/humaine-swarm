import kfp
import kfp.dsl as dsl
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
import os
from utils.helper_functions import get_kubeflow_client, get_kubeflow_old_client
# 1. DATA LOADING COMPONENT
@dsl.component(
    packages_to_install=["numpy", "scikit-learn", "joblib"]
)
def load_data(x_path: dsl.OutputPath(), 
              y_path: dsl.OutputPath(), 
              feature_names_path: dsl.OutputPath(), 
              metadata_path: dsl.OutputPath()):
    import numpy as np
    from sklearn.datasets import load_diabetes
    import joblib
    import json
    
    # Load diabetes dataset
    diabetes = load_diabetes()
    X = diabetes.data
    
    # Convert to binary classification (above/below median)
    y = (diabetes.target > np.median(diabetes.target)).astype(int)
    
    # Save feature names for later use
    feature_names = diabetes.feature_names
    
    # Save data to files - use direct paths for OutputPath
    joblib.dump(X, x_path)
    joblib.dump(y, y_path)
    joblib.dump(feature_names, feature_names_path)
    
    # Generate dataset metadata
    metadata = {
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "class_distribution": np.bincount(y).tolist(),
        "feature_names": list(feature_names)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

# 2. DATA SPLITTING COMPONENT
@dsl.component(
    packages_to_install=["numpy", "scikit-learn", "joblib"]
)
def split_data(x_path: dsl.InputPath(), 
               y_path: dsl.InputPath(), 
               x_train_path: dsl.OutputPath(),
               x_test_path: dsl.OutputPath(),
               y_train_path: dsl.OutputPath(),
               y_test_path: dsl.OutputPath(),
               split_info_path: dsl.OutputPath(),
               test_size: float = 0.3, 
               random_state: int = 42):
    import joblib
    import numpy as np
    import json
    from sklearn.model_selection import train_test_split
    
    # Load the data
    X = joblib.load(x_path)
    y = joblib.load(y_path)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save the splits
    joblib.dump(X_train, x_train_path)
    joblib.dump(X_test, x_test_path)
    joblib.dump(y_train, y_train_path)
    joblib.dump(y_test, y_test_path)
    
    # Convert numpy types to native Python types for JSON serialization
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    
    # Convert numpy arrays to regular Python lists/types
    train_class_distribution = {int(k): int(v) for k, v in zip(train_unique, train_counts)}
    test_class_distribution = {int(k): int(v) for k, v in zip(test_unique, test_counts)}
    
    split_info = {
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_class_distribution": train_class_distribution,
        "test_class_distribution": test_class_distribution
    }
    
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f)

# 3. PREPROCESSING COMPONENT
@dsl.component(
    packages_to_install=["scikit-learn", "joblib"]
)
def preprocess_data(x_train_path: dsl.InputPath(), 
                    x_test_path: dsl.InputPath(),
                    x_train_scaled_path: dsl.OutputPath(),
                    x_test_scaled_path: dsl.OutputPath(),
                    scaler_path: dsl.OutputPath()):
    import joblib
    from sklearn.preprocessing import StandardScaler
    
    # Load training and test data
    X_train = joblib.load(x_train_path)
    X_test = joblib.load(x_test_path)
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the preprocessed data and the scaler
    joblib.dump(X_train_scaled, x_train_scaled_path)
    joblib.dump(X_test_scaled, x_test_scaled_path)
    joblib.dump(scaler, scaler_path)

# 4. MODEL TRAINING COMPONENTS (One for each model type)
@dsl.component(
    packages_to_install=["scikit-learn", "joblib"]
)
def train_decision_tree(x_train_path: dsl.InputPath(), 
                        y_train_path: dsl.InputPath(), 
                        model_path: dsl.OutputPath(),
                        max_depth: int = 5, 
                        random_state: int = 42):
    import joblib
    from sklearn.tree import DecisionTreeClassifier
    
    # Load training data
    X_train = joblib.load(x_train_path)
    y_train = joblib.load(y_train_path)
    
    # Train the model
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, model_path)

@dsl.component(
    packages_to_install=["scikit-learn", "joblib"]
)
def train_random_forest(x_train_path: dsl.InputPath(), 
                       y_train_path: dsl.InputPath(), 
                       model_path: dsl.OutputPath(),
                       n_estimators: int = 100, 
                       random_state: int = 42):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    # Load training data
    X_train = joblib.load(x_train_path)
    y_train = joblib.load(y_train_path)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, model_path)

@dsl.component(
    packages_to_install=["scikit-learn", "joblib"]
)
def train_svm(x_train_path: dsl.InputPath(), 
             y_train_path: dsl.InputPath(), 
             model_path: dsl.OutputPath(),
             C: float = 1.0, 
             kernel: str = 'rbf', 
             random_state: int = 42):
    import joblib
    from sklearn.svm import SVC
    
    # Load training data
    X_train = joblib.load(x_train_path)
    y_train = joblib.load(y_train_path)
    
    # Train the model
    model = SVC(C=C, kernel=kernel, random_state=random_state, probability=True)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, model_path)

# 5. MODEL EVALUATION COMPONENT
@dsl.component(
    packages_to_install=["numpy", "scikit-learn", "joblib", "matplotlib"]
)
def evaluate_model(model_path: dsl.InputPath(), 
                  x_test_path: dsl.InputPath(), 
                  y_test_path: dsl.InputPath(), 
                  feature_names_path: dsl.InputPath(), 
                  model_name: str,
                  metrics_path: dsl.OutputPath(),
                  plots_path: dsl.OutputPath()):
    import joblib
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt
    import json
    import base64
    from io import BytesIO
    
    # Load model, test data, and feature names
    model = joblib.load(model_path)
    X_test = joblib.load(x_test_path)
    y_test = joblib.load(y_test_path)
    feature_names = joblib.load(feature_names_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
    # Store metrics
    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "classification_report": class_report,
        "confusion_matrix": conf_matrix
    }
    
    # Generate plots and convert to base64 strings
    plots = {}
    
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
    
    # Save the confusion matrix plot
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    conf_matrix_image = base64.b64encode(buffer.read()).decode('utf-8')
    plots["confusion_matrix"] = conf_matrix_image
    plt.close()
    
    # Create ROC curve if probability predictions are available
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
        plots["roc_curve"] = roc_image
        
        # Add AUC to metrics
        metrics["roc_auc"] = float(roc_auc)
        plt.close()
    
    # For tree-based models, create feature importance plot
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[indices])
        plt.title(f'Feature Importance - {model_name}')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        importance_image = base64.b64encode(buffer.read()).decode('utf-8')
        plots["feature_importance"] = importance_image
        
        # Add feature importances to metrics
        metrics["feature_importances"] = dict(zip([feature_names[i] for i in indices], importances[indices].tolist()))
        plt.close()
    
    # Save the metrics and plots
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    with open(plots_path, 'w') as f:
        json.dump(plots, f)

# 6. MODEL COMPARISON COMPONENT
@dsl.component(
    packages_to_install=["matplotlib"]
)
def compare_models(dt_metrics_path: dsl.InputPath(), 
                  rf_metrics_path: dsl.InputPath(), 
                  svm_metrics_path: dsl.InputPath(),
                  comparison_path: dsl.OutputPath()):
    import json
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    
    # Load metrics for each model
    with open(dt_metrics_path, 'r') as f:
        dt_metrics = json.load(f)
    
    with open(rf_metrics_path, 'r') as f:
        rf_metrics = json.load(f)
    
    with open(svm_metrics_path, 'r') as f:
        svm_metrics = json.load(f)
    
    # Combine metrics
    models = [dt_metrics, rf_metrics, svm_metrics]
    model_names = [model["model_name"] for model in models]
    accuracies = [model["accuracy"] for model in models]
    
    # Find the best model
    best_model_idx = accuracies.index(max(accuracies))
    best_model = model_names[best_model_idx]
    
    # Create comparison plot
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
    comparison_image = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # Prepare comparison results
    comparison_results = {
        "model_names": model_names,
        "accuracies": accuracies,
        "best_model": best_model,
        "best_accuracy": accuracies[best_model_idx],
        "comparison_plot": comparison_image
    }
    
    # Save comparison results
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f)

# DEFINE THE PIPELINE
@dsl.pipeline(
    name='Diabetes Classification Pipeline',
    description='A demonstration pipeline for diabetes classification using multiple models',
    # pipeline_root='gs://your-bucket/pipeline_root'  # Optional but recommended
)
def diabetes_classification_pipeline(
    test_size: float = 0.3,
    random_state: int = 42,
    dt_max_depth: int = 5,
    rf_n_estimators: int = 100,
    svm_c: float = 1.0,
    svm_kernel: str = 'rbf'
):
    # Load the data
    load_data_task = load_data()
    
    # Split the data
    split_data_task = split_data(
        x_path=load_data_task.outputs["x_path"],
        y_path=load_data_task.outputs["y_path"],
        test_size=test_size,
        random_state=random_state
    )
    
    # Preprocess the data
    preprocess_task = preprocess_data(
        x_train_path=split_data_task.outputs["x_train_path"],
        x_test_path=split_data_task.outputs["x_test_path"]
    )
    
    # Train the models
    dt_train_task = train_decision_tree(
        x_train_path=preprocess_task.outputs["x_train_scaled_path"],
        y_train_path=split_data_task.outputs["y_train_path"],
        max_depth=dt_max_depth,
        random_state=random_state
    )
    
    rf_train_task = train_random_forest(
        x_train_path=preprocess_task.outputs["x_train_scaled_path"],
        y_train_path=split_data_task.outputs["y_train_path"],
        n_estimators=rf_n_estimators,
        random_state=random_state
    )
    
    svm_train_task = train_svm(
        x_train_path=preprocess_task.outputs["x_train_scaled_path"],
        y_train_path=split_data_task.outputs["y_train_path"],
        C=svm_c,
        kernel=svm_kernel,
        random_state=random_state
    )
    
    # Evaluate the models
    dt_eval_task = evaluate_model(
        model_path=dt_train_task.outputs["model_path"],
        x_test_path=preprocess_task.outputs["x_test_scaled_path"],
        y_test_path=split_data_task.outputs["y_test_path"],
        feature_names_path=load_data_task.outputs["feature_names_path"],
        model_name="Decision Tree"
    )
    
    rf_eval_task = evaluate_model(
        model_path=rf_train_task.outputs["model_path"],
        x_test_path=preprocess_task.outputs["x_test_scaled_path"],
        y_test_path=split_data_task.outputs["y_test_path"],
        feature_names_path=load_data_task.outputs["feature_names_path"],
        model_name="Random Forest"
    )
    
    svm_eval_task = evaluate_model(
        model_path=svm_train_task.outputs["model_path"],
        x_test_path=preprocess_task.outputs["x_test_scaled_path"],
        y_test_path=split_data_task.outputs["y_test_path"],
        feature_names_path=load_data_task.outputs["feature_names_path"],
        model_name="Support Vector Machine"
    )
    
    # Compare the models
    compare_models_task = compare_models(
        dt_metrics_path=dt_eval_task.outputs["metrics_path"],
        rf_metrics_path=rf_eval_task.outputs["metrics_path"],
        svm_metrics_path=svm_eval_task.outputs["metrics_path"]
    )

# COMPILE THE PIPELINE
if __name__ == "__main__":
    # Compile the pipeline to a YAML file
    kfp.compiler.Compiler().compile(
        pipeline_func=diabetes_classification_pipeline,
        package_path='diabetes_classification_pipeline.yaml'
    )
    
    print("Pipeline compiled successfully. You can now upload 'diabetes_classification_pipeline.yaml' to the Kubeflow Pipelines UI.")
    
    # HOST='http://hua-kubeflow.ddns.net/'
    # USERNAME = "user@example.com"
    # PASSWORD = "2LZHseTdrLFFvx"
    # NAMESPACE = "kubeflow-user-example-com"
    # # Optional: Run the pipeline locally (requires a Kubeflow connection)
    # client = get_kubeflow_old_client(
    #     host=HOST,
    #     username=USERNAME,
    #     password=PASSWORD,
    #     namespace=NAMESPACE
    # )
    client = get_kubeflow_client()

    client.create_run_from_pipeline_func(
        diabetes_classification_pipeline,
        arguments={
            'test_size': 0.3,
            'random_state': 42,
            'dt_max_depth': 5,
            'rf_n_estimators': 100,
            'svm_c': 1.0,
            'svm_kernel': 'rbf'
        },
        # namespace='innvoacts'
    )