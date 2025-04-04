import kfp
import kfp.dsl as dsl
from kfp.dsl import Dataset, Input, Output, Model, Metrics, ClassificationMetrics, HTML
from utils.helper_functions import get_kubeflow_client, get_kubeflow_old_client

# 1. DATA LOADING COMPONENT
@dsl.component(
    packages_to_install=["numpy", "scikit-learn", "joblib"]
)
def load_data(dataset: Output[Dataset], 
              feature_names: Output[Dataset], 
              metadata: Output[Dataset]):
    import numpy as np
    from sklearn.datasets import load_diabetes
    import joblib
    import json
    import os
    import datetime
    
    # Load diabetes dataset
    diabetes = load_diabetes()
    X = diabetes.data
    
    # Convert to binary classification (above/below median)
    y = (diabetes.target > np.median(diabetes.target)).astype(int)
    
    # Create a combined dataset with features and target
    combined_data = np.column_stack((X, y))
    
    # Save data to local path - the artifact.path gives us the local path
    os.makedirs(os.path.dirname(dataset.path), exist_ok=True)
    joblib.dump(combined_data, dataset.path)
    
    # Save feature names for later use
    feature_names_list = list(diabetes.feature_names)
    joblib.dump(feature_names_list, feature_names.path)
    
    # Generate dataset metadata
    metadata_dict = {
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "class_distribution": np.bincount(y).tolist(),
        "feature_names": feature_names_list
    }
    
    with open(metadata.path, 'w') as f:
        json.dump(metadata_dict, f)
    
    # Add metadata to the dataset artifact
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
    
    feature_names.metadata.update({
        'creation_time': str(datetime.datetime.now()),
        'description': 'Feature names for the diabetes dataset',
    })
    
    metadata.metadata.update({
        'creation_time': str(datetime.datetime.now()),
        'description': 'Metadata about the diabetes dataset'
    })

# 2. DATA SPLITTING COMPONENT
@dsl.component(
    packages_to_install=["numpy", "scikit-learn", "joblib"]
)
def split_data(dataset: Input[Dataset], 
               train_data: Output[Dataset],
               test_data: Output[Dataset],
               split_info: Output[Dataset],
               test_size: float = 0.3, 
               random_state: int = 42):
    import joblib
    import numpy as np
    import json
    import os
    import datetime
    from sklearn.model_selection import train_test_split
    
    # Load the data
    combined_data = joblib.load(dataset.path)
    X = combined_data[:, :-1]  # All columns except the last one
    y = combined_data[:, -1]   # Last column is the target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create combined datasets again
    train_data_combined = np.column_stack((X_train, y_train))
    test_data_combined = np.column_stack((X_test, y_test))
    
    # Save the splits
    joblib.dump(train_data_combined, train_data.path)
    joblib.dump(test_data_combined, test_data.path)
    
    # Convert numpy types to native Python types for JSON serialization
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    
    # Convert numpy arrays to regular Python lists/types
    train_class_distribution = {int(k): int(v) for k, v in zip(train_unique, train_counts)}
    test_class_distribution = {int(k): int(v) for k, v in zip(test_unique, test_counts)}
    
    split_info_dict = {
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_class_distribution": train_class_distribution,
        "test_class_distribution": test_class_distribution
    }
    
    with open(split_info.path, 'w') as f:
        json.dump(split_info_dict, f)
    
    # Add metadata to artifacts
    train_data.metadata.update({
        'creation_time': str(datetime.datetime.now()),
        'description': 'Training data split',
        'samples': int(X_train.shape[0]),
        'features': int(X_train.shape[1]),
        'positive_samples': int(np.sum(y_train)),
        'negative_samples': int(np.sum(y_train == 0)),
        'split_ratio': f'{1-test_size:.2f}/{test_size:.2f}'
    })
    
    test_data.metadata.update({
        'creation_time': str(datetime.datetime.now()),
        'description': 'Test data split',
        'samples': int(X_test.shape[0]),
        'features': int(X_test.shape[1]),
        'positive_samples': int(np.sum(y_test)),
        'negative_samples': int(np.sum(y_test == 0)),
        'split_ratio': f'{1-test_size:.2f}/{test_size:.2f}'
    })
    
    split_info.metadata.update({
        'creation_time': str(datetime.datetime.now()),
        'description': 'Information about the train/test split',
        'test_size': test_size,
        'random_state': random_state
    })

# 3. PREPROCESSING COMPONENT
@dsl.component(
    packages_to_install=["scikit-learn", "joblib"]
)
def preprocess_data(train_data: Input[Dataset], 
                   test_data: Input[Dataset],
                   processed_train_data: Output[Dataset],
                   processed_test_data: Output[Dataset],
                   preprocessor: Output[Model]):
    import joblib
    import numpy as np
    import datetime
    import json
    from sklearn.preprocessing import StandardScaler
    
    # Load training and test data
    train_data_combined = joblib.load(train_data.path)
    test_data_combined = joblib.load(test_data.path)
    
    # Extract features and targets
    X_train = train_data_combined[:, :-1]
    y_train = train_data_combined[:, -1]
    X_test = test_data_combined[:, :-1]
    y_test = test_data_combined[:, -1]
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Recombine with targets
    train_data_scaled = np.column_stack((X_train_scaled, y_train))
    test_data_scaled = np.column_stack((X_test_scaled, y_test))
    
    # Save the preprocessed data and the scaler
    joblib.dump(train_data_scaled, processed_train_data.path)
    joblib.dump(test_data_scaled, processed_test_data.path)
    joblib.dump(scaler, preprocessor.path)
    
    # Add metadata
    preprocessor.metadata.update({
        'creation_time': str(datetime.datetime.now()),
        'description': 'StandardScaler for feature normalization',
        'framework': 'scikit-learn',
        'type': 'StandardScaler'
    })
    
    processed_train_data.metadata.update({
        'creation_time': str(datetime.datetime.now()),
        'description': 'Preprocessed training data',
        'preprocessing': 'StandardScaler',
        'samples': train_data_combined.shape[0]
    })
    
    processed_test_data.metadata.update({
        'creation_time': str(datetime.datetime.now()),
        'description': 'Preprocessed test data',
        'preprocessing': 'StandardScaler',
        'samples': test_data_combined.shape[0]
    })

# 4. MODEL TRAINING COMPONENTS (One for each model type)
@dsl.component(
    packages_to_install=["scikit-learn", "joblib"]
)
def train_decision_tree(train_data: Input[Dataset], 
                        model: Output[Model],
                        max_depth: int = 5, 
                        random_state: int = 42):
    import joblib
    import numpy as np
    import datetime
    import json
    from sklearn.tree import DecisionTreeClassifier
    
    # Load training data
    train_data_combined = joblib.load(train_data.path)
    X_train = train_data_combined[:, :-1]
    y_train = train_data_combined[:, -1]
    
    # Train the model
    dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt_model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(dt_model, model.path)
    
    # Add metadata
    model.metadata.update({
        'framework': 'scikit-learn',
        'model_type': 'DecisionTreeClassifier',
        'creation_time': str(datetime.datetime.now()),
        'version': '1.0',
        'hyperparameters': json.dumps({
            'max_depth': max_depth,
            'random_state': random_state
        }),
        'training_dataset_size': X_train.shape[0],
        'feature_count': X_train.shape[1],
        'description': 'Decision Tree classifier for diabetes prediction'
    })

@dsl.component(
    packages_to_install=["scikit-learn", "joblib"]
)
def train_random_forest(train_data: Input[Dataset], 
                       model: Output[Model],
                       n_estimators: int = 100, 
                       random_state: int = 42):
    import joblib
    import numpy as np
    import json
    import datetime
    from sklearn.ensemble import RandomForestClassifier
    
    # Load training data
    train_data_combined = joblib.load(train_data.path)
    X_train = train_data_combined[:, :-1]
    y_train = train_data_combined[:, -1]
    
    # Train the model
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(rf_model, model.path)
    
    # Add metadata
    model.metadata.update({
        'framework': 'scikit-learn',
        'model_type': 'RandomForestClassifier',
        'creation_time': str(datetime.datetime.now()),
        'version': '1.0',
        'hyperparameters': json.dumps({
            'n_estimators': n_estimators,
            'random_state': random_state
        }),
        'training_dataset_size': X_train.shape[0],
        'feature_count': X_train.shape[1],
        'description': 'Random Forest classifier for diabetes prediction'
    })

@dsl.component(
    packages_to_install=["scikit-learn", "joblib"]
)
def train_svm(train_data: Input[Dataset], 
             model: Output[Model],
             C: float = 1.0, 
             kernel: str = 'rbf', 
             random_state: int = 42):
    import joblib
    import numpy as np
    import json
    import datetime
    from sklearn.svm import SVC
    
    # Load training data
    train_data_combined = joblib.load(train_data.path)
    X_train = train_data_combined[:, :-1]
    y_train = train_data_combined[:, -1]
    
    # Train the model
    svm_model = SVC(C=C, kernel=kernel, random_state=random_state, probability=True)
    svm_model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(svm_model, model.path)
    
    # Add metadata
    model.metadata.update({
        'framework': 'scikit-learn',
        'model_type': 'SVC',
        'creation_time': str(datetime.datetime.now()),
        'version': '1.0',
        'hyperparameters': json.dumps({
            'C': C,
            'kernel': kernel,
            'random_state': random_state,
            'probability': True
        }),
        'training_dataset_size': X_train.shape[0],
        'feature_count': X_train.shape[1],
        'description': 'Support Vector Machine classifier for diabetes prediction'
    })

# 5. MODEL EVALUATION COMPONENT
@dsl.component(
    packages_to_install=["numpy", "scikit-learn", "joblib", "matplotlib"]
)
def evaluate_model(model: Input[Model], 
                  test_data: Input[Dataset], 
                  feature_names: Input[Dataset], 
                  model_name: str,
                  metrics: Output[Metrics],
                  confusion_matrix: Output[ClassificationMetrics],
                  evaluation_plots: Output[HTML]):
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
    
    # Print the keys of class_report to help debugging
    print("Classification report keys:", class_report.keys())
    
    # Store metrics - check if keys are strings or integers
    metrics_dict = {
        "accuracy": float(accuracy)
    }
    
    # Handle different possible key formats in classification report
    if '0' in class_report:
        # String keys
        metrics_dict.update({
            "precision_class_0": class_report['0']['precision'],
            "recall_class_0": class_report['0']['recall'],
            "f1_score_class_0": class_report['0']['f1-score'],
            "precision_class_1": class_report['1']['precision'],
            "recall_class_1": class_report['1']['recall'],
            "f1_score_class_1": class_report['1']['f1-score'],
        })
    elif 0 in class_report:
        # Integer keys
        metrics_dict.update({
            "precision_class_0": class_report[0]['precision'],
            "recall_class_0": class_report[0]['recall'],
            "f1_score_class_0": class_report[0]['f1-score'],
            "precision_class_1": class_report[1]['precision'],
            "recall_class_1": class_report[1]['recall'],
            "f1_score_class_1": class_report[1]['f1-score'],
        })
    else:
        # If neither format works, try to find the actual keys
        # This handles cases where scikit-learn might label classes differently
        class_keys = [k for k in class_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        if len(class_keys) >= 2:
            metrics_dict.update({
                "precision_class_0": class_report[class_keys[0]]['precision'],
                "recall_class_0": class_report[class_keys[0]]['recall'],
                "f1_score_class_0": class_report[class_keys[0]]['f1-score'],
                "precision_class_1": class_report[class_keys[1]]['precision'],
                "recall_class_1": class_report[class_keys[1]]['recall'],
                "f1_score_class_1": class_report[class_keys[1]]['f1-score'],
            })
    
    # Calculate ROC and AUC if probabilities are available
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        metrics_dict["roc_auc"] = float(roc_auc)
    
    # Save confusion matrix data
    with open(confusion_matrix.path, 'w') as f:
        json.dump({
            "target_names": ["Negative", "Positive"],
            "matrix": conf_matrix
        }, f)
    
    # Add confusion matrix as metadata for the ClassificationMetrics artifact
    confusion_matrix.metadata["format"] = "matrix"
    confusion_matrix.metadata["labels"] = ["Negative", "Positive"]
    confusion_matrix.metadata["matrix"] = conf_matrix
    
    # Generate HTML for plots
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
    """
    
    # Add other metrics
    for metric_name, metric_value in metrics_dict.items():
        if metric_name != "accuracy":  # Already added above
            html_content += f"""
            <div class="metric-box">
                <h3>{metric_name.replace('_', ' ').title()}</h3>
                <div class="metric-value">{metric_value:.4f}</div>
            </div>
            """
    
    html_content += """
        </div>
        
        <div class="plot-container">
            <h2>Confusion Matrix</h2>
            <img src="data:image/png;base64,{confusion_matrix_img}" alt="Confusion Matrix">
        </div>
    """
    
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
    confusion_matrix_img = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # Add to HTML
    html_content = html_content.replace("{confusion_matrix_img}", confusion_matrix_img)
    
    # Add ROC curve if available
    if y_pred_proba is not None:
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
    
    # For tree-based models, create feature importance plot
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
        
        # Add feature importances to metrics
        feature_imp_dict = dict(zip([feature_names_list[i] for i in indices], importances[indices].tolist()))
        for feature, importance in feature_imp_dict.items():
            metrics_dict[f"importance_{feature}"] = float(importance)
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML to output
    with open(evaluation_plots.path, 'w') as f:
        f.write(html_content)
    
    # Add metadata to metrics artifact
    metrics.metadata.update({
        'model_name': model_name,
        'creation_time': str(datetime.datetime.now()),
        'accuracy': float(accuracy),
        'dataset_size': len(y_test)
    })
    
    # Add all metrics to the metrics artifact
    for key, value in metrics_dict.items():
        metrics.metadata[key] = value

# 6. MODEL COMPARISON COMPONENT
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

# DEFINE THE PIPELINE
@dsl.pipeline(
    name='Diabetes Classification Pipeline',
    description='A demonstration pipeline for diabetes classification using multiple models with artifact tracking'
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
        dataset=load_data_task.outputs["dataset"],
        test_size=test_size,
        random_state=random_state
    )
    
    # Preprocess the data
    preprocess_task = preprocess_data(
        train_data=split_data_task.outputs["train_data"],
        test_data=split_data_task.outputs["test_data"]
    )
    
    # Train the models
    dt_train_task = train_decision_tree(
        train_data=preprocess_task.outputs["processed_train_data"],
        max_depth=dt_max_depth,
        random_state=random_state
    )
    
    rf_train_task = train_random_forest(
        train_data=preprocess_task.outputs["processed_train_data"],
        n_estimators=rf_n_estimators,
        random_state=random_state
    )
    
    svm_train_task = train_svm(
        train_data=preprocess_task.outputs["processed_train_data"],
        C=svm_c,
        kernel=svm_kernel,
        random_state=random_state
    )
    
    # Evaluate the models
    dt_eval_task = evaluate_model(
        model=dt_train_task.outputs["model"],
        test_data=preprocess_task.outputs["processed_test_data"],
        feature_names=load_data_task.outputs["feature_names"],
        model_name="Decision Tree"
    )
    
    rf_eval_task = evaluate_model(
        model=rf_train_task.outputs["model"],
        test_data=preprocess_task.outputs["processed_test_data"],
        feature_names=load_data_task.outputs["feature_names"],
        model_name="Random Forest"
    )
    
    svm_eval_task = evaluate_model(
        model=svm_train_task.outputs["model"],
        test_data=preprocess_task.outputs["processed_test_data"],
        feature_names=load_data_task.outputs["feature_names"],
        model_name="Support Vector Machine"
    )
    
    # Compare the models
    compare_task = compare_models(
        dt_metrics=dt_eval_task.outputs["metrics"],
        rf_metrics=rf_eval_task.outputs["metrics"],
        svm_metrics=svm_eval_task.outputs["metrics"]
    )

# COMPILE THE PIPELINE
if __name__ == "__main__":
    # Compile the pipeline to a YAML file
    kfp.compiler.Compiler().compile(
        pipeline_func=diabetes_classification_pipeline,
        package_path='diabetes_classification_pipeline.yaml'
    )
    
    print("Pipeline compiled successfully. You can now upload 'diabetes_classification_pipeline.yaml' to the Kubeflow Pipelines UI.")
    
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
    )