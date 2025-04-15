import os
import kfp
import kfp.dsl as dsl
from kfp.dsl import Dataset, Input, Output, Model, Metrics, ClassificationMetrics, HTML
from utils.helper_functions import get_kubeflow_client
import re

# Add utility component for MinIO authentication
@dsl.component(
    packages_to_install=["minio==7.1.15", "requests"]
)
def authenticate_minio(
    keycloak_url: str,
    keycloak_client_id: str, 
    keycloak_client_secret: str,
    keycloak_username: str,
    keycloak_password: str,
    minio_endpoint: str,
    minio_credentials: Output[Dataset],
    duration_seconds: int = 43200
):
    import json
    import requests
    import datetime
    import xml.etree.ElementTree as ET
    
    # Authenticate with Keycloak to get access token
    payload = {
        'client_id': keycloak_client_id,
        'client_secret': keycloak_client_secret,
        'username': keycloak_username,
        'password': keycloak_password,
        'grant_type': 'password'
    }
    
    # Clean up any whitespace in the URL
    keycloak_url = keycloak_url.strip()
    
    response = requests.post(keycloak_url, data=payload)
    if response.status_code != 200:
        raise Exception(f"Failed to authenticate with Keycloak: {response.text}")
    
    access_token = response.json()['access_token']
    
    # Use the access token to get temporary MinIO credentials
    # Using Method 2 from the test script which worked
    sts_endpoint = f"https://{minio_endpoint}"
    
    sts_payload = {
        'Action': 'AssumeRoleWithWebIdentity', 
        'Version': '2011-06-15',
        'DurationSeconds': str(duration_seconds),
        'WebIdentityToken': access_token
    }
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    sts_response = requests.post(sts_endpoint, data=sts_payload, headers=headers)
    
    if sts_response.status_code != 200:
        raise Exception(f"Failed to get MinIO credentials: {sts_response.text}")
    
    # Parse XML response to get credentials
    try:
        # Using Method A from the test script which worked
        ET.register_namespace('', "https://sts.amazonaws.com/doc/2011-06-15/")
        root = ET.fromstring(sts_response.text)
        ns = {'sts': 'https://sts.amazonaws.com/doc/2011-06-15/'}
        credentials = root.find(".//sts:Credentials", ns)
        
        if credentials is not None:
            access_key = credentials.find("sts:AccessKeyId", ns).text
            secret_key = credentials.find("sts:SecretAccessKey", ns).text
            session_token = credentials.find("sts:SessionToken", ns).text
        else:
            # Fall back to direct string search
            access_key = None
            secret_key = None
            session_token = None
            
            # Look for AccessKeyId
            access_key_match = sts_response.text.find("<AccessKeyId>")
            if access_key_match != -1:
                access_key_end = sts_response.text.find("</AccessKeyId>", access_key_match)
                if access_key_end != -1:
                    access_key = sts_response.text[access_key_match + len("<AccessKeyId>"):access_key_end]
            
            # Look for SecretAccessKey
            secret_key_match = sts_response.text.find("<SecretAccessKey>")
            if secret_key_match != -1:
                secret_key_end = sts_response.text.find("</SecretAccessKey>", secret_key_match)
                if secret_key_end != -1:
                    secret_key = sts_response.text[secret_key_match + len("<SecretAccessKey>"):secret_key_end]
            
            # Look for SessionToken
            session_token_match = sts_response.text.find("<SessionToken>")
            if session_token_match != -1:
                session_token_end = sts_response.text.find("</SessionToken>", session_token_match)
                if session_token_end != -1:
                    session_token = sts_response.text[session_token_match + len("<SessionToken>"):session_token_end]
            
            if not (access_key and secret_key and session_token):
                raise Exception("Could not extract credentials using string search")
        
        # Use current time for expiration if not found
        expiration = datetime.datetime.now() + datetime.timedelta(seconds=duration_seconds)
        expiration_str = expiration.isoformat()
        
    except Exception as e:
        raise Exception(f"Failed to parse STS response: {e}")
    
    # Save credentials to output
    minio_creds = {
        'access_key': access_key,
        'secret_key': secret_key,
        'session_token': session_token,
        'expiration': expiration_str,
        'endpoint': minio_endpoint
    }
    
    with open(minio_credentials.path, 'w') as f:
        json.dump(minio_creds, f)
    
    minio_credentials.metadata.update({
        'creation_time': str(datetime.datetime.now()),
        'expiration_time': expiration_str,
        'endpoint': minio_endpoint
    })
    
    print("MinIO credentials saved successfully")

# Update save_to_minio component to not create separate metadata files
@dsl.component(
    packages_to_install=["minio==7.1.15", "requests", "joblib", "pillow"]
)
def save_to_minio(
    minio_credentials: Input[Dataset],
    artifact: Input[dsl.Artifact],
    bucket_name: str,
    pipeline_name: str,
    run_id: str,
    artifact_type: str,
    artifact_name: str,
    fail_on_missing: bool = False
):
    import json
    import os
    import re
    import sys
    from minio import Minio
    from minio.commonconfig import Tags
    import joblib
    import time
    
    def sanitize_s3_path(path):
        """Sanitize a path for use as an S3 object key"""
        # Clean up double slashes
        while '//' in path:
            path = path.replace('//', '/')
        
        # Remove leading and trailing slashes
        path = path.strip('/')
        
        # Ensure the path doesn't contain invalid characters
        path = re.sub(r'[\x00-\x1F\x7F]', '', path)  # Remove control characters
        path = re.sub(r'^\s+|\s+$', '', path)  # Remove leading/trailing whitespace
        
        # Sanitize path parts (but keep slashes)
        parts = path.split('/')
        sanitized_parts = []
        for part in parts:
            # Keep only allowed characters for S3 keys
            sanitized = re.sub(r'[^a-zA-Z0-9\._\-+]', '_', part)
            sanitized_parts.append(sanitized)
        
        return '/'.join(sanitized_parts)
    
    # Build the object path using the provided components
    # Sanitize each component individually
    safe_pipeline_name = re.sub(r'[^a-zA-Z0-9\._\-]', '-', pipeline_name)
    safe_run_id = re.sub(r'[^a-zA-Z0-9\._\-]', '-', run_id)
    safe_artifact_type = re.sub(r'[^a-zA-Z0-9\._\-]', '-', artifact_type)
    safe_artifact_name = re.sub(r'[^a-zA-Z0-9\._\-]', '-', artifact_name)
    
    # Construct the path
    object_path = f"kubeflow/{safe_pipeline_name}/{safe_run_id}/{safe_artifact_type}/{safe_artifact_name}"
    
    print(f"Saving to MinIO: {object_path}")
    
    # Load MinIO credentials
    with open(minio_credentials.path, 'r') as f:
        creds = json.load(f)
    
    # Check if artifact path exists
    print(f"Checking artifact path: {artifact.path}")
    
    if not os.path.exists(artifact.path):
        error_msg = f"Artifact path does not exist: {artifact.path}"
        print(f"ERROR: {error_msg}")
        
        if fail_on_missing:
            # Fail the component with a clear error message
            raise FileNotFoundError(f"Required artifact missing: {artifact.path}")
        else:
            print(f"WARNING: Creating placeholder file instead of failing")
            # Create placeholder with warning content that this is a placeholder
            
            # Determine file type based on extension and create appropriate placeholder
            ext = os.path.splitext(artifact.path)[1].lower()
            if ext in ['.json', '.yaml', '.yml', '.txt']:
                with open(artifact.path, 'w') as f:
                    f.write('{"error": "Artifact file was not properly generated"}')
            elif ext in ['.png', '.jpg', '.jpeg', '.gif']:
                # Create a tiny blank image
                try:
                    from PIL import Image
                    img = Image.new('RGB', (100, 100), color = 'white')
                    img.save(artifact.path)
                except ImportError:
                    with open(artifact.path, 'wb') as f:
                        f.write(b'')
            else:
                # For other files (like .joblib), create an empty binary file
                with open(artifact.path, 'wb') as f:
                    f.write(b'')
    
    # Setup MinIO client
    client = Minio(
        creds['endpoint'],
        access_key=creds['access_key'],
        secret_key=creds['secret_key'],
        session_token=creds['session_token'],
        secure=True
    )
    
    # Create bucket if it doesn't exist
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"Created bucket: {bucket_name}")
    
    # Sanitize the path to ensure S3 compatibility
    full_path = sanitize_s3_path(object_path)
    print(f"S3 path after sanitization: {full_path}")
    
    # Implement retry logic for transient failures
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            client.fput_object(bucket_name, full_path, artifact.path)
            # Verify upload worked
            stat = client.stat_object(bucket_name, full_path)
            print(f"Upload verified: size={stat.size} bytes, etag={stat.etag}")
            break
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"All retry attempts failed")
                raise
    
    # Add metadata 
    if hasattr(artifact, 'metadata') and artifact.metadata:
        try:
            tags = Tags.new_object_tags()
            tag_count = 0
            
            for key, value in artifact.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    # Use less restrictive regex for keys
                    valid_key = re.sub(r'[^a-zA-Z0-9\-\.\_\:\@]', '-', str(key))[:128]
                    
                    # Use less restrictive regex for values
                    str_value = str(value)
                    if len(str_value) > 250:
                        str_value = str_value[:250]
                    
                    # Allow more characters in values, just remove control chars
                    str_value = re.sub(r'[\x00-\x1F\x7F]', '', str_value)
                    
                    # Only add if both key and value are valid
                    if valid_key and str_value and tag_count < 10:  # S3 has 10 tag limit
                        tags[valid_key] = str_value
                        tag_count += 1
            
            if tags:
                client.set_object_tags(bucket_name, full_path, tags)
                print(f"Added {tag_count} metadata tags to the object")
        except Exception as e:
            print(f"Warning: Could not set object tags: {e}")
    
    print(f"Successfully uploaded {artifact.path} to s3://{bucket_name}/{full_path}")

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

# 4. MODEL TRAINING COMPONENT - SVM 
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
    import datetime
    import json
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
            'random_state': random_state
        }),
        'training_dataset_size': X_train.shape[0],
        'feature_count': X_train.shape[1],
        'description': 'Support Vector Machine classifier for diabetes prediction'
    })

# 5. MODEL EVALUATION COMPONENT
@dsl.component(
    packages_to_install=["scikit-learn", "numpy", "pandas", "matplotlib", "joblib"]
)
def evaluate_model(
    model: Input[Model],
    test_data: Input[Dataset],
    feature_names: Input[Dataset],
    model_name: str,
    metrics: Output[Metrics],
    confusion_matrix_plot: Output[HTML],
    roc_curve_plot: Output[HTML],
    feature_importance_plot: Output[HTML]
):
    import joblib
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import base64
    from io import BytesIO
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
    import os
    
    # Load model and test data
    clf = joblib.load(model.path)
    test_data_array = joblib.load(test_data.path)
    feature_names_list = joblib.load(feature_names.path)
    
    # Split features and target
    X_test = test_data_array[:, :-1]
    y_test = test_data_array[:, -1]
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = None
    
    # Try to get probability predictions, but handle models that don't support predict_proba
    try:
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
    except:
        print(f"Model {model_name} doesn't support predict_proba")
        # Use a simple placeholder probability (0.7 for positive predictions, 0.3 for negative)
        y_pred_proba = np.where(y_pred == 1, 0.7, 0.3)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create metrics dictionary
    metrics_dict = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(roc_auc),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Save metrics to json
    print(f"Writing metrics to: {metrics.path}")
    os.makedirs(os.path.dirname(metrics.path), exist_ok=True)
    with open(metrics.path, 'w') as f:
        json.dump(metrics_dict, f)
    
    # Record metrics for Kubeflow UI
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("precision", precision)
    metrics.log_metric("recall", recall)
    metrics.log_metric("f1", f1)
    metrics.log_metric("auc", roc_auc)
    
    # Helper function to convert matplotlib figure to HTML
    def fig_to_html(fig, title):
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Create HTML with embedded image
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    text-align: center;
                }}
                .plot-container {{
                    margin: 0 auto;
                    max-width: 800px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                h2 {{
                    color: #333366;
                }}
            </style>
        </head>
        <body>
            <div class="plot-container">
                <h2>{title}</h2>
                <img src="data:image/png;base64,{img_str}" alt="{title}">
            </div>
        </body>
        </html>
        """
        return html
    
    # Generate confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    # Convert to HTML and save
    cm_html = fig_to_html(plt.gcf(), f"Confusion Matrix - {model_name}")
    os.makedirs(os.path.dirname(confusion_matrix_plot.path), exist_ok=True)
    with open(confusion_matrix_plot.path, 'w') as f:
        f.write(cm_html)
    print(f"Saved confusion matrix as HTML to {confusion_matrix_plot.path}")
    plt.close()
    
    # Generate ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Convert to HTML and save
    roc_html = fig_to_html(plt.gcf(), f"ROC Curve - {model_name}")
    os.makedirs(os.path.dirname(roc_curve_plot.path), exist_ok=True)
    with open(roc_curve_plot.path, 'w') as f:
        f.write(roc_html)
    print(f"Saved ROC curve as HTML to {roc_curve_plot.path}")
    plt.close()
    
    # Generate feature importance plot if the model supports it
    plt.figure(figsize=(10, 6))
    title = ""
    try:
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.bar(range(X_test.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_test.shape[1]), [feature_names_list[i] for i in indices], rotation=90)
            plt.title(f'Feature Importances - {model_name}')
            title = f'Feature Importances - {model_name}'
            plt.tight_layout()
        elif hasattr(clf, 'coef_'):
            coefs = clf.coef_[0]
            indices = np.argsort(np.abs(coefs))[::-1]
            plt.bar(range(X_test.shape[1]), coefs[indices], align='center')
            plt.xticks(range(X_test.shape[1]), [feature_names_list[i] for i in indices], rotation=90)
            plt.title(f'Feature Coefficients - {model_name}')
            title = f'Feature Coefficients - {model_name}'
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, "Feature importance not available for this model type",
                    horizontalalignment='center', verticalalignment='center')
            plt.title(f'No Feature Importance Available - {model_name}')
            title = f'No Feature Importance Available - {model_name}'
    except Exception as e:
        plt.text(0.5, 0.5, f"Error obtaining feature importance: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        plt.title(f'Error Getting Feature Importance - {model_name}')
        title = f'Error Getting Feature Importance - {model_name}'
    
    # Convert to HTML and save
    importance_html = fig_to_html(plt.gcf(), title)
    os.makedirs(os.path.dirname(feature_importance_plot.path), exist_ok=True)
    with open(feature_importance_plot.path, 'w') as f:
        f.write(importance_html)
    print(f"Saved feature importance as HTML to {feature_importance_plot.path}")
    plt.close()
    
    print(f"Evaluation complete for {model_name}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")

# Add a component to create run metadata
@dsl.component(
    packages_to_install=[]
)
def create_run_metadata(
    run_id: str,
    pipeline_name: str,
    test_size: float,
    random_state: int,
    svm_C: float,
    svm_kernel: str,
    metadata_file: Output[Dataset]
):
    import json
    import datetime
    
    # Prepare run metadata
    run_metadata = {
        "run_id": run_id,
        "pipeline_name": pipeline_name,
        "creation_time": str(datetime.datetime.now()),
        "parameters": {
            "test_size": test_size,
            "random_state": random_state,
            "svm_C": svm_C,
            "svm_kernel": svm_kernel
        },
        "environment": {
            "timestamp": str(datetime.datetime.now()),
            "component": "create_run_metadata"
        }
    }
    
    # Write the metadata to the output file
    with open(metadata_file.path, 'w') as f:
        json.dump(run_metadata, f, indent=2)
    
    # Add metadata to the artifact itself
    metadata_file.metadata.update({
        "creation_time": str(datetime.datetime.now()),
        "pipeline_name": pipeline_name,
        "run_id": run_id
    })
    
    print(f"Created run metadata JSON: {metadata_file.path}")

# DEFINE THE PIPELINE WITH ONLY SVM MODEL
@dsl.pipeline(
    name='Diabetes SVM Classification Pipeline',
    description='A demonstration pipeline for diabetes classification using SVM model with artifact tracking in MinIO'
)
def diabetes_classification_pipeline(
    # Original parameters
    test_size: float = 0.3,
    random_state: int = 42,
    svm_C: float = 1.0,
    svm_kernel: str = 'rbf',
    # Added MinIO/Keycloak parameters
    keycloak_url: str = "https://keycloak.humaine-horizon.eu/realms/humaine/protocol/openid-connect/token",
    keycloak_client_id: str = "minio",
        # Hardcoded credentials instead of empty defaults
    keycloak_client_secret: str = "CJHIv1jYJfokZc73lUqwtkL12YBi69IB",
    keycloak_username: str = "g.fatouros-dev",
    keycloak_password: str = "g.fatouros-huma1ne!",
    minio_endpoint: str = "s3-minio.humaine-horizon.eu",
    minio_bucket: str = "innov-test-bucket",
    pipeline_name: str = "diabetes-svm-classification",
    run_name: str = "",  # Use run_name instead of run_id for better compatibility with KFP
):
    import datetime

    # Generate a consistent run ID
    if run_name!="":
        version_id = run_name
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        version_id = f"run-{timestamp}"

    print(f"Run ID: {version_id}")
    
    # Authenticate with MinIO
    auth_task = authenticate_minio(
        keycloak_url=keycloak_url,
        keycloak_client_id=keycloak_client_id,
        keycloak_client_secret=keycloak_client_secret,
        keycloak_username=keycloak_username,
        keycloak_password=keycloak_password,
        minio_endpoint=minio_endpoint
    )
    
    # Load the data
    load_data_task = load_data()
    
    # Save dataset metadata to MinIO with component-level path construction
    save_metadata_task = save_to_minio(
        minio_credentials=auth_task.outputs["minio_credentials"],
        artifact=load_data_task.outputs["metadata"],
        bucket_name=minio_bucket,
        pipeline_name=pipeline_name,
        run_id=version_id,
        artifact_type="metadata",
        artifact_name="dataset_metadata.json"
    )
    
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
    
    # Train SVM model
    svm_train_task = train_svm(
        train_data=preprocess_task.outputs["processed_train_data"],
        C=svm_C,
        kernel=svm_kernel,
        random_state=random_state
    )
    
    # Save SVM model to MinIO
    save_svm_model_task = save_to_minio(
        minio_credentials=auth_task.outputs["minio_credentials"],
        artifact=svm_train_task.outputs["model"],
        bucket_name=minio_bucket,
        pipeline_name=pipeline_name,
        run_id=version_id,
        artifact_type="models",
        artifact_name="svm_model.joblib"
    )
    
    # Evaluate the SVM model
    svm_eval_task = evaluate_model(
        model=svm_train_task.outputs["model"],
        test_data=preprocess_task.outputs["processed_test_data"],
        feature_names=load_data_task.outputs["feature_names"],
        model_name="Support Vector Machine"
    )
    
    # Save SVM evaluation metrics and plots to MinIO
    save_svm_metrics_task = save_to_minio(
        minio_credentials=auth_task.outputs["minio_credentials"],
        artifact=svm_eval_task.outputs["metrics"],
        bucket_name=minio_bucket,
        pipeline_name=pipeline_name,
        run_id=version_id,
        artifact_type="metrics",
        artifact_name="svm_metrics.json"
    )
    
    # Save confusion matrix plot
    save_svm_cm_plot_task = save_to_minio(
        minio_credentials=auth_task.outputs["minio_credentials"],
        artifact=svm_eval_task.outputs["confusion_matrix_plot"],
        bucket_name=minio_bucket,
        pipeline_name=pipeline_name,
        run_id=version_id,
        artifact_type="plots",
        artifact_name="svm_confusion_matrix.html"
    )
    
    # Save ROC curve plot
    save_svm_roc_plot_task = save_to_minio(
        minio_credentials=auth_task.outputs["minio_credentials"],
        artifact=svm_eval_task.outputs["roc_curve_plot"],
        bucket_name=minio_bucket,
        pipeline_name=pipeline_name,
        run_id=version_id,
        artifact_type="plots",
        artifact_name="svm_roc_curve.html"
    )
    
    # Save feature importance plot for SVM
    save_svm_importance_plot_task = save_to_minio(
        minio_credentials=auth_task.outputs["minio_credentials"],
        artifact=svm_eval_task.outputs["feature_importance_plot"],
        bucket_name=minio_bucket,
        pipeline_name=pipeline_name,
        run_id=version_id,
        artifact_type="plots",
        artifact_name="svm_feature_importance.html"
    )
    
    # Create run metadata
    create_metadata_task = create_run_metadata(
        run_id=version_id,
        pipeline_name=pipeline_name,
        test_size=test_size,
        random_state=random_state,
        svm_C=svm_C,
        svm_kernel=svm_kernel
    )
    
    # Save run metadata using save_to_minio
    save_run_metadata_task = save_to_minio(
        minio_credentials=auth_task.outputs["minio_credentials"],
        artifact=create_metadata_task.outputs["metadata_file"],
        bucket_name=minio_bucket,
        pipeline_name=pipeline_name,
        run_id=version_id,
        artifact_type="metadata",
        artifact_name="run_parameters.json"
    )

# COMPILE THE PIPELINE
if __name__ == "__main__":
    # Compile the pipeline to a YAML file
    pipeline_package_path = 'kubeflow/diabetes_svm_classification_pipeline_minio.yaml'
    
    kfp.compiler.Compiler().compile(
        pipeline_func=diabetes_classification_pipeline,
        package_path=pipeline_package_path
    )
    
    print(f"Pipeline compiled successfully to '{pipeline_package_path}'")
    
    # Get the Kubeflow client
    client = get_kubeflow_client()
    
    # Upload the pipeline
    pipeline_name = 'diabetes-svm-classification-pipeline'
    pipeline_description = 'A demonstration pipeline for diabetes classification using SVM model with artifact tracking in MinIO'
    
    uploaded_pipeline = client.upload_pipeline(
        pipeline_package_path=pipeline_package_path,
        pipeline_name=pipeline_name,
        description=pipeline_description,
        namespace="kubeflow-user-example-com"
    )
    
    print(f"Pipeline uploaded successfully: {uploaded_pipeline}")
    
    # Create a run from the uploaded pipeline
    run = client.create_run_from_pipeline_func(
        pipeline_func=diabetes_classification_pipeline,
        arguments={
            'test_size': 0.5,
            'random_state': 42,
            'svm_C': 1.0,
            'svm_kernel': 'rbf',
            # 'keycloak_client_secret': os.getenv('KEYCLOAK_CLIENT_SECRET'),
            # 'keycloak_username': os.getenv('KEYCLOAK_USERNAME'),
            # 'keycloak_password': os.getenv('KEYCLOAK_PASSWORD'),
            # 'pipeline_name': 'diabetes-svm-classification',
            # 'minio_bucket': os.getenv('MINIO_BUCKET'),
            'run_name': 'run-1'
        },
        namespace='kubeflow-user-example-com',
        experiment_name='Diabetes Classification Experiments'
    )
    
    print(f"Pipeline run created with ID: {run}")
    print(f"You can monitor the run at the Kubeflow Pipelines UI")