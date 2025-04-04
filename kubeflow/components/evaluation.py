import kfp
from kfp import dsl
from kfp.dsl import Dataset, Model, ClassificationMetrics, Input, Output
import pandas as pd
import numpy as np
import joblib
import os
import json
import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc

@dsl.component
def evaluate_model(
    test_data: Input[Dataset],
    model: Input[Model],
    metrics: Output[ClassificationMetrics]
):
    """Evaluates the model on test data and outputs evaluation metrics."""
    
    # Read the test data
    df = pd.read_csv(test_data.path)
    
    # Extract features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Load the model
    clf = joblib.load(model.path)
    
    # Make predictions
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Prepare metrics data
    metrics_data = {
        'confusion_matrix': conf_matrix,
        'roc_auc': roc_auc
    }
    
    # Save ROC curve data
    roc_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }
    
    # Save metrics to file
    os.makedirs(os.path.dirname(metrics.path), exist_ok=True)
    with open(metrics.path, 'w') as f:
        json.dump({
            'metrics': metrics_data,
            'roc_data': roc_data
        }, f)
    
    # Set metadata on the metrics artifact
    metrics.metadata['confusion_matrix'] = json.dumps(conf_matrix)
    metrics.metadata['roc_auc'] = roc_auc
    metrics.metadata['accuracy'] = (conf_matrix[0][0] + conf_matrix[1][1]) / sum(map(sum, conf_matrix))
    metrics.metadata['evaluation_timestamp'] = str(datetime.datetime.now())
    metrics.metadata['model_uri'] = model.uri
    metrics.metadata['test_dataset_rows'] = len(df) 