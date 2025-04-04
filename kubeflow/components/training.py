import kfp
from kfp import dsl
from kfp.dsl import Dataset, Model, Metrics, Input, Output
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import json
import datetime

@dsl.component
def train_model(
    training_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    hyperparameters: dict = {"n_estimators": 100, "max_depth": 10}
):
    """Trains a model on the input dataset and outputs the model and metrics."""
    
    # Read the training data
    df = pd.read_csv(training_data.path)
    
    # Extract features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Initialize and train the model
    clf = RandomForestClassifier(**hyperparameters)
    clf.fit(X, y)
    
    # Evaluate the model
    accuracy = clf.score(X, y)
    feature_importance = clf.feature_importances_.tolist()
    
    # Save the model
    os.makedirs(os.path.dirname(model.path), exist_ok=True)
    joblib.dump(clf, model.path)
    
    # Save metrics
    metrics_data = {
        'accuracy': accuracy,
        'feature_importance': feature_importance
    }
    
    os.makedirs(os.path.dirname(metrics.path), exist_ok=True)
    with open(metrics.path, 'w') as f:
        json.dump(metrics_data, f)
    
    # Set metadata on the model artifact
    model.metadata['framework'] = 'scikit-learn'
    model.metadata['model_type'] = 'RandomForestClassifier'
    model.metadata['hyperparameters'] = json.dumps(hyperparameters)
    model.metadata['training_dataset_rows'] = len(df)
    model.metadata['training_dataset_features'] = len(X.columns)
    model.metadata['training_accuracy'] = accuracy
    model.metadata['training_timestamp'] = str(datetime.datetime.now())
    
    # Set metadata on the metrics artifact
    metrics.metadata.update(metrics_data)
    metrics.metadata['timestamp'] = str(datetime.datetime.now()) 