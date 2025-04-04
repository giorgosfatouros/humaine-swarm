import kfp
from kfp import dsl
from kfp.dsl import Dataset, Model, ClassificationMetrics
import datetime

from kubeflow.components.preprocessing import preprocess_data
from kubeflow.components.training import train_model  
from kubeflow.components.evaluation import evaluate_model

@dsl.pipeline(
    name="ML Training Pipeline with Artifacts",
    description="A pipeline that demonstrates proper artifact metadata handling"
)
def ml_pipeline(raw_data_path: str) -> Model:
    """ML pipeline with proper artifact handling."""
    
    # Create a dataset artifact for the input
    raw_data = Dataset(uri=raw_data_path, metadata={
        'source': 'raw_input',
        'timestamp': str(datetime.datetime.now()),
        'description': 'Raw input dataset'
    })
    
    # Preprocess data
    preprocess_task = preprocess_data(raw_data=raw_data)
    
    # Train model
    hyperparameters = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    train_task = train_model(
        training_data=preprocess_task.outputs['processed_data'],
        hyperparameters=hyperparameters
    )
    
    # Evaluate model
    evaluate_task = evaluate_model(
        test_data=preprocess_task.outputs['processed_data'],
        model=train_task.outputs['model']
    )
    
    # Return the model as the pipeline output
    return train_task.outputs['model'] 