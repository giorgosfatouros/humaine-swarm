import kfp
from kfp import dsl
from kfp.dsl import Dataset, Input, Output
import pandas as pd
import os
import datetime

@dsl.component
def preprocess_data(
    raw_data: Input[Dataset],
    processed_data: Output[Dataset]
):
    """Preprocesses the input dataset and outputs a processed dataset."""
    
    # Read the input dataset
    df = pd.read_csv(raw_data.path)
    
    # Preprocessing steps (example)
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Calculate some statistics for metadata
    row_count = len(df)
    column_count = len(df.columns)
    missing_values = df.isna().sum().sum()
    
    # Save the processed data
    os.makedirs(os.path.dirname(processed_data.path), exist_ok=True)
    df.to_csv(processed_data.path, index=False)
    
    # Set metadata on the output artifact
    processed_data.metadata['rows'] = row_count
    processed_data.metadata['columns'] = column_count
    processed_data.metadata['missing_values'] = missing_values
    processed_data.metadata['preprocessing_timestamp'] = str(datetime.datetime.now())
    processed_data.metadata['preprocessing_steps'] = 'dropna,drop_duplicates' 