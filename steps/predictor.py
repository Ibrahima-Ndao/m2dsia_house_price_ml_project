import json
import numpy as np
import pandas as pd
import pickle
import logging
import os


def predictor(model_path: str, input_data: str) -> np.ndarray:
    """Run an inference request against a trained model.

    Args:
        model_path (str): Path to the saved model file.
        input_data (str): The input data as a JSON string.

    Returns:
        np.ndarray: The model's prediction.
    """
    logger = logging.getLogger("predictor")
    logger.info("Starting prediction step.")
    
    try:
        # Load the trained model
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        with open(model_path, 'rb') as f:
            trained_model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")

        # Load the input data from JSON string
        data = json.loads(input_data)

        # Extract the actual data and expected columns
        data.pop("columns", None)  # Remove 'columns' if it's present
        data.pop("index", None)  # Remove 'index' if it's present

        # Define the columns the model expects
        expected_columns = [
            'date',
            'price',
            'bedrooms',
            'bathrooms',
            'sqft_living',
            'sqft_lot',
            'floors',
            'waterfront',
            'view',
            'condition',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'street',
            'city',
            'statezip',
            'country'
        ]

        # Convert the data into a DataFrame with the correct columns
        df = pd.DataFrame(data["data"], columns=expected_columns)
        logger.info(f"Input data converted to DataFrame with shape: {df.shape}")

        # Run the prediction
        prediction = trained_model.predict(df)
        logger.info("Prediction completed successfully.")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise