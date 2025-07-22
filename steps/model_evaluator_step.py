import logging
from typing import Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy


def model_evaluator_step(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[dict, float]:
    """
    Evaluates the trained model using ModelEvaluator and RegressionModelEvaluationStrategy.

    Parameters:
    trained_model (Pipeline): The trained pipeline containing the model and preprocessing steps.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels/target.

    Returns:
    Tuple[dict, float]: A dictionary containing evaluation metrics and MSE value.
    """
    logger = logging.getLogger("model_evaluator_step")
    logger.info("Starting model evaluation step.")
    
    try:
        # Ensure the inputs are of the correct type
        if not isinstance(X_test, pd.DataFrame):
            logger.error("X_test must be a pandas DataFrame.")
            raise TypeError("X_test must be a pandas DataFrame.")
        if not isinstance(y_test, pd.Series):
            logger.error("y_test must be a pandas Series.")
            raise TypeError("y_test must be a pandas Series.")

        logger.info("Applying preprocessing to the test data.")

        # Apply the preprocessing and model prediction
        X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

        # Initialize the evaluator with the regression strategy
        evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())

        # Perform the evaluation
        evaluation_metrics = evaluator.evaluate(
            trained_model.named_steps["model"], X_test_processed, y_test
        )

        # Ensure that the evaluation metrics are returned as a dictionary
        if not isinstance(evaluation_metrics, dict):
            logger.error("Evaluation metrics must be returned as a dictionary.")
            raise ValueError("Evaluation metrics must be returned as a dictionary.")
            
        mse = evaluation_metrics.get("Mean Squared Error", None)
        
        logger.info("Model evaluation completed.")
        logger.info(f"Evaluation metrics: {evaluation_metrics}")
        
        return evaluation_metrics, mse
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise