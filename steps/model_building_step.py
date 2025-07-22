import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pickle
import os


def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Builds and trains a Linear Regression model using scikit-learn wrapped in a pipeline.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.

    Returns:
    Pipeline: The trained scikit-learn pipeline including preprocessing and the Linear Regression model.
    """
    logger = logging.getLogger("model_building_step")
    logger.info("Starting model building step.")
    
    try:
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            logger.error("X_train must be a pandas DataFrame.")
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            logger.error("y_train must be a pandas Series.")
            raise TypeError("y_train must be a pandas Series.")

        # Identify categorical and numerical columns
        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
        numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

        logger.info(f"Categorical columns: {categorical_cols.tolist()}")
        logger.info(f"Numerical columns: {numerical_cols.tolist()}")

        # Define preprocessing for categorical and numerical features
        numerical_transformer = SimpleImputer(strategy="mean")
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # Define the model training pipeline
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])

        logger.info("Building and training the Linear Regression model.")
        pipeline.fit(X_train, y_train)
        logger.info("Model training completed.")

        # Save the model
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "prices_predictor.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        logger.info(f"Model saved to {model_path}")

        # Log the columns that the model expects
        if categorical_cols.any():
            onehot_encoder = (
                pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
            )
            expected_columns = numerical_cols.tolist() + list(
                onehot_encoder.get_feature_names_out(categorical_cols)
            )
        else:
            expected_columns = numerical_cols.tolist()
            
        logger.info(f"Model expects the following columns: {expected_columns}")

        return pipeline
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise