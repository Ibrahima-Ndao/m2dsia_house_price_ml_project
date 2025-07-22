import pandas as pd
from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
)
import logging


def feature_engineering_step(
    df: pd.DataFrame, strategy: str = "log", features: list = None
) -> pd.DataFrame:
    """Performs feature engineering using FeatureEngineer and selected strategy."""
    logger = logging.getLogger("feature_engineering_step")
    logger.info(f"Starting feature engineering step with strategy: {strategy}")

    try:
        # Ensure features is a list, even if not provided
        if features is None:
            features = []  # or raise an error if features are required

        if strategy == "log":
            engineer = FeatureEngineer(LogTransformation(features))
        elif strategy == "standard_scaling":
            engineer = FeatureEngineer(StandardScaling(features))
        elif strategy == "minmax_scaling":
            engineer = FeatureEngineer(MinMaxScaling(features))
        elif strategy == "onehot_encoding":
            engineer = FeatureEngineer(OneHotEncoding(features))
        else:
            logger.error(f"Unsupported feature engineering strategy: {strategy}")
            raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

        transformed_df = engineer.apply_feature_engineering(df)
        logger.info(f"Feature engineering completed. Output shape: {transformed_df.shape}")
        return transformed_df
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise