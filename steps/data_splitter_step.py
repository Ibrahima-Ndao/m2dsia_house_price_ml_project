from typing import Tuple
import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
import logging


def data_splitter_step(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the data into training and testing sets using DataSplitter and a chosen strategy."""
    logger = logging.getLogger("data_splitter_step")
    logger.info("Starting data splitting step.")
    
    try:
        splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
        X_train, X_test, y_train, y_test = splitter.split(df, target_column)
        
        logger.info(f"Data split completed - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise
