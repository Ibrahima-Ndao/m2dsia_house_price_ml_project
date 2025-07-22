import pandas as pd
from src.handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingValueHandler,
)
import logging


def handle_missing_values_step(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Handles missing values using MissingValueHandler and the specified strategy."""
    logger = logging.getLogger("handle_missing_values_step")
    logger.info(f"Starting missing values handling step with strategy: {strategy}")

    try:
        if strategy == "drop":
            handler = MissingValueHandler(DropMissingValuesStrategy(axis=0))
        elif strategy in ["mean", "median", "mode", "constant"]:
            handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
        else:
            logger.error(f"Unsupported missing value handling strategy: {strategy}")
            raise ValueError(f"Unsupported missing value handling strategy: {strategy}")

        cleaned_df = handler.handle_missing_values(df)
        logger.info(f"Missing values handling completed. Output shape: {cleaned_df.shape}")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error during missing values handling: {e}")
        raise