import logging
import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection


def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Detects and removes outliers using OutlierDetector."""
    logger = logging.getLogger("outlier_detection_step")
    logger.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

    try:
        if df is None:
            logger.error("Received a NoneType DataFrame.")
            raise ValueError("Input df must be a non-null pandas DataFrame.")

        if not isinstance(df, pd.DataFrame):
            logger.error(f"Expected pandas DataFrame, got {type(df)} instead.")
            raise ValueError("Input df must be a pandas DataFrame.")

        if column_name not in df.columns:
            logger.error(f"Column '{column_name}' does not exist in the DataFrame.")
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
            
        # Ensure only numeric columns are passed
        df_numeric = df.select_dtypes(include=[int, float])
        logger.info(f"Processing numeric columns: {df_numeric.columns.tolist()}")

        outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
        outliers = outlier_detector.detect_outliers(df_numeric)
        logger.info(f"Detected {len(outliers)} outliers")
        
        df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")
        logger.info(f"Outlier detection completed. Output shape: {df_cleaned.shape}")
        
        return df_cleaned
        
    except Exception as e:
        logger.error(f"Error during outlier detection: {e}")
        raise
