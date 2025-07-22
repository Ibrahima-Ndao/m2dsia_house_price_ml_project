import pandas as pd
from src.ingest_data import DataIngestorFactory
import logging


def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a ZIP file using the appropriate DataIngestor."""
    logger = logging.getLogger("data_ingestion_step")
    logger.info("Starting data ingestion step.")
    
    try:
        # Determine the file extension
        file_extension = ".zip"  # Since we're dealing with ZIP files, this is hardcoded

        # Get the appropriate DataIngestor
        data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

        # Ingest the data and load it into a DataFrame
        df = data_ingestor.ingest(file_path)
        logger.info(f"Data ingestion completed. DataFrame shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise