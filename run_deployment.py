import click
import logging
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)
from rich import print
from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step


@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def run_main(stop_service: bool):
    """Run the prices predictor deployment pipeline"""
    model_name = "prices_predictor"

    if stop_service:
        # Get the MLflow model deployer stack component
        # This part of the code is no longer relevant as MLflow is removed.
        # The original code had MLflow specific logic here.
        # For now, we'll just print a message indicating the service cannot be stopped.
        print("MLflow prediction service is no longer managed by this script.")
        return

    # Run the continuous deployment pipeline
    continuous_deployment_pipeline()

    # Get the active model deployer
    # This part of the code is no longer relevant as MLflow is removed.
    # The original code had MLflow specific logic here.
    # For now, we'll just print a message indicating the service cannot be started.
    print("MLflow prediction service is no longer managed by this script.")

    # Fetch existing services with the same pipeline name, step name, and model name
    # This part of the code is no longer relevant as MLflow is removed.
    # The original code had MLflow specific logic here.
    # For now, we'll just print a message indicating the service cannot be checked.
    print("MLflow prediction service is no longer managed by this script.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run_main()
