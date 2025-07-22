import os
import joblib
from datetime import datetime
from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step


def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""
    
    print("Starting ML Pipeline...")
    
    # Data Ingestion Step
    print("Step 1: Data Ingestion")
    raw_data = data_ingestion_step(
        file_path="D:/Data science/MLOPS/price_house_project/house-price-project/data/archive.zip"
    )

    # Handling Missing Values Step
    print("Step 2: Handling Missing Values")
    filled_data = handle_missing_values_step(raw_data)

    # Feature Engineering Step
    print("Step 3: Feature Engineering")
    engineered_data = feature_engineering_step(
        filled_data, strategy="log", features=["sqft_living", "price", "sqft_lot"]
    )

    # Outlier Detection Step
    print("Step 4: Outlier Detection")
    clean_data = outlier_detection_step(engineered_data, column_name="price")

    # Data Splitting Step
    print("Step 5: Data Splitting")
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="price")

    # Model Building Step
    print("Step 6: Model Building")
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model Evaluation Step
    print("Step 7: Model Evaluation")
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )
    
    print("Evaluation Metrics:")
    print(evaluation_metrics)
    print(f"MSE: {mse}")
    
    # Save the model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"prices_predictor_{timestamp}.pkl")
    
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save model metadata
    metadata = {
        'model_path': model_path,
        'timestamp': timestamp,
        'evaluation_metrics': evaluation_metrics,
        'mse': mse,
        'model_name': 'prices_predictor'
    }
    
    metadata_path = os.path.join(model_dir, f"model_metadata_{timestamp}.pkl")
    joblib.dump(metadata, metadata_path)
    
    # Save the latest model path for deployment
    latest_model_path = os.path.join(model_dir, "latest_model_path.txt")
    with open(latest_model_path, 'w') as f:
        f.write(model_path)
    
    print("ML Pipeline completed successfully!")
    return model, model_path


if __name__ == "__main__":
    # Running the pipeline
    trained_model, model_path = ml_pipeline()
    print(f"Training completed. Model saved at: {model_path}")