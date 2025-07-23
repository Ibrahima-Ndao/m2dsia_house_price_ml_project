import os
import joblib
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step


def calculate_accuracy_metrics(y_true, y_pred, tolerance_percent=10):
    """
    Calculate accuracy-like metrics for regression problems.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values  
        tolerance_percent: Tolerance percentage for accuracy calculation
        
    Returns:
        dict: Dictionary containing various accuracy metrics
    """
    # Mean Absolute Percentage Error
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Accuracy within tolerance (percentage of predictions within X% of true value)
    percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
    accuracy_within_tolerance = np.mean(percentage_errors <= tolerance_percent) * 100
    
    # R² Score (coefficient of determination) 
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R² (more conservative)
    n = len(y_true)
    p = 1  # assuming single target variable
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Normalized RMSE (NRMSE) - normalized by the range of actual values
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    
    # Custom accuracy score (1 - MAPE/100)
    custom_accuracy = max(0, (1 - mape) * 100)
    
    return {
        'mape': mape,
        'accuracy_within_tolerance': accuracy_within_tolerance,
        'tolerance_percent': tolerance_percent,
        'r2_score': r2,
        'adjusted_r2': adjusted_r2,
        'mae': mae,
        'rmse': rmse,
        'nrmse': nrmse,
        'custom_accuracy': custom_accuracy
    }


def print_detailed_metrics(metrics_dict, mse):
    """Print detailed evaluation metrics in a formatted way."""
    
    print("\n" + "="*60)
    print("🎯 MODEL EVALUATION RESULTS")
    print("="*60)
    
    print("\n📊 ACCURACY METRICS:")
    print(f"   • Custom Accuracy Score: {metrics_dict['custom_accuracy']:.2f}%")
    print(f"   • Accuracy within ±{metrics_dict['tolerance_percent']}%: {metrics_dict['accuracy_within_tolerance']:.2f}%")
    print(f"   • R² Score (Coefficient of Determination): {metrics_dict['r2_score']:.4f}")
    print(f"   • Adjusted R² Score: {metrics_dict['adjusted_r2']:.4f}")
    
    print("\n📈 ERROR METRICS:")
    print(f"   • Mean Absolute Percentage Error (MAPE): {metrics_dict['mape']:.4f}")
    print(f"   • Mean Absolute Error (MAE): ${metrics_dict['mae']:,.2f}")
    print(f"   • Root Mean Square Error (RMSE): ${metrics_dict['rmse']:,.2f}")
    print(f"   • Normalized RMSE (NRMSE): {metrics_dict['nrmse']:.4f}")
    print(f"   • Mean Squared Error (MSE): ${mse:,.2f}")
    
    print("\n🎪 PERFORMANCE INTERPRETATION:")
    if metrics_dict['r2_score'] >= 0.8:
        print("   ✅ EXCELLENT: Model explains >80% of variance in house prices")
    elif metrics_dict['r2_score'] >= 0.6:
        print("   ✅ GOOD: Model explains 60-80% of variance in house prices") 
    elif metrics_dict['r2_score'] >= 0.4:
        print("   ⚠️ FAIR: Model explains 40-60% of variance in house prices")
    else:
        print("   ❌ POOR: Model explains <40% of variance in house prices")
    
    if metrics_dict['accuracy_within_tolerance'] >= 70:
        print(f"   ✅ HIGH PRECISION: {metrics_dict['accuracy_within_tolerance']:.1f}% of predictions within ±{metrics_dict['tolerance_percent']}%")
    elif metrics_dict['accuracy_within_tolerance'] >= 50:
        print(f"   ⚠️ MODERATE PRECISION: {metrics_dict['accuracy_within_tolerance']:.1f}% of predictions within ±{metrics_dict['tolerance_percent']}%")
    else:
        print(f"   ❌ LOW PRECISION: Only {metrics_dict['accuracy_within_tolerance']:.1f}% of predictions within ±{metrics_dict['tolerance_percent']}%")
        
    print("="*60)


def ml_pipeline():
    """Define an enhanced end-to-end machine learning pipeline with comprehensive metrics."""
    
    print("🚀 Starting Enhanced ML Pipeline...")
    print("="*60)
    
    # Data Ingestion Step
    print("📥 Step 1: Data Ingestion")
    raw_data = data_ingestion_step(
        file_path="D:/Data science/MLOPS/price_house_project/house-price-project/data/archive.zip"
    )
    print(f"   ✅ Loaded {len(raw_data)} records")

    # Handling Missing Values Step
    print("\n🛠️ Step 2: Handling Missing Values")
    filled_data = handle_missing_values_step(raw_data)
    missing_before = raw_data.isnull().sum().sum()
    missing_after = filled_data.isnull().sum().sum()
    print(f"   ✅ Reduced missing values from {missing_before} to {missing_after}")

    # Feature Engineering Step
    print("\n⚙️ Step 3: Feature Engineering")
    engineered_data = feature_engineering_step(
        filled_data, strategy="log", features=["sqft_living", "price", "sqft_lot"]
    )
    print(f"   ✅ Applied feature engineering to key variables")

    # Outlier Detection Step
    print("\n🎯 Step 4: Outlier Detection")
    initial_count = len(engineered_data)
    clean_data = outlier_detection_step(engineered_data, column_name="price")
    outliers_removed = initial_count - len(clean_data)
    print(f"   ✅ Removed {outliers_removed} outliers ({outliers_removed/initial_count*100:.1f}% of data)")

    # Data Splitting Step
    print("\n✂️ Step 5: Data Splitting")
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="price")
    print(f"   ✅ Split data into train ({len(X_train)}) and test ({len(X_test)}) sets")

    # Model Building Step
    print("\n🤖 Step 6: Model Building")
    model = model_building_step(X_train=X_train, y_train=y_train)
    print(f"   ✅ Model trained successfully")

    # Model Evaluation Step with Enhanced Metrics
    print("\n📊 Step 7: Model Evaluation")
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )
    
    # Calculate predictions for accuracy metrics
    y_pred = model.predict(X_test)
    
    # Calculate comprehensive accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(y_test, y_pred, tolerance_percent=10)
    
    # Combine all metrics
    all_metrics = {
        **evaluation_metrics,
        **accuracy_metrics,
        'mse': mse
    }
    
    # Display detailed results
    print_detailed_metrics(accuracy_metrics, mse)
    
    # Additional model insights
    print(f"\n📈 ADDITIONAL INSIGHTS:")
    price_range = y_test.max() - y_test.min()
    avg_price = y_test.mean()
    print(f"   • Price range in test set: ${y_test.min():,.0f} - ${y_test.max():,.0f}")
    print(f"   • Average price in test set: ${avg_price:,.0f}")
    print(f"   • Average prediction error: ${accuracy_metrics['mae']:,.0f} ({accuracy_metrics['mae']/avg_price*100:.1f}% of avg price)")
    
    # Save the model with enhanced metadata
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"prices_predictor_{timestamp}.pkl")
    
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save comprehensive model metadata
    metadata = {
        'model_path': model_path,
        'timestamp': timestamp,
        'training_timestamp': datetime.now().isoformat(),
        'model_name': 'prices_predictor',
        'data_stats': {
            'total_records': len(raw_data),
            'records_after_cleaning': len(clean_data),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'outliers_removed': outliers_removed,
            'missing_values_handled': missing_before
        },
        'evaluation_metrics': all_metrics,
        'model_performance': {
            'r2_score': accuracy_metrics['r2_score'],
            'custom_accuracy': accuracy_metrics['custom_accuracy'],
            'accuracy_within_10_percent': accuracy_metrics['accuracy_within_tolerance'],
            'mape': accuracy_metrics['mape']
        },
        'price_statistics': {
            'min_price': float(y_test.min()),
            'max_price': float(y_test.max()),
            'avg_price': float(y_test.mean()),
            'price_std': float(y_test.std())
        }
    }
    
    metadata_path = os.path.join(model_dir, f"model_metadata_{timestamp}.pkl")
    joblib.dump(metadata, metadata_path)
    print(f"💾 Metadata saved to: {metadata_path}")
    
    # Save the latest model path for deployment
    latest_model_path = os.path.join(model_dir, "latest_model_path.txt")
    with open(latest_model_path, 'w') as f:
        f.write(model_path)
    
    # Save performance summary for quick reference
    performance_summary = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'key_metrics': {
            'r2_score': f"{accuracy_metrics['r2_score']:.4f}",
            'custom_accuracy': f"{accuracy_metrics['custom_accuracy']:.2f}%",
            'accuracy_within_10_percent': f"{accuracy_metrics['accuracy_within_tolerance']:.2f}%",
            'mape': f"{accuracy_metrics['mape']:.4f}",
            'mae': f"${accuracy_metrics['mae']:,.2f}",
            'rmse': f"${accuracy_metrics['rmse']:,.2f}"
        }
    }
    
    summary_path = os.path.join(model_dir, "latest_performance.json")
    import json
    with open(summary_path, 'w') as f:
        json.dump(performance_summary, f, indent=2)
    print(f"📋 Performance summary saved to: {summary_path}")
    
    print("\n🎉 ML Pipeline completed successfully!")
    print("="*60)
    
    return model, model_path


if __name__ == "__main__":
    # Running the enhanced pipeline
    print("🏠 House Price Prediction - Enhanced ML Pipeline")
    trained_model, model_path = ml_pipeline()
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Model saved at: {model_path}")
    print(f"🚀 Ready for deployment!")
    