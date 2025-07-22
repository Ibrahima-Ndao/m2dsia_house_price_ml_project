import click
import logging
import time
import os
import joblib
import requests
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)
from rich import print


@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
@click.option(
    "--run-inference",
    is_flag=True,
    default=False,
    help="Run batch inference pipeline",
)
@click.option(
    "--deploy-only",
    is_flag=True,
    default=False,
    help="Only deploy without training (use existing model)",
)
def run_main(stop_service: bool, run_inference: bool, deploy_only: bool):
    """Run the prices predictor deployment pipeline"""
    
    model_name = "prices_predictor"
    
    print(f"🚀 Starting {model_name} deployment pipeline...")
    
    if stop_service:
        print("🛑 Stopping prediction service...")
        try:
            # Try to gracefully shutdown the service
            response = requests.get("http://localhost:5000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Service is running. Note: Manual shutdown required (Ctrl+C)")
            else:
                print("❌ Service not responding")
        except requests.exceptions.RequestException:
            print("❌ No service found running on port 5000")
        return

    if run_inference:
        print("🔍 Running batch inference pipeline...")
        predictions = inference_pipeline()
        if predictions is not None:
            print("✅ Batch inference completed successfully!")
        else:
            print("❌ Batch inference failed!")
        return

    if deploy_only:
        print("📦 Deploying existing model...")
        try:
            # Check if there's a trained model
            with open("models/latest_model_path.txt", 'r') as f:
                model_path = f.read().strip()
            
            if not os.path.exists(model_path):
                print("❌ No trained model found. Please train a model first.")
                return
            
            # Deploy the existing model
            from pipelines.deployment_pipeline import SimpleDeploymentService
            deployment_service = SimpleDeploymentService(model_path, port=5000)
            deployment_service.start()
            
            # Save deployment info
            deployment_info = {
                'model_path': model_path,
                'service_port': 5000,
                'service_url': 'http://localhost:5000',
                'status': 'deployed'
            }
            
            deployment_path = "models/deployment_info.pkl"
            joblib.dump(deployment_info, deployment_path)
            
            print("✅ Model deployed successfully!")
            print("🌐 Service available at: http://localhost:5000")
            print("📋 Endpoints:")
            print("   - POST /predict : Make predictions")
            print("   - GET /health  : Health check")
            
            # Keep the service running
            try:
                print("🔄 Service is running... Press Ctrl+C to stop")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
                deployment_service.stop()
            
        except FileNotFoundError:
            print("❌ No trained model found. Please train a model first.")
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
        return

    # Run the full continuous deployment pipeline
    print("🏃‍♂️ Running continuous deployment pipeline...")
    
    try:
        deployment_service = continuous_deployment_pipeline()
        
        print("✅ Deployment pipeline completed successfully!")
        print("🌐 Model service is now running at: http://localhost:5000")
        print("📋 Available endpoints:")
        print("   - POST /predict : Make predictions")
        print("   - GET /health  : Health check")
        
        # Test the service
        print("🧪 Testing the deployed service...")
        try:
            response = requests.get("http://localhost:5000/health", timeout=10)
            if response.status_code == 200:
                print("✅ Service health check passed!")
                
                # Example prediction test
                test_data = {
                    'date':['2014-05-02'],
                    'bedrooms':[2],
                    'bathrooms':[1],
                    'sqft_living':[2570],
                    'sqft_lot':[10193],
                    'floors':[2],
                    'waterfront':[0],
                    'view':[4],
                    'condition':[5],
                    'sqft_above':[2170],
                    'sqft_basement':[0],
                    'yr_built':[1910],
                    'yr_renovated':[0],
                    'street':['West 9th Street'],
                    'city':['New York'],
                    'statezip':['NY 10011'],
                    'country':['USA']
                }
                
                pred_response = requests.post(
                    "http://localhost:5000/predict",
                    json=test_data,
                    timeout=10
                )
                
                if pred_response.status_code == 200:
                    result = pred_response.json()
                    print("✅ Test prediction successful!")
                    print(f"   Sample prediction: {result['predictions'][0]:.2f}")
                else:
                    print("⚠️  Test prediction failed")
                    
            else:
                print("❌ Service health check failed")
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Could not test service: {e}")
        
        # Keep the service running
        print("\n🔄 Service is running... Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            deployment_service.stop()
            
    except Exception as e:
        print(f"❌ Deployment pipeline failed: {e}")
        logging.error(f"Deployment pipeline error: {e}", exc_info=True)


@click.command()
def test_service():
    """Test the deployed model service."""
    
    print("🧪 Testing deployed model service...")
    
    try:
        # Health check
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Service is healthy!")
        else:
            print("❌ Service health check failed")
            return
        
        # Test prediction
        test_data = {
            'date':['2014-05-02'],
            'bedrooms':[3],
            'bathrooms':[2],
            'sqft_living':[1180],
            'sqft_lot':[5650],
            'floors':[1],
            'waterfront':[0],
            'view':[0],
            'condition':[3],
            'sqft_above':[1180],
            'sqft_basement':[0],
            'yr_built':[2003],
            'yr_renovated':[0],
            'street':['Northwest 105th Street'],
            'city':['Seattle'],
            'statezip':['WA 98105'],
            'country':['USA']
        }
        
        pred_response = requests.post(
            "http://localhost:5000/predict",
            json=test_data,
            timeout=10
        )
        
        if pred_response.status_code == 200:
            result = pred_response.json()
            print("✅ Prediction successful!")
            print(f"   Predicted price: ${result['predictions'][0]:,.2f}")
        else:
            print("❌ Prediction failed")
            print(f"   Error: {pred_response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to service: {e}")
        print("   Make sure the service is running on http://localhost:5000")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run_main()