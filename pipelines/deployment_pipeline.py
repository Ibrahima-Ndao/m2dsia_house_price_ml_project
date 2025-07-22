import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import threading
import time
from pipelines.training_pipeline import ml_pipeline
from steps.dynamic_importer import dynamic_importer


class ModelService:
    """Simple model service to replace MLflow deployment."""
    
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.model_path = model_path
        
    def predict(self, data):
        """Make predictions on input data."""
        if isinstance(data, pd.DataFrame):
            return self.model.predict(data)
        else:
            # Convert to DataFrame if needed
            df = pd.DataFrame(data)
            return self.model.predict(df)


class SimpleDeploymentService:
    """Simple deployment service with Swagger documentation."""
    
    def __init__(self, model_path, port=5000):
        self.model_service = ModelService(model_path)
        self.port = port
        self.app = Flask(__name__)
        
        # Initialize Flask-RESTX
        self.api = Api(
            self.app,
            version='1.0',
            title='House Price Prediction API',
            description='API for house price predictions using machine learning',
            doc='/swagger/',  # Swagger UI will be available at /swagger/
            prefix='/api/v1'
        )
        
        self.setup_models()
        self.setup_routes()
        self.server_thread = None
    
    def setup_models(self):
        """Define Swagger models for request/response serialization."""
        
        # Model for single house prediction request
        self.house_data_model = self.api.model('HouseData', {
            'date': fields.List(fields.String, required=True, description='Date in YYYY-MM-DD format', example=['2014-05-02']),
            'bedrooms': fields.List(fields.Integer, required=True, description='Number of bedrooms', example=[3]),
            'bathrooms': fields.List(fields.Float, required=True, description='Number of bathrooms', example=[2.0]),
            'sqft_living': fields.List(fields.Integer, required=True, description='Square feet of living space', example=[1180]),
            'sqft_lot': fields.List(fields.Integer, required=True, description='Square feet of lot', example=[5650]),
            'floors': fields.List(fields.Float, required=True, description='Number of floors', example=[1.0]),
            'waterfront': fields.List(fields.Integer, required=True, description='Waterfront property (0 or 1)', example=[0]),
            'view': fields.List(fields.Integer, required=True, description='Quality of view (0-4)', example=[0]),
            'condition': fields.List(fields.Integer, required=True, description='Condition of the house (1-5)', example=[3]),
            'sqft_above': fields.List(fields.Integer, required=True, description='Square feet above ground', example=[1180]),
            'sqft_basement': fields.List(fields.Integer, required=True, description='Square feet of basement', example=[0]),
            'yr_built': fields.List(fields.Integer, required=True, description='Year built', example=[2003]),
            'yr_renovated': fields.List(fields.Integer, required=True, description='Year renovated (0 if never)', example=[0]),
            'street': fields.List(fields.String, required=True, description='Street address', example=['Northwest 105th Street']),
            'city': fields.List(fields.String, required=True, description='City', example=['Seattle']),
            'statezip': fields.List(fields.String, required=True, description='State and ZIP code', example=['WA 98105']),
            'country': fields.List(fields.String, required=True, description='Country', example=['USA'])
        })
        
        # Model for multiple houses prediction request
        self.batch_prediction_model = self.api.model('BatchPrediction', {
            'houses': fields.List(fields.Nested(self.house_data_model), required=True, description='List of houses to predict')
        })
        
        # Response models
        self.prediction_response_model = self.api.model('PredictionResponse', {
            'predictions': fields.List(fields.Float, description='Predicted house prices'),
            'status': fields.String(description='Response status'),
            'count': fields.Integer(description='Number of predictions made')
        })
        
        self.error_response_model = self.api.model('ErrorResponse', {
            'error': fields.String(description='Error message'),
            'status': fields.String(description='Error status')
        })
        
        self.health_response_model = self.api.model('HealthResponse', {
            'status': fields.String(description='Service health status'),
            'model_loaded': fields.Boolean(description='Whether model is loaded'),
            'timestamp': fields.String(description='Current timestamp')
        })
        
    def setup_routes(self):
        """Setup API routes with Swagger documentation."""
        
        ns = self.api.namespace('predictions', description='House price prediction operations')
        
        @ns.route('/predict')
        class Predict(Resource):
            @ns.expect(self.house_data_model)
            @ns.marshal_with(self.prediction_response_model, code=200)
            @ns.marshal_with(self.error_response_model, code=400)
            def post(self):
                """Make a single house price prediction"""
                try:
                    data = request.get_json()
                    predictions = self.model_service.predict(data)
                    
                    return {
                        'predictions': predictions.tolist(),
                        'status': 'success',
                        'count': len(predictions)
                    }, 200
                    
                except Exception as e:
                    return {
                        'error': str(e),
                        'status': 'error'
                    }, 400
        
        @ns.route('/predict/batch')
        class BatchPredict(Resource):
            @ns.expect(self.batch_prediction_model)
            @ns.marshal_with(self.prediction_response_model, code=200)
            @ns.marshal_with(self.error_response_model, code=400)
            def post(self):
                """Make predictions for multiple houses"""
                try:
                    data = request.get_json()
                    houses = data.get('houses', [])
                    
                    all_predictions = []
                    for house_data in houses:
                        predictions = self.model_service.predict(house_data)
                        all_predictions.extend(predictions.tolist())
                    
                    return {
                        'predictions': all_predictions,
                        'status': 'success',
                        'count': len(all_predictions)
                    }, 200
                    
                except Exception as e:
                    return {
                        'error': str(e),
                        'status': 'error'
                    }, 400
        
        # Health check endpoint
        health_ns = self.api.namespace('health', description='Service health operations')
        
        @health_ns.route('/status')
        class Health(Resource):
            @health_ns.marshal_with(self.health_response_model)
            def get(self):
                """Check service health status"""
                return {
                    'status': 'healthy',
                    'model_loaded': self.model_service.model is not None,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
    
    def start(self):
        """Start the deployment service."""
        def run_server():
            self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
        
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Wait a bit for server to start
        time.sleep(2)
        print(f"Model service started on http://localhost:{self.port}")
        print(f"Swagger UI available at: http://localhost:{self.port}/swagger/")
        print(f"API endpoints:")
        print(f"  - POST /api/v1/predictions/predict")
        print(f"  - POST /api/v1/predictions/predict/batch")
        print(f"  - GET /api/v1/health/status")
        
    def stop(self):
        """Stop the deployment service."""
        print("Stopping model service...")
        # Note: Flask development server doesn't have a clean shutdown method
        # In production, you'd use a proper WSGI server like Gunicorn


def continuous_deployment_pipeline():
    """Run a training job and deploy a simple model deployment."""
    
    print("Starting Continuous Deployment Pipeline...")
    
    # Run the training pipeline
    trained_model, model_path = ml_pipeline()
    
    # Deploy the trained model
    print("Deploying model with Swagger documentation...")
    deployment_service = SimpleDeploymentService(model_path, port=5000)
    deployment_service.start()
    
    # Save deployment info
    deployment_info = {
        'model_path': model_path,
        'service_port': 5000,
        'service_url': 'http://localhost:5000',
        'swagger_url': 'http://localhost:5000/swagger/',
        'status': 'deployed'
    }
    
    deployment_path = "models/deployment_info.pkl"
    joblib.dump(deployment_info, deployment_path)
    
    print("Model deployed successfully!")
    print(f"Service available at: http://localhost:5000")
    print(f"Swagger UI available at: http://localhost:5000/swagger/")
    
    return deployment_service


def inference_pipeline():
    """Run a batch inference job with data loaded from an API."""
    
    print("Starting Inference Pipeline...")
    
    # Load batch data for inference
    print("Loading batch data...")
    batch_data = dynamic_importer()

    # Load the deployed model service info
    try:
        deployment_info = joblib.load("models/deployment_info.pkl")
        model_path = deployment_info['model_path']
        
        # Load model directly for batch inference
        model_service = ModelService(model_path)
        
        # Run predictions on the batch data
        print("Making predictions on batch data...")
        predictions = model_service.predict(batch_data)
        
        print(f"Generated {len(predictions)} predictions")
        print("Sample predictions:", predictions[:5])
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'predictions': predictions
        })
        predictions_df.to_csv("models/batch_predictions.csv", index=False)
        print("Predictions saved to models/batch_predictions.csv")
        
        return predictions
        
    except FileNotFoundError:
        print("No deployment info found. Please run continuous_deployment_pipeline first.")
        return None
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


def predict_single(data, model_path=None):
    """Make a single prediction."""
    
    if model_path is None:
        # Load latest model
        try:
            with open("models/latest_model_path.txt", 'r') as f:
                model_path = f.read().strip()
        except FileNotFoundError:
            print("No trained model found. Please run the training pipeline first.")
            return None
    
    model_service = ModelService(model_path)
    prediction = model_service.predict(data)
    
    return prediction


def create_test_client():
    """Create a test client with example data."""
    
    # Example test data
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
    
    print("Example test data:")
    print("POST /api/v1/predictions/predict")
    print("Content-Type: application/json")
    print()
    print("Request body:")
    import json
    print(json.dumps(test_data, indent=2))
    
    return test_data


if __name__ == "__main__":
    # Example usage
    service = continuous_deployment_pipeline()
    
    # Print test data example
    create_test_client()
    
    # Keep the service running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        service.stop()