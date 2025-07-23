import click
import logging
import time
import os
import json
import joblib
import requests
import asyncio
from typing import Optional
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
    FastAPIDeploymentService
)

console = Console()


def display_model_performance():
    """Display the latest model performance metrics in a nice table."""
    try:
        with open("models/latest_performance.json", 'r') as f:
            performance = json.load(f)
        
        table = Table(title="🎯 Latest Model Performance", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Interpretation", style="yellow")
        
        # Add rows with interpretations
        metrics = performance['key_metrics']
        
        r2 = float(metrics['r2_score'])
        r2_interp = "Excellent" if r2 >= 0.8 else "Good" if r2 >= 0.6 else "Fair" if r2 >= 0.4 else "Poor"
        table.add_row("R² Score", metrics['r2_score'], r2_interp)
        
        acc = float(metrics['custom_accuracy'].rstrip('%'))
        acc_interp = "Excellent" if acc >= 80 else "Good" if acc >= 60 else "Fair" if acc >= 40 else "Poor" 
        table.add_row("Custom Accuracy", metrics['custom_accuracy'], acc_interp)
        
        within_10 = float(metrics['accuracy_within_10_percent'].rstrip('%'))
        within_interp = "High Precision" if within_10 >= 70 else "Moderate" if within_10 >= 50 else "Low Precision"
        table.add_row("Accuracy ±10%", metrics['accuracy_within_10_percent'], within_interp)
        
        table.add_row("MAPE", metrics['mape'], "Lower is better")
        table.add_row("MAE", metrics['mae'], "Average error")
        table.add_row("RMSE", metrics['rmse'], "Root mean squared error")
        
        console.print(table)
        console.print(f"📅 Model trained: {performance['timestamp']}")
        
    except FileNotFoundError:
        console.print("❌ No performance metrics found. Train a model first.")


async def test_api_endpoints(base_url: str = "http://localhost:8000"):
    """Test all API endpoints asynchronously."""
    
    console.print("🧪 Testing API endpoints...")
    
    import aiohttp
    
    # Test data
    test_house = {
        "date": "2014-05-02",
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1180,
        "sqft_lot": 5650,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "sqft_above": 1180,
        "sqft_basement": 0,
        "yr_built": 2003,
        "yr_renovated": 0,
        "street": "Northwest 105th Street",
        "city": "Seattle", 
        "statezip": "WA 98105",
        "country": "USA"
    }
    
    async with aiohttp.ClientSession() as session:
        # Test health endpoint
        try:
            async with session.get(f"{base_url}/health") as resp:
                if resp.status == 200:
                    health_data = await resp.json()
                    console.print("✅ Health check passed")
                    console.print(f"   Model loaded: {health_data['model_loaded']}")
                    console.print(f"   Uptime: {health_data['uptime_seconds']:.1f}s")
                else:
                    console.print("❌ Health check failed")
                    return False
        except Exception as e:
            console.print(f"❌ Could not connect to API: {e}")
            return False
        
        # Test single prediction
        try:
            async with session.post(f"{base_url}/predict", json=test_house) as resp:
                if resp.status == 200:
                    pred_data = await resp.json()
                    predicted_price = pred_data['predictions'][0]
                    console.print("✅ Single prediction test passed")
                    console.print(f"   Predicted price: ${predicted_price:,.2f}")
                    console.print(f"   Model version: {pred_data.get('model_version', 'Unknown')}")
                else:
                    console.print("❌ Single prediction test failed")
                    error_text = await resp.text()
                    console.print(f"   Error: {error_text}")
                    return False
        except Exception as e:
            console.print(f"❌ Single prediction failed: {e}")
            return False
        
        # Test batch prediction
        try:
            batch_data = {"houses": [test_house, test_house]}  # Two identical houses for testing
            async with session.post(f"{base_url}/predict/batch", json=batch_data) as resp:
                if resp.status == 200:
                    batch_result = await resp.json()
                    console.print("✅ Batch prediction test passed")
                    console.print(f"   Predictions count: {batch_result['count']}")
                    console.print(f"   Sample predictions: ${batch_result['predictions'][0]:,.2f}")
                else:
                    console.print("❌ Batch prediction test failed")
                    return False
        except Exception as e:
            console.print(f"❌ Batch prediction failed: {e}")
            return False
        
        # Test model info endpoint
        try:
            async with session.get(f"{base_url}/model/info") as resp:
                if resp.status == 200:
                    info_data = await resp.json()
                    console.print("✅ Model info endpoint test passed")
                    console.print(f"   Prediction count: {info_data.get('prediction_count', 0)}")
                else:
                    console.print("❌ Model info test failed")
        except Exception as e:
            console.print(f"⚠️  Model info test skipped: {e}")
    
    return True


@click.group()
def cli():
    """🏠 House Price Prediction MLOps CLI"""
    pass


@cli.command()
@click.option(
    "--port",
    default=8000,
    help="Port to run the FastAPI server on"
)
@click.option(
    "--show-performance",
    is_flag=True,
    default=False,
    help="Display model performance metrics before deployment"
)
def deploy(port: int, show_performance: bool):
    """🚀 Deploy the house price prediction API"""
    
    console.print("[bold green]🚀 Starting FastAPI Deployment Pipeline...[/bold green]")
    
    if show_performance:
        display_model_performance()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Training task
        train_task = progress.add_task("Training model...", total=None)
        
        try:
            deployment_service = continuous_deployment_pipeline()
            progress.update(train_task, description="✅ Model trained and deployed!")
            
            console.print(f"\n🌐 [bold green]Service running at: http://localhost:{port}[/bold green]")
            console.print(f"📚 [bold blue]Swagger UI: http://localhost:{port}/docs[/bold blue]")
            console.print(f"📖 [bold blue]ReDoc: http://localhost:{port}/redoc[/bold blue]")
            
            # Display performance metrics
            display_model_performance()
            
            # Wait a moment then test the API
            time.sleep(2)
            console.print("\n🧪 Testing deployed API...")
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            test_result = loop.run_until_complete(test_api_endpoints(f"http://localhost:{port}"))
            loop.close()
            
            if test_result:
                console.print("\n✅ [bold green]All API tests passed![/bold green]")
            else:
                console.print("\n⚠️  [bold yellow]Some API tests failed[/bold yellow]")
            
            # Keep service running
            console.print("\n🔄 [bold cyan]Service is running... Press Ctrl+C to stop[/bold cyan]")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                console.print("\n🛑 [bold red]Shutting down...[/bold red]")
                console.print("✅ Service stopped!")
                
        except Exception as e:
            progress.update(train_task, description="❌ Deployment failed!")
            console.print(f"❌ [bold red]Deployment error: {e}[/bold red]")


@cli.command()
@click.option(
    "--port", 
    default=8000,
    help="Port where the API service is running"
)
def test(port: int):
    """🧪 Test the deployed API service"""
    
    console.print("[bold blue]🧪 Testing API Service...[/bold blue]")
    
    # Run async test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    test_result = loop.run_until_complete(test_api_endpoints(f"http://localhost:{port}"))
    loop.close()
    
    if test_result:
        console.print("\n✅ [bold green]All tests passed![/bold green]")
    else:
        console.print("\n❌ [bold red]Tests failed![/bold red]")


@cli.command()
def inference():
    """🔍 Run batch inference pipeline"""
    
    console.print("[bold blue]🔍 Starting Batch Inference...[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Running inference...", total=None)
        
        try:
            predictions = inference_pipeline()
            if predictions is not None:
                progress.update(task, description="✅ Batch inference completed!")
                console.print(f"\n✅ [bold green]Generated {len(predictions)} predictions[/bold green]")
                console.print(f"📊 Sample predictions: {predictions[:3].tolist()}")
            else:
                progress.update(task, description="❌ Batch inference failed!")
                console.print("❌ [bold red]Batch inference failed![/bold red]")
                
        except Exception as e:
            progress.update(task, description="❌ Inference error!")
            console.print(f"❌ [bold red]Inference error: {e}[/bold red]")


@cli.command()
def performance():
    """📊 Show model performance metrics"""
    
    console.print("[bold blue]📊 Model Performance Metrics[/bold blue]")
    display_model_performance()


@cli.command()
@click.option(
    "--output-format",
    type=click.Choice(['json', 'curl', 'python']),
    default='json',
    help="Format for the example request"
)
def example(output_format: str):
    """📋 Generate example API requests"""
    
    example_house = {
        "date": "2014-05-02",
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1180,
        "sqft_lot": 5650,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "sqft_above": 1180,
        "sqft_basement": 0,
        "yr_built": 2003,
        "yr_renovated": 0,
        "street": "Northwest 105th Street",
        "city": "Seattle",
        "statezip": "WA 98105",
        "country": "USA"
    }
    
    console.print(f"[bold blue]📋 Example Request ({output_format.upper()})[/bold blue]")
    
    if output_format == 'json':
        console.print("\n[bold cyan]Single Prediction:[/bold cyan]")
        console.print(f"POST http://localhost:8000/predict")
        console.print(f"Content-Type: application/json")
        console.print(json.dumps(example_house, indent=2))
        
        console.print("\n[bold cyan]Batch Prediction:[/bold cyan]")
        console.print(f"POST http://localhost:8000/predict/batch")
        console.print(json.dumps({"houses": [example_house]}, indent=2))
        
    elif output_format == 'curl':
        console.print("\n[bold cyan]Single Prediction:[/bold cyan]")
        curl_single = f"""curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{json.dumps(example_house)}'"""
        console.print(curl_single)
        
        console.print("\n[bold cyan]Batch Prediction:[/bold cyan]")
        curl_batch = f"""curl -X POST "http://localhost:8000/predict/batch" \\
     -H "Content-Type: application/json" \\
     -d '{json.dumps({"houses": [example_house]})}'"""
        console.print(curl_batch)
        
    elif output_format == 'python':
        console.print("\n[bold cyan]Python Example:[/bold cyan]")
        python_code = f'''import requests
import json

# Single prediction
house_data = {json.dumps(example_house, indent=4)}

response = requests.post(
    "http://localhost:8000/predict",
    json=house_data
)

if response.status_code == 200:
    result = response.json()
    predicted_price = result["predictions"][0]
    print(f"Predicted price: ${{predicted_price:,.2f}}")
else:
    print(f"Error: {{response.text}}")

# Batch prediction
batch_data = {{"houses": [house_data]}}
batch_response = requests.post(
    "http://localhost:8000/predict/batch", 
    json=batch_data
)

if batch_response.status_code == 200:
    batch_result = batch_response.json()
    print(f"Batch predictions: {{batch_result['predictions']}}")'''
        console.print(python_code)


@cli.command()
def status():
    """📊 Check service status and health"""
    
    console.print("[bold blue]📊 Service Status[/bold blue]")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            
            table = Table(title="Service Health", show_header=True, header_style="bold green")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Status", health_data['status'])
            table.add_row("Model Loaded", str(health_data['model_loaded']))
            table.add_row("Uptime", f"{health_data['uptime_seconds']:.1f} seconds")
            table.add_row("Timestamp", health_data['timestamp'])
            
            if health_data.get('memory_usage_mb'):
                table.add_row("Memory Usage", f"{health_data['memory_usage_mb']:.1f} MB")
            
            console.print(table)
            
            # Try to get model info
            try:
                info_response = requests.get("http://localhost:8000/model/info", timeout=5)
                if info_response.status_code == 200:
                    info_data = info_response.json()
                    console.print(f"\n📈 [bold cyan]Model Info:[/bold cyan]")
                    console.print(f"   Version: {info_data.get('model_version', 'Unknown')}")
                    console.print(f"   Predictions made: {info_data.get('prediction_count', 0)}")
            except:
                pass
                
        else:
            console.print("❌ [bold red]Service is not healthy[/bold red]")
            
    except requests.exceptions.RequestException:
        console.print("❌ [bold red]Cannot connect to service at http://localhost:8000[/bold red]")
        console.print("💡 Make sure the API is running with: python run_deployment.py deploy")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    cli()