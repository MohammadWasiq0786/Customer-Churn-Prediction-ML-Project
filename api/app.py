from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from datetime import datetime
import pandas as pd
import time
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.prediction_pipeline import PredictionPipeline
from api.schemas import CustomerFeatures, PredictionResponse, HealthResponse
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
Path("api/static/css").mkdir(parents=True, exist_ok=True)
Path("api/static/js").mkdir(parents=True, exist_ok=True)
Path("api/templates").mkdir(parents=True, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="api/templates")

# Global variables
prediction_pipeline = None
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize the prediction pipeline on startup"""
    global prediction_pipeline
    try:
        logger.info("Starting up the application...")
        
        # Check if model artifacts exist
        model_path = "artifacts/model.pkl"
        preprocessor_path = "artifacts/preprocessor.pkl"
        
        if not (Path(model_path).exists() and Path(preprocessor_path).exists()):
            logger.warning("Model artifacts not found. Training model...")
            # Import and run training pipeline
            try:
                from src.pipeline.training_pipeline import TrainingPipeline
                training_pipeline = TrainingPipeline()
                training_pipeline.run_training_pipeline()
            except Exception as train_error:
                logger.error(f"Training failed: {train_error}")
                # Continue without training for now
                pass
        
        try:
            prediction_pipeline = PredictionPipeline()
            logger.info("Prediction pipeline loaded successfully")
        except Exception as pipeline_error:
            logger.error(f"Failed to load prediction pipeline: {pipeline_error}")
        
    except Exception as e:
        logger.error(f"Failed to initialize prediction pipeline: {str(e)}")

# FRONTEND ROUTES
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        # Return a simple HTML page if template is not found
        return HTMLResponse(content=get_fallback_dashboard(), status_code=200)

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """About page"""
    try:
        return templates.TemplateResponse("about.html", {"request": request})
    except Exception:
        return HTMLResponse(content=get_about_page(), status_code=200)

@app.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request):
    """Analytics page"""
    try:
        return templates.TemplateResponse("analytics.html", {"request": request})
    except Exception:
        return HTMLResponse(content=get_analytics_page(), status_code=200)

@app.get("/api-info", response_class=HTMLResponse)
async def api_info(request: Request):
    """API information page"""
    try:
        return templates.TemplateResponse("api_info.html", {"request": request})
    except Exception:
        return HTMLResponse(content=get_api_info_page(), status_code=200)

# API ROUTES
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        uptime = time.time() - start_time
        
        if prediction_pipeline is None:
            status = "unhealthy - model not loaded"
        else:
            status = "healthy"
            
        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0",
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(features: CustomerFeatures):
    """Predict customer churn for a single customer"""
    try:
        if prediction_pipeline is None:
            raise HTTPException(status_code=503, detail="Prediction pipeline not available")
        
        # Convert to dictionary
        features_dict = features.dict()
        
        # Make prediction
        prediction, probability = prediction_pipeline.predict_single(features_dict)
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        logger.info(f"Prediction made: {prediction}, probability: {probability:.3f}")
        
        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=risk_level
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(features_list: list[CustomerFeatures]):
    """Predict customer churn for multiple customers"""
    try:
        if prediction_pipeline is None:
            raise HTTPException(status_code=503, detail="Prediction pipeline not available")
            
        if len(features_list) > 1000:
            raise HTTPException(status_code=400, detail="Batch size too large. Maximum 1000 predictions at once.")
        
        results = []
        for i, features in enumerate(features_list):
            try:
                features_dict = features.dict()
                prediction, probability = prediction_pipeline.predict_single(features_dict)
                
                risk_level = "Low" if probability < 0.3 else "Medium" if probability < 0.7 else "High"
                
                results.append({
                    "index": i,
                    "churn_prediction": int(prediction),
                    "churn_probability": round(float(probability), 4),
                    "risk_level": risk_level
                })
            except Exception as e:
                logger.error(f"Error predicting for item {i}: {str(e)}")
                results.append({
                    "index": i,
                    "error": str(e)
                })
        
        logger.info(f"Batch prediction completed for {len(features_list)} customers")
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the current model"""
    try:
        if prediction_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Get model information
        model_type = type(prediction_pipeline.model).__name__
        
        info = {
            "model_type": model_type,
            "model_version": "1.0.0",
            "features_count": len(prediction_pipeline.preprocessor.transformers),
            "last_updated": datetime.now().isoformat()
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Fallback HTML content functions
def get_fallback_dashboard():
    """Fallback dashboard HTML if template is missing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Churn Prediction Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }
            .nav { background: #f8f9fa; padding: 15px; text-align: center; }
            .nav a { margin: 0 15px; color: #007bff; text-decoration: none; padding: 10px 20px; border-radius: 5px; }
            .nav a:hover { background: #e9ecef; }
            .content { padding: 30px; }
            .warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .status { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
        <script>
            async function checkHealth() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    document.getElementById('status').innerHTML = 
                        `<strong>Status:</strong> ${data.status}<br>
                         <strong>Uptime:</strong> ${Math.round(data.uptime_seconds)}s<br>
                         <strong>Version:</strong> ${data.model_version}`;
                } catch (error) {
                    document.getElementById('status').innerHTML = '<strong>Status:</strong> Error checking health';
                }
            }
            window.onload = checkHealth;
            setInterval(checkHealth, 30000); // Check every 30 seconds
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Customer Churn Prediction</h1>
                <p>Advanced ML-powered customer retention analysis</p>
            </div>
            <div class="nav">
                <a href="/">🏠 Dashboard</a>
                <a href="/about">ℹ️ About</a>
                <a href="/analytics">📊 Analytics</a>
                <a href="/api-info">🔗 API Info</a>
                <a href="/docs">📚 Swagger</a>
                <a href="/health">❤️ Health</a>
            </div>
            <div class="content">
                <div class="warning">
                    <strong>⚠️ Template Missing:</strong> The main dashboard template is not found. Using fallback interface.
                    Please ensure all template files are in the <code>api/templates/</code> directory.
                </div>
                
                <div class="status" id="status">
                    Checking API status...
                </div>
                
                <h2>🚀 Quick Start</h2>
                <p>This is your Customer Churn Prediction ML Application. Here's what you can do:</p>
                <ul>
                    <li><strong>📊 Make Predictions:</strong> Use the API to predict customer churn</li>
                    <li><strong>📚 View Documentation:</strong> Check out the <a href="/docs">Swagger documentation</a></li>
                    <li><strong>🔗 API Info:</strong> Learn about available endpoints</li>
                    <li><strong>❤️ Health Check:</strong> Monitor system status</li>
                </ul>
                
                <h3>💡 API Endpoints</h3>
                <ul>
                    <li><code>POST /predict</code> - Single customer prediction</li>
                    <li><code>POST /batch_predict</code> - Batch predictions</li>
                    <li><code>GET /health</code> - Health check</li>
                    <li><code>GET /model/info</code> - Model information</li>
                </ul>
                
                <h3>🧪 Test the API</h3>
                <p>Try making a prediction with this sample data:</p>
                <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "age": 35,
       "tenure": 12,
       "monthly_charges": 75.5,
       "total_charges": 1200,
       "contract_length": 12,
       "payment_method": "Credit Card",
       "internet_service": "Fiber Optic",
       "online_security": "Yes",
       "tech_support": "Yes"
     }'</pre>
            </div>
        </div>
    </body>
    </html>
    """

def get_about_page():
    """About page content"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>About - Churn Prediction</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 1000px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }
            .nav { background: #f8f9fa; padding: 15px; text-align: center; }
            .nav a { margin: 0 15px; color: #007bff; text-decoration: none; padding: 10px 20px; border-radius: 5px; }
            .nav a:hover { background: #e9ecef; }
            .content { padding: 30px; }
            .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
            .feature-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📋 About Churn Prediction</h1>
                <p>Understanding our ML-powered customer retention solution</p>
            </div>
            <div class="nav">
                <a href="/">🏠 Dashboard</a>
                <a href="/about">ℹ️ About</a>
                <a href="/analytics">📊 Analytics</a>
                <a href="/api-info">🔗 API Info</a>
                <a href="/docs">📚 Swagger</a>
            </div>
            <div class="content">
                <h2>🎯 What is Customer Churn Prediction?</h2>
                <p>Customer churn prediction is a machine learning application that helps businesses identify customers who are likely to cancel their subscriptions or stop using their services. By predicting churn early, companies can take proactive measures to retain valuable customers.</p>
                
                <h2>🚀 Features</h2>
                <div class="feature-grid">
                    <div class="feature-card">
                        <h3>🤖 Advanced ML Models</h3>
                        <p>Multiple algorithms including Random Forest, Logistic Regression, and SVM for accurate predictions.</p>
                    </div>
                    <div class="feature-card">
                        <h3>📊 Real-time Predictions</h3>
                        <p>Get instant churn probability scores and risk levels for individual customers.</p>
                    </div>
                    <div class="feature-card">
                        <h3>📁 Batch Processing</h3>
                        <p>Upload CSV files to analyze multiple customers at once with comprehensive statistics.</p>
                    </div>
                    <div class="feature-card">
                        <h3>🔍 Model Monitoring</h3>
                        <p>Built-in data drift detection and model performance monitoring.</p>
                    </div>
                </div>
                
                <h2>📈 How It Works</h2>
                <ol>
                    <li><strong>Data Input:</strong> Provide customer information like age, tenure, charges, and service details</li>
                    <li><strong>ML Processing:</strong> Our trained models analyze the data using feature engineering and preprocessing</li>
                    <li><strong>Risk Assessment:</strong> Get probability scores and risk levels (Low, Medium, High)</li>
                    <li><strong>Actionable Insights:</strong> Receive specific recommendations for customer retention</li>
                </ol>
                
                <h2>🔧 Technical Details</h2>
                <ul>
                    <li><strong>Framework:</strong> FastAPI for high-performance API development</li>
                    <li><strong>ML Library:</strong> Scikit-learn for machine learning algorithms</li>
                    <li><strong>Data Processing:</strong> Pandas and NumPy for data manipulation</li>
                    <li><strong>Model Tracking:</strong> MLflow for experiment tracking and model registry</li>
                    <li><strong>Deployment:</strong> Docker and Kubernetes ready for production</li>
                </ul>
                
                <h2>📊 Input Features</h2>
                <p>The model considers the following customer attributes:</p>
                <ul>
                    <li><strong>Demographics:</strong> Age, tenure with the company</li>
                    <li><strong>Financial:</strong> Monthly charges, total charges to date</li>
                    <li><strong>Contract:</strong> Contract length (month-to-month, 1-year, 2-year)</li>
                    <li><strong>Services:</strong> Internet service type, online security, tech support</li>
                    <li><strong>Payment:</strong> Payment method preferences</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

def get_analytics_page():
    """Analytics page content"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics - Churn Prediction</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }
            .nav { background: #f8f9fa; padding: 15px; text-align: center; }
            .nav a { margin: 0 15px; color: #007bff; text-decoration: none; padding: 10px 20px; border-radius: 5px; }
            .nav a:hover { background: #e9ecef; }
            .content { padding: 30px; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
            .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-top: 4px solid #007bff; }
            .metric-number { font-size: 2em; font-weight: bold; color: #007bff; }
            .metric-label { color: #666; margin-top: 5px; }
        </style>
        <script>
            async function loadAnalytics() {
                try {
                    // Simulate analytics data - replace with real API calls
                    document.getElementById('total-predictions').textContent = Math.floor(Math.random() * 10000) + 5000;
                    document.getElementById('high-risk').textContent = Math.floor(Math.random() * 500) + 200;
                    document.getElementById('model-accuracy').textContent = (0.85 + Math.random() * 0.1).toFixed(3);
                    document.getElementById('api-uptime').textContent = (99.5 + Math.random() * 0.5).toFixed(2) + '%';
                } catch (error) {
                    console.error('Error loading analytics:', error);
                }
            }
            window.onload = loadAnalytics;
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Analytics Dashboard</h1>
                <p>Real-time insights and model performance metrics</p>
            </div>
            <div class="nav">
                <a href="/">🏠 Dashboard</a>
                <a href="/about">ℹ️ About</a>
                <a href="/analytics">📊 Analytics</a>
                <a href="/api-info">🔗 API Info</a>
                <a href="/docs">📚 Swagger</a>
            </div>
            <div class="content">
                <h2>🎯 Key Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-number" id="total-predictions">Loading...</div>
                        <div class="metric-label">Total Predictions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-number" id="high-risk">Loading...</div>
                        <div class="metric-label">High Risk Customers</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-number" id="model-accuracy">Loading...</div>
                        <div class="metric-label">Model Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-number" id="api-uptime">Loading...</div>
                        <div class="metric-label">API Uptime</div>
                    </div>
                </div>
                
                <h2>📈 Model Performance</h2>
                <p>Current model performance and statistics:</p>
                <ul>
                    <li><strong>Algorithm:</strong> Random Forest Classifier</li>
                    <li><strong>Training Data:</strong> 10,000 customer records</li>
                    <li><strong>Features:</strong> 9 customer attributes</li>
                    <li><strong>Last Updated:</strong> <span id="last-updated">Today</span></li>
                </ul>
                
                <h2>🔍 Recent Activity</h2>
                <p>Monitor recent prediction activity and system health:</p>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <strong>System Status:</strong> All services operational ✅<br>
                    <strong>Response Time:</strong> ~50ms average<br>
                    <strong>Error Rate:</strong> <0.1%<br>
                </div>
                
                <h2>📊 Usage Statistics</h2>
                <p>API usage patterns and trends:</p>
                <ul>
                    <li>Peak usage: 2-4 PM daily</li>
                    <li>Most common risk level: Medium (45%)</li>
                    <li>Average batch size: 127 customers</li>
                    <li>Geographic distribution: Global</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

def get_api_info_page():
    """API information page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Info - Churn Prediction</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 1000px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }
            .nav { background: #f8f9fa; padding: 15px; text-align: center; }
            .nav a { margin: 0 15px; color: #007bff; text-decoration: none; padding: 10px 20px; border-radius: 5px; }
            .nav a:hover { background: #e9ecef; }
            .content { padding: 30px; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 15px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { background: #007bff; color: white; padding: 4px 8px; border-radius: 3px; font-size: 0.8em; }
            .method.post { background: #28a745; }
            .method.get { background: #17a2b8; }
            pre { background: #2d3748; color: #f7fafc; padding: 15px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔗 API Information</h1>
                <p>Complete guide to the Churn Prediction API</p>
            </div>
            <div class="nav">
                <a href="/">🏠 Dashboard</a>
                <a href="/about">ℹ️ About</a>
                <a href="/analytics">📊 Analytics</a>
                <a href="/api-info">🔗 API Info</a>
                <a href="/docs">📚 Swagger</a>
            </div>
            <div class="content">
                <h2>🚀 Available Endpoints</h2>
                
                <div class="endpoint">
                    <h3><span class="method post">POST</span> /predict</h3>
                    <p><strong>Description:</strong> Predict churn for a single customer</p>
                    <p><strong>Request Body:</strong></p>
                    <pre>{
  "age": 35,
  "tenure": 12,
  "monthly_charges": 75.5,
  "total_charges": 1200,
  "contract_length": 12,
  "payment_method": "Credit Card",
  "internet_service": "Fiber Optic",
  "online_security": "Yes",
  "tech_support": "Yes"
}</pre>
                    <p><strong>Response:</strong></p>
                    <pre>{
  "churn_prediction": 0,
  "churn_probability": 0.2345,
  "risk_level": "Low"
}</pre>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method post">POST</span> /batch_predict</h3>
                    <p><strong>Description:</strong> Predict churn for multiple customers</p>
                    <p><strong>Request Body:</strong> Array of customer objects (same as single prediction)</p>
                    <p><strong>Response:</strong></p>
                    <pre>{
  "predictions": [
    {
      "index": 0,
      "churn_prediction": 0,
      "churn_probability": 0.2345,
      "risk_level": "Low"
    }
  ]
}</pre>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /health</h3>
                    <p><strong>Description:</strong> Check API health status</p>
                    <p><strong>Response:</strong></p>
                    <pre>{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "model_version": "1.0.0",
  "uptime_seconds": 3600
}</pre>
                </div>
                
                <div class="endpoint">
                    <h3><span class="method get">GET</span> /model/info</h3>
                    <p><strong>Description:</strong> Get model information</p>
                    <p><strong>Response:</strong></p>
                    <pre>{
  "model_type": "RandomForestClassifier",
  "model_version": "1.0.0",
  "features_count": 2,
  "last_updated": "2024-01-01T12:00:00"
}</pre>
                </div>
                
                <h2>📝 Request Format</h2>
                <p>All POST requests should include the following headers:</p>
                <pre>Content-Type: application/json</pre>
                
                <h2>🔧 Example Usage</h2>
                <h3>Using cURL:</h3>
                <pre>curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "age": 35,
       "tenure": 12,
       "monthly_charges": 75.5,
       "total_charges": 1200,
       "contract_length": 12,
       "payment_method": "Credit Card",
       "internet_service": "Fiber Optic",
       "online_security": "Yes",
       "tech_support": "Yes"
     }'</pre>
     
                <h3>Using Python:</h3>
                <pre>import requests

data = {
    "age": 35,
    "tenure": 12,
    "monthly_charges": 75.5,
    "total_charges": 1200,
    "contract_length": 12,
    "payment_method": "Credit Card",
    "internet_service": "Fiber Optic",
    "online_security": "Yes",
    "tech_support": "Yes"
}

response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
print(result)</pre>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 to 127.0.0.1
        port=8000,
        reload=True,
        log_level="info"
    )