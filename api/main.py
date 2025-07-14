from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
            from src.pipeline.training_pipeline import TrainingPipeline
            training_pipeline = TrainingPipeline()
            training_pipeline.run_training_pipeline()
        
        prediction_pipeline = PredictionPipeline()
        logger.info("Prediction pipeline loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize prediction pipeline: {str(e)}")

# FRONTEND ROUTES
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api", response_class=HTMLResponse)
async def api_info():
    """API information page"""
    html_content = """
    <html>
        <head>
            <title>Churn Prediction API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 Customer Churn Prediction API</h1>
                <p>Welcome to the Customer Churn Prediction API!</p>
                <h3>📖 Available Endpoints:</h3>
                <ul>
                    <li><a href="/">🏠 Main Dashboard</a></li>
                    <li><a href="/docs">📚 API Documentation (Swagger)</a></li>
                    <li><a href="/redoc">📖 API Documentation (ReDoc)</a></li>
                    <li><a href="/health">❤️ Health Check</a></li>
                    <li><a href="/model/info">🤖 Model Information</a></li>
                </ul>
                <br>
                <a href="/" class="btn">🔙 Back to Dashboard</a>
            </div>
        </body>
    </html>
    """
    return html_content

# API ROUTES (your existing routes)
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )