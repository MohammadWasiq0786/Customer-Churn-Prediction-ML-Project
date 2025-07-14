from pydantic import BaseModel, Field
from typing import Optional

class CustomerFeatures(BaseModel):
    age: float = Field(..., ge=18, le=100, description="Customer age")
    tenure: float = Field(..., ge=0, description="Customer tenure in months")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    total_charges: float = Field(..., ge=0, description="Total charges")
    contract_length: int = Field(..., ge=1, le=24, description="Contract length in months")
    payment_method: str = Field(..., description="Payment method")
    internet_service: str = Field(..., description="Internet service type")
    online_security: str = Field(..., description="Online security service")
    tech_support: str = Field(..., description="Tech support service")

    class Config:
        schema_extra = {
            "example": {
                "age": 35.0,
                "tenure": 12.0,
                "monthly_charges": 75.5,
                "total_charges": 1200.0,
                "contract_length": 12,
                "payment_method": "Credit Card",
                "internet_service": "Fiber Optic",
                "online_security": "Yes",
                "tech_support": "Yes"
            }
        }

class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_prediction: int = Field(..., description="0 for no churn, 1 for churn")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_version: str
    uptime_seconds: Optional[float] = None
