from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from src.pipeline.unified_pipeline import detect_scam
from src.logger import logger
from src.db import init_db

app = FastAPI(title="Aegis Scam Detector API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    await init_db()
    logger.info("Initializing global SQLite boundaries")

class DetectionResponse(BaseModel):
    project: str
    offchain_risk: float
    onchain_risk: float
    final_risk: float
    severity: str
    confidence: float
    trend: str
    action: str
    explanation: List[str]
    error: Optional[str] = None
    details: Optional[str] = None

@app.get("/detect", response_model=DetectionResponse)
async def detect(project: str):
    """Analyzes a crypto project and returns its scam risk profile."""
    if not project or len(project) < 2:
        raise HTTPException(status_code=400, detail="Invalid project name. Please provide a valid query.")
        
    logger.info(f"API Request - Detect: {project}")
    result = await detect_scam(project)
    
    if "error" in result:
        # We can still return 200 with error metadata or 500 depending on the use case
        # Here we raise a 500 so the client knows it clearly failed.
        raise HTTPException(status_code=500, detail=result.get("error"))
        
    return result
