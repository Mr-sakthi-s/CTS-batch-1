import json
import os
from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(prefix="/api/incident", tags=["Incident"])

# Path to our persistent JSON database (saves in the same folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "incident_db.json")

def read_db() -> Dict[str, Any]:
    """Reads the current state from the JSON file."""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def write_db(data: Dict[str, Any]):
    """Writes the state to the JSON file to persist across refreshes."""
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

@router.post("/store")
def store_incident(data: Dict[str, Any]):
    """Stores the raw telemetry incident initially."""
    db = read_db()
    db["data"] = data
    db["prediction"] = None    # Clear previous prediction for the new incident
    db["agent_result"] = None  # Clear previous RCA
    write_db(db)
    return {"success": True, "message": "Incident stored successfully"}

@router.post("/store-prediction")
def store_prediction(payload: Dict[str, Any]):
    """Updates the store with ML predictions and Agent RCA results."""
    db = read_db()
    
    # Merge the new prediction and agent results into the database
    if "prediction" in payload:
        db["prediction"] = payload["prediction"]
    if "agent_result" in payload:
        db["agent_result"] = payload["agent_result"]
        
    write_db(db)
    return {"success": True, "message": "Prediction and RCA stored successfully"}

@router.get("/latest-with-prediction")
def get_latest():
    """Dashboard endpoint: returns the persisted data."""
    db = read_db()
    
    if not db.get("data"):
        return {"success": False, "message": "No incident data available"}
    
    return {
        "success": True,
        "data": db.get("data"),
        "prediction": db.get("prediction"),
        "agent_result": db.get("agent_result")
    }