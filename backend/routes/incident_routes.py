from fastapi import APIRouter

from backend.controllers.incident_controllers import incident_controller

router = APIRouter(prefix="/api/incident", tags=["incident"])


@router.post("/store")
def store_incident(payload: dict):
    return incident_controller.store(payload)


@router.post("/store-prediction")
def store_prediction(payload: dict):
    return incident_controller.store_prediction(payload)


@router.get("/latest")
def get_latest_incident():
    return incident_controller.latest()


@router.get("/latest-prediction")
def get_latest_prediction():
    return incident_controller.latest_prediction()


@router.get("/latest-with-prediction")
def get_latest_incident_with_prediction():
    return incident_controller.latest_with_prediction()


__all__ = ["router"]
