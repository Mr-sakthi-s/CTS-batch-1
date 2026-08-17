"""Login routes for FastAPI."""

from typing import Optional

from fastapi import APIRouter, Header

from backend.controllers.login_controllers import login_controller
from backend.schemas.login_schema import LoginRequest

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login")
def login(payload: LoginRequest):
    return login_controller.login(payload)


@router.get("/verify-token")
def verify_token(authorization: Optional[str] = Header(default=None, alias="Authorization")):
    return login_controller.verify_token(authorization)


@router.post("/logout")
def logout(authorization: Optional[str] = Header(default=None, alias="Authorization")):
    return login_controller.logout(authorization)


__all__ = ["router"]
