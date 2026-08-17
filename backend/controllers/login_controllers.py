"""Controller layer for authentication flows."""

from typing import Optional

from fastapi import HTTPException, Header

from backend.schemas.login_schema import LoginRequest
from backend.services.login_services import login_service


class LoginController:
    """Handle authentication HTTP logic."""

    def login(self, payload: LoginRequest):
        user_data, error = login_service.authenticate_user(
            payload.user_id.strip(),
            payload.password.strip(),
            payload.user_type.strip().lower(),
        )

        if error:
            raise HTTPException(status_code=401, detail=error)

        return {
            "success": True,
            "message": "Login successful",
            "data": user_data,
        }

    def verify_token(self, authorization: Optional[str] = Header(default=None, alias="Authorization")):
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header is required")

        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization header format")

        token = parts[1]
        user_data, error = login_service.verify_token(token)

        if error:
            raise HTTPException(status_code=401, detail=error)

        return {
            "success": True,
            "message": "Token is valid",
            "data": user_data,
        }

    def logout(self, authorization: Optional[str] = Header(default=None, alias="Authorization")):
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header is required")

        return {
            "success": True,
            "message": "Logout successful",
        }


login_controller = LoginController()
