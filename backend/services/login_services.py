"""Login services for auth checks against a JSON user store."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import jwt


class LoginService:
    """Service for handling login business logic."""

    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        self.jwt_algorithm = "HS256"
        self.token_expiry_hours = 24
        self.users_file = Path(__file__).resolve().parent.parent / "data" / "users.json"

    def _load_users(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        if not self.users_file.exists():
            return {"admin": {}, "noc": {}}

        with open(self.users_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return {"admin": {}, "noc": {}}

        return data

    def validate_credentials(self, user_id: str, password: str, user_type: str) -> Tuple[bool, str]:
        if not user_id or not user_id.strip():
            return False, "User ID is required"

        if not password or not password.strip():
            return False, "Password is required"

        if user_type not in ["noc", "admin"]:
            return False, "Invalid user type"

        user_data = self._get_user_from_db(user_id, user_type)
        if not user_data:
            return False, "Invalid user ID or password"

        if user_data.get("password") != password:
            return False, "Invalid user ID or password"

        return True, ""

    def authenticate_user(self, user_id: str, password: str, user_type: str) -> Tuple[Optional[Dict], Optional[str]]:
        is_valid, error_msg = self.validate_credentials(user_id, password, user_type)

        if not is_valid:
            return None, error_msg

        user_data = self._get_user_from_db(user_id, user_type)
        if not user_data:
            return None, "User not found"

        token = self._generate_token(user_id, user_type, user_data)

        response_data = {
            "user_id": user_id,
            "user_type": user_type,
            "name": user_data.get("name", ""),
            "email": user_data.get("email", ""),
            "token": token,
            "expires_in": self.token_expiry_hours * 3600,
        }

        return response_data, None

    def verify_token(self, token: str) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            decoded = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return decoded, None
        except jwt.ExpiredSignatureError:
            return None, "Token has expired"
        except jwt.InvalidTokenError:
            return None, "Invalid token"

    def _generate_token(self, user_id: str, user_type: str, user_data: Dict) -> str:
        payload = {
            "user_id": user_id,
            "user_type": user_type,
            "name": user_data.get("name", ""),
            "email": user_data.get("email", ""),
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
        }

        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def _get_user_from_db(self, user_id: str, user_type: str) -> Optional[Dict]:
        users = self._load_users()
        user_type = user_type.lower()
        return users.get(user_type, {}).get(user_id)


login_service = LoginService()
