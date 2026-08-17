from pydantic import BaseModel


class LoginRequest(BaseModel):
    user_id: str
    password: str
    user_type: str


class LoginResponse(BaseModel):
    success: bool
    message: str
    data: dict | None = None