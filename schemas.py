# Schemas
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    name: str = Field(..., min_length=1)
    password: str = Field(..., min_length=8)

class UserOut(BaseModel):
    id: str
    email: EmailStr
    username: str
    name: str

class Token(BaseModel):
    access_token: str
    token_type: str
    session_id: Optional[str] = None

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

class RequestPasswordReset(BaseModel):
    email: EmailStr

class ConfirmPasswordReset(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)