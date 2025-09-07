# Auth
import os
import uuid
import secrets
from datetime import datetime
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from typing import Optional
from bd import users_coll, sessions_coll
from dotenv import load_dotenv
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

#Tokens
async def save_token(user_id, token):
    session_id = str(uuid.uuid4())
    await sessions_coll.insert_one({
        "user_id": user_id,
        "token": token,
        "session_id": session_id,
        "created_at": datetime.utcnow()
    })
    return session_id

async def revoke_token(token):
    await sessions_coll.delete_one({"token": token})
    
async def revoke_all_tokens(user_id):
    await sessions_coll.delete_many({"user_id": user_id})

async def is_token_valid(token):
    session = await sessions_coll.find_one({"token": token})
    return session is not None

#reseteo de contraseña
async def create_password_reset_token(email: str) -> Optional[str]:
    """Crea un token de reseteo de contraseña con caducidad de 1 hora."""
    # Verificar que el usuario existe
    user = await users_coll.find_one({"email": email})
    if not user:
        # No indicamos si el usuario existe o no por seguridad
        return None
        
    # Generar token seguro aleatorio
    reset_token = secrets.token_urlsafe(32)
    
    # Guardar el token en la base de datos
    await users_coll.update_one(
        {"email": email},
        {"$set": {
            "reset_token": reset_token,
            "reset_token_expires": datetime.utcnow() + timedelta(hours=1)
        }}
    )
    
    return reset_token

async def verify_reset_token(token: str) -> Optional[str]:
    """Verifica si el token de reseteo es válido y devuelve el email asociado."""
    user = await users_coll.find_one({
        "reset_token": token,
        "reset_token_expires": {"$gt": datetime.utcnow()}
    })
    
    if not user:
        return None
        
    return user["email"]

async def reset_password(token: str, new_password: str) -> bool:
    """Resetea la contraseña usando un token válido."""
    email = await verify_reset_token(token)
    if not email:
        return False
        
    # Hash de la nueva contraseña
    hashed_pw = get_password_hash(new_password)
    
    # Actualizar contraseña y eliminar el token de reseteo
    await users_coll.update_one(
        {"email": email},
        {
            "$set": {"hashed_password": hashed_pw},
            "$unset": {"reset_token": "", "reset_token_expires": ""}
        }
    )
    
    return True

#Funciones
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(sub: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = {"exp": expire, "sub": sub}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error_code": "UNAUTHORIZED", "message": "Credenciales inválidas"},
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        if not await is_token_valid(token):
            raise credentials_exception
            
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await users_coll.find_one({"email": email})
    if not user:
        raise credentials_exception
    return user