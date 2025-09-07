# Login
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from bson import ObjectId
from bd import users_coll
from auth import create_password_reset_token, reset_password
from typing import Optional
from schemas import UserCreate, UserOut, Token, PasswordChange, RequestPasswordReset, ConfirmPasswordReset
from auth import get_password_hash, verify_password, create_access_token, get_current_user, save_token, revoke_token, revoke_all_tokens

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    if await users_coll.find_one({"email": user.email}):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
            detail={"error_code":"USER_EXISTS","message":"Email ya registrado"})
    if await users_coll.find_one({"username": user.username}):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
            detail={"error_code":"USERNAME_TAKEN","message":"Usuario ya en uso"})
    hashed_pw = get_password_hash(user.password)
    res = await users_coll.insert_one({
        "email": user.email,
        "username": user.username,
        "name": user.name,
        "hashed_password": hashed_pw
    })
    return UserOut(id=str(res.inserted_id), email=user.email, username=user.username, name=user.name)

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Intentamos buscar por email primero
    user = await users_coll.find_one({"email": form_data.username})
    # Si no encontramos por email, intentamos por username
    if not user:
        user = await users_coll.find_one({"username": form_data.username})
    
    # Verificamos si el usuario existe y la contraseña es correcta
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
            detail={"error_code":"INVALID_CREDENTIALS","message":"Email o contraseña incorrectos"},
            headers={"WWW-Authenticate": "Bearer"})
    
    # Generamos el token con el email del usuario
    token = create_access_token(sub=user["email"])
    
    # Guardamos el token en la colección de sesiones
    session_id = await save_token(str(user["_id"]), token)
    
    return {"access_token": token, "token_type": "bearer", "session_id": session_id}

@router.post("/logout")
async def logout(current_user=Depends(get_current_user), token: str = Depends(oauth2_scheme)):
    await revoke_token(token)
    return {"message": "Sesión cerrada correctamente"}

@router.post("/logout-all")
async def logout_all(current_user=Depends(get_current_user)):
    await revoke_all_tokens(str(current_user["_id"]))
    return {"message": "Todas las sesiones fueron cerradas correctamente"}

@router.put("/change-password")
async def change_password(data: PasswordChange, current_user=Depends(get_current_user)):
    if not verify_password(data.old_password, current_user["hashed_password"]):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
            detail={"error_code":"INVALID_OLD_PASSWORD","message":"Contraseña antigua incorrecta"})
    new_hashed = get_password_hash(data.new_password)
    await users_coll.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": {"hashed_password": new_hashed}}
    )
    return {"message": "Contraseña actualizada correctamente"}

@router.post("/request-password-reset")
async def request_password_reset(request_data: RequestPasswordReset):
    """Solicitar un token para resetear la contraseña."""
    user = await users_coll.find_one({"email": request_data.email})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": "EMAIL_NOT_FOUND", "message": "Correo no encontrado"}
        )

    reset_token = await create_password_reset_token(request_data.email)
    
    return {
        "message": "Correo encontrado!",
        "found": True,
        "debug_token": reset_token
    }

@router.post("/reset-password")
async def confirm_password_reset(reset_data: ConfirmPasswordReset):
    success = await reset_password(reset_data.token, reset_data.new_password)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "INVALID_RESET_TOKEN", "message": "Token inválido o expirado"}
        )
    
    return {"message": "Contraseña restablecida correctamente"}

@router.get("/me", response_model=UserOut)
async def read_me(current_user=Depends(get_current_user)):
    return UserOut(
        id=str(current_user["_id"]),
        email=current_user["email"],
        username=current_user["username"],
        name=current_user["name"]
    )

@router.post("/verify-token", response_model=UserOut)
async def verify_token(token: str = Depends(oauth2_scheme)):
    """Verifica el token y devuelve el usuario asociado."""
    user = await get_current_user(token)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "INVALID_TOKEN", "message": "Token inválido o expirado"}
        )
    
    return UserOut(
        id=str(user["_id"]),
        email=user["email"],
        username=user["username"],
        name=user["name"]
    )
