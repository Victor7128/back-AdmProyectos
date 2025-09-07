#Filtro yape
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

class NotYapeTransaction(Exception):
    pass

def is_yape_transaction(
    image_bytes: bytes,
    purple_ratio_thresh: float = 0.2,
    white_ratio_thresh: float = 0.3
) -> bool:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("Imagen corrupta o formato no soportado.")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    purple_ratio = cv2.countNonZero(mask_purple) / (img.shape[0] * img.shape[1])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    white_ratio = cv2.countNonZero(mask_white) / (img.shape[0] * img.shape[1])
    if purple_ratio > purple_ratio_thresh and white_ratio > white_ratio_thresh:
        return True

    raise NotYapeTransaction(
        f"No es una imagen de transacción valida"
    )

@router.post("/filter_yape")
async def filter_yape(file: UploadFile = File(...)):
    img_bytes = await file.read()
    try:
        is_yape_transaction(img_bytes)
        return {"resultado": "ok"}
    except NotYapeTransaction as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {e}")