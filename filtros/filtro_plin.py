# Filtro Plin
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

class NotPlinTransaction(Exception):
    """Excepción lanzada cuando la imagen no corresponde a una transacción Plin."""
    pass

def is_plin_transaction(
    image_bytes: bytes,
    turquoise_ratio_thresh: float = 0.15,
    white_ratio_thresh: float = 0.25
) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None or img.size == 0:
        raise ValueError("Imagen corrupta o formato no soportado.")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_pixels = img.shape[0] * img.shape[1]
    lower_turquoise = np.array([75, 40, 50])
    upper_turquoise = np.array([105, 255, 255])
    mask_turquoise = cv2.inRange(hsv, lower_turquoise, upper_turquoise)
    turquoise_ratio = cv2.countNonZero(mask_turquoise) / total_pixels
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    white_ratio = cv2.countNonZero(mask_white) / total_pixels
    
    if turquoise_ratio >= turquoise_ratio_thresh and white_ratio >= white_ratio_thresh:
        return {
            "es_plin": True,
            "ratio_turquesa": round(turquoise_ratio * 100, 2),
            "ratio_blanco": round(white_ratio * 100, 2),
            "confianza": "Alta" if turquoise_ratio > 0.25 else "Media"
        }
    
    raise NotPlinTransaction(
        f"No es una transacción Plin válida. "
        f"Turquesa detectado: {turquoise_ratio*100:.1f}% (mínimo {turquoise_ratio_thresh*100:.1f}%), "
        f"Blanco detectado: {white_ratio*100:.1f}% (mínimo {white_ratio_thresh*100:.1f}%)"
    )

@router.post("/filter_plin")
async def filter_plin(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=422,
            detail="❌ El archivo debe ser una imagen válida (JPG, PNG, etc.)"
        )
    
    try:
        img_bytes = await file.read()
        if len(img_bytes) == 0:
            raise HTTPException(
                status_code=422,
                detail="❌ El archivo está vacío"
            )
        resultado = is_plin_transaction(img_bytes)
        
        return {
            "resultado": "✅ Transacción Plin válida",
            "es_valido": True,
            **resultado
        }
        
    except NotPlinTransaction as e:
        raise HTTPException(
            status_code=400, 
            detail=f"❌ {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=422, 
            detail=f"❌ {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"❌ Error procesando imagen: {str(e)}"
        )