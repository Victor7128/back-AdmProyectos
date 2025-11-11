# Filtro Plin Calibrado
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

class NotPlinTransaction(Exception):
    pass

def is_plin_transaction(
    image_bytes: bytes,
    turquoise_ratio_thresh: float = 0.025,  # 2.5% (calibrado desde 2.8%)
    white_ratio_thresh: float = 0.65        # 65% (calibrado desde 66.9%)
) -> dict:
    """
    Detecta si una imagen es una transacción Plin válida
    
    Parámetros calibrados basados en análisis real de capturas Plin:
    - Turquesa: mínimo 2.5% (rango amplio H[70-110])
    - Blanco: mínimo 65% (rango flexible para fondos claros)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None or img.size == 0:
        raise ValueError("Imagen corrupta o formato no soportado.")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_pixels = img.shape[0] * img.shape[1]
    
    # Rangos HSV optimizados (calibrados con imágenes reales)
    lower_turquoise = np.array([70, 25, 35])
    upper_turquoise = np.array([110, 255, 255])
    
    mask_turquoise = cv2.inRange(hsv, lower_turquoise, upper_turquoise)
    turquoise_ratio = cv2.countNonZero(mask_turquoise) / total_pixels
    
    # Rango blanco flexible para fondos claros
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    white_ratio = cv2.countNonZero(mask_white) / total_pixels
    
    # Detección complementaria de azul/cyan (opcional, suma confianza)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_ratio = cv2.countNonZero(mask_blue) / total_pixels
    
    lower_cyan = np.array([85, 50, 50])
    upper_cyan = np.array([95, 255, 255])
    mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
    cyan_ratio = cv2.countNonZero(mask_cyan) / total_pixels
    
    # Colores Plin combinados
    combined_color_ratio = turquoise_ratio + blue_ratio + cyan_ratio
    
    # Validación principal
    is_valid = (
        combined_color_ratio >= turquoise_ratio_thresh and 
        white_ratio >= white_ratio_thresh
    )
    
    if is_valid:
        # Calcular nivel de confianza
        color_score = combined_color_ratio / turquoise_ratio_thresh
        white_score = white_ratio / white_ratio_thresh
        confidence_score = (color_score + white_score) / 2
        
        if confidence_score >= 2.0:
            confidence = "Alta"
        elif confidence_score >= 1.3:
            confidence = "Media"
        else:
            confidence = "Baja"
        
        return {
            "es_plin": True,
            "ratio_turquesa": round(turquoise_ratio * 100, 2),
            "ratio_azul": round(blue_ratio * 100, 2),
            "ratio_cyan": round(cyan_ratio * 100, 2),
            "ratio_colores_plin": round(combined_color_ratio * 100, 2),
            "ratio_blanco": round(white_ratio * 100, 2),
            "confianza": confidence,
            "score_confianza": round(confidence_score, 2)
        }
    
    raise NotPlinTransaction(
        f"No es una transacción Plin válida. "
        f"Colores Plin detectados: {combined_color_ratio*100:.2f}% "
        f"(mínimo {turquoise_ratio_thresh*100:.1f}%), "
        f"Blanco detectado: {white_ratio*100:.2f}% "
        f"(mínimo {white_ratio_thresh*100:.1f}%). "
        f"[Turquesa: {turquoise_ratio*100:.2f}%, Azul: {blue_ratio*100:.2f}%, "
        f"Cyan: {cyan_ratio*100:.2f}%]"
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

    """
    Endpoint de análisis detallado (para testing y debug)
    Retorna todas las métricas sin validar
    """
    try:
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=422, detail="Imagen inválida")
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        total = img.shape[0] * img.shape[1]
        
        # Detectar todos los rangos
        masks = {
            "turquesa_amplio": ([70, 25, 35], [110, 255, 255]),
            "azul": ([100, 50, 50], [130, 255, 255]),
            "cyan": ([85, 50, 50], [95, 255, 255]),
            "blanco_flexible": ([0, 0, 180], [180, 40, 255]),
        }
        
        resultados = {}
        for nombre, (lower, upper) in masks.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            ratio = (cv2.countNonZero(mask) / total) * 100
            resultados[nombre] = ratio
        
        colores_plin = resultados["turquesa_amplio"] + resultados["azul"] + resultados["cyan"]
        
        return {
            "dimensiones": f"{img.shape[1]}x{img.shape[0]}",
            "total_pixels": total,
            "detecciones": resultados,
            "colores_plin_combinados": round(colores_plin, 2),
            "pasa_validacion": {
                "colores": colores_plin >= 2.5,
                "blanco": resultados["blanco_flexible"] >= 65.0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))