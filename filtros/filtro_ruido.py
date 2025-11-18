import cv2
import tempfile
import os
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

def porcentaje_nitidez(image_path, max_var=400):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("No se pudo cargar la imagen.")
    
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    
    porcentaje = min(100.0, (laplacian_var / max_var) * 100.0)
    return porcentaje

def clasificar_autenticidad(porcentaje):
    if porcentaje < 50:
        return "Alterado"
    elif porcentaje < 70:
        return "Sospechoso"
    else:
        return "Auténtico"

@router.post("/filtro_ruido")
async def filtro_ruido(file: UploadFile = File(...)):
    tmp_path = None
    
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=422, 
                detail="❌ El archivo debe ser una imagen válida (JPG, PNG, etc.)"
            )

        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=422,
                detail="❌ El archivo está vacío"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            porcentaje = porcentaje_nitidez(tmp_path)
        except ValueError as e:
            raise HTTPException(
                status_code=422, 
                detail=f"❌ Error al procesar la imagen: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"❌ Error interno al analizar la imagen: {str(e)}"
            )

        advertencia = clasificar_autenticidad(porcentaje)
        es_valido = advertencia == "Auténtico"

        return {
            "porcentaje_nitidez": round(porcentaje, 2),
            "advertencia": advertencia,
            "es_valido": es_valido,
            "mensaje": f"El recibo es clasificado como: {advertencia}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"❌ Error inesperado: {str(e)}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass