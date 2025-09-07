import cv2
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

def porcentaje_nitidez(image_path, max_var=399):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("No se pudo cargar la imagen.")
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    porcentaje = min(100.0, (laplacian_var / max_var) * 100.0)
    return porcentaje

@router.post("/filtro_ruido")
async def filtro_ruido(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=422, detail="❌ El archivo debe ser una imagen")

        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            porcentaje = porcentaje_nitidez(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"❌ Error al procesar la imagen: {e}")

        advertencia = ""
        if porcentaje < 70:
            advertencia = "Alterado"
        elif porcentaje < 90:
            advertencia = "Sospechoso"
        else:
            advertencia = "Auténtico"

        return {
            "porcentaje_nitidez": round(porcentaje, 2),
            "advertencia": advertencia
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"❌ Error al leer el archivo: {e}")