import cv2
import numpy as np
import os
import glob
import tempfile
from typing import List, Tuple, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

def alinear_imagen(sospechosa_gray: np.ndarray, plantilla_gray: np.ndarray) -> Tuple[np.ndarray, int]:
    if sospechosa_gray is None or plantilla_gray is None:
        raise ValueError("Una o ambas imágenes están vacías")
    if sospechosa_gray.shape[0] < 50 or sospechosa_gray.shape[1] < 50:
        raise ValueError("La imagen sospechosa es demasiado pequeña")
    if plantilla_gray.shape[0] < 50 or plantilla_gray.shape[1] < 50:
        raise ValueError("La imagen plantilla es demasiado pequeña")
    if sospechosa_gray.shape == plantilla_gray.shape:
        if np.array_equal(sospechosa_gray, plantilla_gray):
            return sospechosa_gray, 1000    
    orb = cv2.ORB.create(nfeatures=1000)
    mask1 = np.ones(plantilla_gray.shape, dtype=np.uint8)
    mask2 = np.ones(sospechosa_gray.shape, dtype=np.uint8)
    kp1, des1 = orb.detectAndCompute(plantilla_gray, mask1)
    kp2, des2 = orb.detectAndCompute(sospechosa_gray, mask2)
    if des1 is None or des2 is None:
        raise ValueError("No se pudieron extraer características de una o ambas imágenes")    
    if len(des1) < 10 or len(des2) < 10:
        raise ValueError(f"Muy pocas características detectadas: plantilla={len(des1) if des1 is not None else 0}, sospechosa={len(des2) if des2 is not None else 0}")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    buenos = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                buenos.append(m)    
    if len(buenos) < 4:
        raise ValueError(f"Pocos matches buenos: {len(buenos)}. Características detectadas: plantilla={len(des1)}, sospechosa={len(des2)}")
    src_pts = np.array([[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] for m in buenos], dtype=np.float32).reshape(-1,1,2)
    dst_pts = np.array([[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in buenos], dtype=np.float32).reshape(-1,1,2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("No se pudo calcular la homografía")
    h, w = plantilla_gray.shape
    alineada = cv2.warpPerspective(sospechosa_gray, H, (w, h))
    return alineada, len(buenos)

def evaluar_similitud(plantilla_path: str, sospechosa_gray: np.ndarray, threshold: int = 30) -> Tuple[Optional[np.ndarray], float, int, str]:
    plantilla = cv2.imread(plantilla_path, cv2.IMREAD_GRAYSCALE)
    if plantilla is None:
        return None, 0.0, 0, f"❌ No se pudo cargar plantilla: {plantilla_path}"
    try:
        alineada, matches = alinear_imagen(sospechosa_gray, plantilla)
        diff = cv2.absdiff(plantilla, alineada)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        pixeles_diferentes = int((mask > 0).sum())
        total_pixeles = mask.size
        similitud = 100.0 - (pixeles_diferentes / total_pixeles * 100.0)
        return mask, similitud, matches, "✅ Comparación exitosa"
    except Exception as e:
        print(f"DEBUG: Error en evaluar_similitud: {e}")
        return None, 0.0, 0, f"❌ Error: {e}"

def detectar_diferencias(plantillas_paths: List[str], sospechosa_gray: np.ndarray, threshold: int = 30):
    resultados = []
    for plantilla_path in plantillas_paths:
        mask, similitud, matches, mensaje = evaluar_similitud(
            plantilla_path, sospechosa_gray, threshold
        )
        if mask is not None:
            resultados.append({
                'plantilla': os.path.basename(plantilla_path),
                'porcentaje': similitud,
                'coincidencias': matches,
                'mensaje': mensaje
            })
    if not resultados:
        return None
    mejor_resultado = max(resultados, key=lambda x: x['porcentaje'])
    return {
        'porcentaje': round(mejor_resultado['porcentaje'], 2),
        'coincidencias': mejor_resultado['coincidencias']
    }

@router.post("/filtro_pixeles")
async def filtro_pixeles(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=422, detail="❌ El archivo debe ser una imagen")

        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            nparr = np.frombuffer(content, np.uint8)
            sospechosa_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if sospechosa_gray is None:
                raise Exception("❌ No se pudo decodificar la imagen subida")
            if sospechosa_gray.size == 0:
                raise Exception("❌ La imagen está vacía")
            plantillas_dir = "./filtros/plantillas/"
            extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            plantillas_paths = []
            for ext in extensiones:
                plantillas_paths.extend(glob.glob(os.path.join(plantillas_dir, ext)))
            if not plantillas_paths:
                raise Exception("❌ No hay plantillas disponibles para comparar")
            threshold = 30
            result = detectar_diferencias(plantillas_paths, sospechosa_gray, threshold)
            if result is None:
                raise Exception("❌ No se pudo comparar la imagen con las plantillas")
            porcentaje = result['porcentaje']
            advertencia = ""
            if porcentaje <= 85:
                advertencia = "Alterado"
            elif porcentaje <= 98:
                advertencia = "Sospechoso"
            else:
                advertencia = "Auténtico"

            return {
                "porcentaje_coincidencia": round(porcentaje, 2),
                "coincidencias": result['coincidencias'],
                "advertencia": advertencia
            }
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"❌ Error al procesar la imagen: {e}")

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"❌ Error al leer el archivo: {e}")