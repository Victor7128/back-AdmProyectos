import cv2
import numpy as np
import os
import tempfile
from typing import Dict, List, Optional, Tuple
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from enum import Enum

router = APIRouter()

LOGO_PLIN_PATH = "./filtros/logo.jpg"
PLANTILLAS_DIR = "./filtros/plantillas/plin.jpg"

class TipoLogo(str, Enum):
    PLIN = "plin"
    INTERBANK = "interbank"
    AMBOS = "ambos"

def detectar_cuadro_blanco(imagen: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    if imagen is None or imagen.size == 0:
        return imagen, None
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, umbral = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return imagen, None
    mayor_area = 0
    cuadro_blanco_box = None
    
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 1000 and area > mayor_area:
            mayor_area = area
            x, y, w, h = cv2.boundingRect(contorno)
            cuadro_blanco_box = (x, y, w, h)
    
    return imagen, cuadro_blanco_box

def detectar_logo_multiescala(
    imagen: np.ndarray,
    logo: np.ndarray,
    umbral: float = 0.65
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], float]:
    if imagen is None or logo is None:
        return imagen, None, 0.0
    
    img_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    
    mejor_confianza = 0
    mejor_top_left = None
    mejor_bottom_right = None
    mejor_escala = 1.0

    for escala in np.linspace(0.3, 2.0, 30)[::-1]:
        ancho_nuevo = int(logo_gray.shape[1] * escala)
        alto_nuevo = int(logo_gray.shape[0] * escala)
        if ancho_nuevo < 10 or alto_nuevo < 10:
            continue
        logo_redimensionado = cv2.resize(
            logo_gray,
            (ancho_nuevo, alto_nuevo),
            interpolation=cv2.INTER_AREA
        )
        if (img_gray.shape[0] < logo_redimensionado.shape[0] or 
            img_gray.shape[1] < logo_redimensionado.shape[1]):
            continue
        resultado = cv2.matchTemplate(
            img_gray,
            logo_redimensionado,
            cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(resultado)
        
        if max_val > mejor_confianza:
            mejor_confianza = max_val
            mejor_top_left = max_loc
            mejor_bottom_right = (max_loc[0] + ancho_nuevo, max_loc[1] + alto_nuevo)
            mejor_escala = escala
    
    if mejor_confianza >= umbral and mejor_top_left and mejor_bottom_right:
        x, y = mejor_top_left
        w = mejor_bottom_right[0] - x
        h = mejor_bottom_right[1] - y
        return imagen, (x, y, w, h), mejor_confianza
    
    return imagen, None, mejor_confianza

def remarcar_contorno_recibo(imagen: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    if imagen is None or imagen.size == 0:
        return imagen, None
    
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, umbral = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY_INV)
    
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return imagen, None
    contorno_mayor = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_mayor)
    
    return imagen, (x, y, w, h)

def calcular_distancias(
    logo_box: Tuple[int, int, int, int],
    borde_box: Tuple[int, int, int, int],
    cuadro_blanco_box: Tuple[int, int, int, int]
) -> Tuple[Dict[str, int], Tuple[int, int], Dict[str, Tuple[int, int]]]:
    lx, ly, lw, lh = logo_box
    bx, by, bw, bh = borde_box
    _, y_cuadro_blanco, _, _ = cuadro_blanco_box
    centro_logo = (lx + lw // 2, ly + lh // 2)
    puntos_borde = {
        "Izquierda": (bx, centro_logo[1]),
        "Derecha": (bx + bw, centro_logo[1]),
        "Arriba": (centro_logo[0], by),
        "Abajo": (centro_logo[0], y_cuadro_blanco),
    }
    distancias = {
        lado: int(np.linalg.norm(np.array(centro_logo) - np.array(punto)))
        for lado, punto in puntos_borde.items()
    }
    
    return distancias, centro_logo, puntos_borde

def procesar_imagen_plantilla(
    ruta_plantilla: str,
    logo: np.ndarray,
    factor_escala: float = 0.6
) -> Optional[Dict[str, int]]:
    if not os.path.exists(ruta_plantilla):
        return None    
    imagen = cv2.imread(ruta_plantilla)
    if imagen is None:
        return None
    imagen = cv2.resize(
        imagen,
        None,
        fx=factor_escala,
        fy=factor_escala,
        interpolation=cv2.INTER_AREA
    )
    imagen, cuadro_blanco_box = detectar_cuadro_blanco(imagen)
    imagen, pos_logo, _ = detectar_logo_multiescala(imagen, logo)
    imagen, borde_recibo = remarcar_contorno_recibo(imagen)
    if pos_logo and borde_recibo and cuadro_blanco_box:
        distancias, _, _ = calcular_distancias(pos_logo, borde_recibo, cuadro_blanco_box)
        return distancias
    
    return None

def calcular_porcentaje_cambio(
    distancias_nueva: Dict[str, int],
    distancias_plantilla: Dict[str, int]
) -> Tuple[float, Dict]:
    if not distancias_nueva or not distancias_plantilla:
        return float('inf'), {}    
    cambios = []
    comparacion = {}    
    for lado in distancias_nueva:
        if lado in distancias_plantilla:
            val_nueva = distancias_nueva[lado]
            val_plantilla = distancias_plantilla[lado]
            
            if val_plantilla != 0:
                diferencia = val_nueva - val_plantilla
                porcentaje = abs(diferencia / val_plantilla) * 100
                cambios.append(porcentaje)
                
                comparacion[lado] = {
                    "distancia_actual": val_nueva,
                    "distancia_plantilla": val_plantilla,
                    "diferencia_px": diferencia,
                    "cambio_porcentaje": round(porcentaje, 2)
                }    
    if cambios:
        porcentaje_promedio = sum(cambios) / len(cambios)
        return porcentaje_promedio, comparacion    
    return float('inf'), {}

def obtener_plantillas_plin() -> List[str]:
    if not os.path.exists(PLANTILLAS_DIR):
        return []    
    extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    plantillas = []    
    import glob
    for ext in extensiones:
        plantillas.extend(glob.glob(os.path.join(PLANTILLAS_DIR, ext)))    
    return plantillas

def clasificar_autenticidad(porcentaje_cambio: float) -> str:
    if porcentaje_cambio <= 10:
        return "Auténtico"
    elif porcentaje_cambio <= 25:
        return "Sospechoso"
    else:
        return "Alterado"

@router.post("/filtro_logo")
async def filtro_logo(
    file: UploadFile = File(...),
    tipo_logo: TipoLogo = TipoLogo.PLIN
):
    temp_path = None    
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=422,
                detail="❌ El archivo debe ser una imagen válida"
            )
        archivos_faltantes = []
        logos_a_usar = []
        
        if tipo_logo in [TipoLogo.PLIN, TipoLogo.AMBOS]:
            if os.path.exists(LOGO_PLIN_PATH):
                logos_a_usar.append(("Plin", LOGO_PLIN_PATH))
            else:
                archivos_faltantes.append(f"Logo Plin: {LOGO_PLIN_PATH}")
        
        if not logos_a_usar:
            raise HTTPException(
                status_code=500,
                detail=f"❌ No se encontraron logos. Faltantes: {', '.join(archivos_faltantes)}"
            )
        plantillas = obtener_plantillas_plin()
        
        if not plantillas:
            raise HTTPException(
                status_code=500,
                detail=f"❌ No se encontraron plantillas en: {PLANTILLAS_DIR}"
            )
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=422,
                detail="❌ El archivo está vacío"
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(content)
            temp_path = temp.name
        
        imagen = cv2.imread(temp_path)
        if imagen is None:
            raise HTTPException(
                status_code=422,
                detail="❌ No se pudo decodificar la imagen"
            )
        factor_escala = 0.6
        imagen = cv2.resize(
            imagen,
            None,
            fx=factor_escala,
            fy=factor_escala,
            interpolation=cv2.INTER_AREA
        )
        imagen, cuadro_blanco_box = detectar_cuadro_blanco(imagen)
        imagen, borde_recibo = remarcar_contorno_recibo(imagen)
        if not cuadro_blanco_box:
            raise HTTPException(
                status_code=422,
                detail="❌ No se detectó el cuadro blanco del recibo. "
                       "Verifica que la imagen sea clara y esté completa."
            )
        
        if not borde_recibo:
            raise HTTPException(
                status_code=422,
                detail="❌ No se detectó el contorno del recibo"
            )
        resultados_logos = []
        
        for nombre_logo, ruta_logo in logos_a_usar:
            logo = cv2.imread(ruta_logo)
            if logo is None:
                continue
            
            _, pos_logo, confianza = detectar_logo_multiescala(imagen, logo, umbral=0.60)
            
            if pos_logo:
                resultados_logos.append({
                    "tipo": nombre_logo,
                    "bounding_box": pos_logo,
                    "confianza": round(confianza * 100, 2)
                })
        
        if not resultados_logos:
            return JSONResponse(
                content={
                    "logo_detectado": False,
                    "es_valido": False,
                    "mensaje": "❌ No se detectó ningún logo de Plin/Interbank en la imagen",
                    "sugerencia": "Verifica que la imagen sea un recibo Plin válido y esté completa"
                },
                status_code=200
            )
        mejor_logo = max(resultados_logos, key=lambda x: x["confianza"])
        pos_logo = mejor_logo["bounding_box"]
        
        distancias_nueva, centro_logo, _ = calcular_distancias(
            pos_logo,
            borde_recibo,
            cuadro_blanco_box
        )
        
        comparaciones = []
        logo_obj = cv2.imread(logos_a_usar[0][1])
        
        for idx, plantilla_path in enumerate(plantillas, 1):
            distancias_plantilla = procesar_imagen_plantilla(
                plantilla_path,
                logo_obj,
                factor_escala
            )
            
            if distancias_plantilla:
                porcentaje, detalles = calcular_porcentaje_cambio(
                    distancias_nueva,
                    distancias_plantilla
                )
                
                if porcentaje != float('inf'):
                    comparaciones.append({
                        "plantilla": os.path.basename(plantilla_path),
                        "porcentaje_cambio": round(porcentaje, 2),
                        "detalles_comparacion": detalles
                    })
        
        if not comparaciones:
            raise HTTPException(
                status_code=500,
                detail="❌ No se pudo procesar ninguna plantilla correctamente. "
                       "Verifica que las plantillas sean válidas."
            )
        mejor_coincidencia = min(comparaciones, key=lambda x: x["porcentaje_cambio"])
        porcentaje_minimo = mejor_coincidencia["porcentaje_cambio"]
        advertencia = clasificar_autenticidad(porcentaje_minimo)
        es_valido = advertencia == "Auténtico"
        
        return JSONResponse(
            content={
                "logo_detectado": True,
                "es_valido": es_valido,
                "tipo_logo_detectado": mejor_logo["tipo"],
                "confianza_deteccion": mejor_logo["confianza"],
                "porcentaje_cambio_minimo": porcentaje_minimo,
                "advertencia": advertencia,
                "mejor_coincidencia": mejor_coincidencia,
                "todas_las_comparaciones": comparaciones[:3],
                "total_plantillas_analizadas": len(comparaciones),
                "mensaje": f"{'✅' if es_valido else '⚠️'} Recibo clasificado como: {advertencia}",
                "distancias_detectadas": distancias_nueva
            },
            status_code=200
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"❌ Error inesperado: {str(e)}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

@router.post("/filtro_logo/plin")
async def filtro_logo_plin(file: UploadFile = File(...)):
    return await filtro_logo(file, TipoLogo.PLIN)

@router.post("/filtro_logo/interbank")
async def filtro_logo_interbank(file: UploadFile = File(...)):
    return await filtro_logo(file, TipoLogo.INTERBANK)

@router.post("/filtro_logo/ambos")
async def filtro_logo_ambos(file: UploadFile = File(...)):
    return await filtro_logo(file, TipoLogo.AMBOS)