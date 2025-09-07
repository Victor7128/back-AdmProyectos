import cv2
import numpy as np
import os
import tempfile
from typing import Dict, List
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()
plantilla1 = "./filtros/plantillas/yape.jpg"
plantilla2 = "./filtros/plantillas/yape2.jpg"
logo_path = "./filtros/Logo.jpg"

def detectar_cuadro_blanco(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, umbral = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mayor_area = 0
    cuadro_blanco_box = None

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 1000 and area > mayor_area:
            mayor_area = area
            x, y, w, h = cv2.boundingRect(contorno)
            cuadro_blanco_box = (x, y, w, h)

    return imagen, cuadro_blanco_box

def detectar_logo_multiescala(imagen, logo, umbral=0.7):
    img_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

    mejor_confianza = 0
    mejor_top_left = None
    mejor_bottom_right = None

    for escala in np.linspace(0.4, 1.5, 20)[::-1]:
        ancho_nuevo = int(logo_gray.shape[1] * escala)
        alto_nuevo = int(logo_gray.shape[0] * escala)
        if ancho_nuevo < 10 or alto_nuevo < 10:
            continue

        logo_redimensionado = cv2.resize(logo_gray, (ancho_nuevo, alto_nuevo), interpolation=cv2.INTER_AREA)

        if img_gray.shape[0] < logo_redimensionado.shape[0] or img_gray.shape[1] < logo_redimensionado.shape[1]:
            continue

        resultado = cv2.matchTemplate(img_gray, logo_redimensionado, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(resultado)

        if max_val > mejor_confianza and max_val >= umbral:
            mejor_confianza = max_val
            mejor_top_left = max_loc
            mejor_bottom_right = (max_loc[0] + ancho_nuevo, max_loc[1] + alto_nuevo)

    if mejor_top_left and mejor_bottom_right:
        x, y = mejor_top_left
        w = mejor_bottom_right[0] - x
        h = mejor_bottom_right[1] - y
        return imagen, (x, y, w, h)
    else:
        return imagen, None

def remarcar_contorno_recibo(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, umbral = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        return imagen, None

    contorno_mayor = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_mayor)
    return imagen, (x, y, w, h)

def calcular_distancias(logo_box, borde_box, cuadro_blanco_box):
    lx, ly, lw, lh = logo_box
    bx, by, bw, bh = borde_box
    _, y_cuadro_blanco, _, _ = cuadro_blanco_box

    centro_logo = (lx + lw // 2, ly + lh // 2)

    puntos_borde = {
        "Izquierda": (bx, centro_logo[1]),
        "Derecha": (bx + bw, centro_logo[1]),
        "Arriba": (centro_logo[0], by),
        "Abajo (cuadro blanco)": (centro_logo[0], y_cuadro_blanco),
    }

    distancias = {
        lado: int(np.linalg.norm(np.array(centro_logo) - np.array(punto)))
        for lado, punto in puntos_borde.items()
    }

    return distancias, centro_logo, puntos_borde

def procesar_imagen_plantilla(ruta_plantilla, logo):
    """Procesa una imagen plantilla y extrae las distancias del logo"""
    if not os.path.exists(ruta_plantilla):
        return None
    
    imagen = cv2.imread(ruta_plantilla)
    if imagen is None:
        return None
    
    # Procesar la plantilla de la misma forma que la imagen de entrada
    imagen = cv2.resize(imagen, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    imagen, cuadro_blanco_box = detectar_cuadro_blanco(imagen)
    imagen, pos_logo = detectar_logo_multiescala(imagen, logo)
    imagen, borde_recibo = remarcar_contorno_recibo(imagen)
    
    if pos_logo and borde_recibo and cuadro_blanco_box:
        distancias, _, _ = calcular_distancias(pos_logo, borde_recibo, cuadro_blanco_box)
        return distancias
    
    return None

def calcular_porcentaje_cambio(distancias_nueva, distancias_plantilla):
    """Calcula el porcentaje promedio de cambio entre dos conjuntos de distancias"""
    if not distancias_nueva or not distancias_plantilla:
        return float('inf')
    
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
                    "nuevo": val_nueva,
                    "plantilla": val_plantilla,
                    "cambio_px": diferencia,
                    "cambio_porcentaje": round(porcentaje, 2)
                }
    
    if cambios:
        porcentaje_promedio = sum(cambios) / len(cambios)
        return porcentaje_promedio, comparacion
    
    return float('inf'), {}

@router.post("/logo")
async def procesar_imagen_logo(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, "Debe ser una imagen")
    
    temp_path = None
    try:
        archivos_faltantes = []
        if not os.path.exists(logo_path):
            archivos_faltantes.append(f"logo: {logo_path}")
        if not os.path.exists(plantilla1):
            archivos_faltantes.append(f"plantilla1: {plantilla1}")
        if not os.path.exists(plantilla2):
            archivos_faltantes.append(f"plantilla2: {plantilla2}")
        
        if archivos_faltantes:
            raise HTTPException(400, f"Archivos no encontrados: {', '.join(archivos_faltantes)}")
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(content)
            temp_path = temp.name
        imagen = cv2.imread(temp_path)
        if imagen is None:
            raise HTTPException(400, "No se pudo leer la imagen subida")

        logo = cv2.imread(logo_path)
        if logo is None:
            raise HTTPException(400, "No se pudo leer el archivo de logo")
        imagen = cv2.resize(imagen, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        imagen, cuadro_blanco_box = detectar_cuadro_blanco(imagen)
        imagen, pos_logo = detectar_logo_multiescala(imagen, logo)
        imagen, borde_recibo = remarcar_contorno_recibo(imagen)

        if not pos_logo:
            return JSONResponse(content={
                "logo_detectado": False,
                "mensaje": "No se detectó el logo en la imagen"
            }, status_code=200)

        if not (borde_recibo and cuadro_blanco_box):
            elementos_faltantes = []
            if not borde_recibo:
                elementos_faltantes.append("contorno del recibo")
            if not cuadro_blanco_box:
                elementos_faltantes.append("cuadro blanco")
            
            return JSONResponse(content={
                "logo_detectado": True,
                "error": f"No se detectaron elementos necesarios para el cálculo: {', '.join(elementos_faltantes)}"
            }, status_code=400)

        distancias_nueva, _, _ = calcular_distancias(pos_logo, borde_recibo, cuadro_blanco_box)
        
        distancias_plantilla1 = procesar_imagen_plantilla(plantilla1, logo)
        distancias_plantilla2 = procesar_imagen_plantilla(plantilla2, logo)
        
        resultados = []
        

        if distancias_plantilla1:
            porcentaje1 = float('inf')  
            resultado_porcentaje1 = calcular_porcentaje_cambio(distancias_nueva, distancias_plantilla1)
            comparacion1 = {}
            if isinstance(resultado_porcentaje1, tuple):
                porcentaje1, comparacion1 = resultado_porcentaje1
            if porcentaje1 != float('inf'):
                resultados.append({
                    "plantilla": "plantilla1",
                    "porcentaje_cambio": round(porcentaje1, 2),
                    "detalles": comparacion1
                })
    
        if distancias_plantilla2:
            resultado_porcentaje2 = calcular_porcentaje_cambio(distancias_nueva, distancias_plantilla2)
            porcentaje2, comparacion2 = (resultado_porcentaje2 if isinstance(resultado_porcentaje2, tuple) else (float('inf'), {}))
            if porcentaje2 != float('inf'):
                resultados.append({
                    "plantilla": "plantilla2",
                    "porcentaje_cambio": round(porcentaje2, 2),
                    "detalles": comparacion2
                })
        
        if not resultados:
            return JSONResponse(content={
                "logo_detectado": True,
                "error": "No se pudo procesar ninguna de las plantillas o no se detectó el logo en las plantillas"
            }, status_code=400)
        
        mejor_resultado = min(resultados, key=lambda x: x["porcentaje_cambio"])
        advertencia = ""
        if mejor_resultado["porcentaje_cambio"] <= 95:
            advertencia = "Sospechoso"
        elif mejor_resultado["porcentaje_cambio"] < 90:
            advertencia = "Alterado"
        else:
            advertencia = "Auténtico"

        return JSONResponse(content={
            "logo_detectado": True,
            "mejor_coincidencia": mejor_resultado,
            "todas_las_comparaciones": resultados,
            "porcentaje_cambio_minimo": mejor_resultado["porcentaje_cambio"],
            "advertencia": advertencia
        }, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error procesando la imagen: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)