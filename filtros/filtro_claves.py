import requests
import math
import tempfile
import os
from typing import Dict, List
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

PLANTILLA1 = [
    {"WordText": "Yapeaste!", "Left": 85, "Top": 313},
    {"WordText": "S/", "Left": 73, "Top": 380},
    {"WordText": "CÖDIGO", "Left": 73, "Top": 639},
    {"WordText": "SEGURIDAD", "Left": 199, "Top": 643},
    {"WordText": "DATOS", "Left": 73, "Top": 757},
    {"WordText": "TRANSACCIÖN", "Left": 222, "Top": 753},
    {"WordText": "celular", "Left": 184, "Top": 821},
    {"WordText": "Destino", "Left": 73, "Top": 876},
    {"WordText": "operaciön", "Left": 184, "Top": 930}
]

PLANTILLA2 = [
    {"WordText": "iTe", "Left": 103, "Top": 426},
    {"WordText": "Yapearon!", "Left": 199, "Top": 426},
    {"WordText": "Sl", "Left": 101, "Top": 542},
    {"WordText": "CÖDIGO", "Left": 100, "Top": 896},
    {"WordText": "SEGURIDAD", "Left": 305, "Top": 904},
    {"WordText": "DATOS", "Left": 101, "Top": 1061},
    {"WordText": "TRANSACCIÖN", "Left": 336, "Top": 1053},
    {"WordText": "celular", "Left": 276, "Top": 1155},
    {"WordText": "Destino", "Left": 102, "Top": 1244},
    {"WordText": "operaci6n", "Left": 276, "Top": 1332}
]

OCR_API_KEY = "e0b0a3ad7d88957"

def ocr_api(file_path: str) -> Dict:
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                "https://api.ocr.space/parse/image",
                data={
                    'apikey': OCR_API_KEY,
                    'language': 'spa',
                    'isOverlayRequired': 'true'
                },
                files={'file': f}
            )
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def normalizar_texto(texto: str) -> str:
    if not texto:
        return ""
    texto = texto.lower().strip()
    cambios = {'ö': 'o', 'ü': 'u', 'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u'}
    for k, v in cambios.items():
        texto = texto.replace(k, v)
    return texto

def extraer_palabras(data_ocr: Dict) -> List[Dict]:
    palabras = []
    try:
        for resultado in data_ocr.get('ParsedResults', []):
            for linea in resultado.get('TextOverlay', {}).get('Lines', []):
                for palabra in linea.get('Words', []):
                    palabras.append({
                        'WordText': palabra.get('WordText', ''),
                        'Left': palabra.get('Left', 0),
                        'Top': palabra.get('Top', 0)
                    })
    except:
        pass
    return palabras

def calcular_similitud(plantilla: List[Dict], palabras_ocr: List[Dict]) -> float:
    if not plantilla or not palabras_ocr:
        return 0.0

    coincidencias = 0
    usadas = set()

    for item_plantilla in plantilla:
        texto_plantilla = normalizar_texto(item_plantilla['WordText'])
        mejor_puntuacion = 0.0
        mejor_indice = -1

        for i, palabra_ocr in enumerate(palabras_ocr):
            if i in usadas:
                continue

            texto_ocr = normalizar_texto(palabra_ocr['WordText'])

            if texto_plantilla == texto_ocr:
                sim_texto = 1.0
            elif texto_plantilla in texto_ocr or texto_ocr in texto_plantilla:
                sim_texto = 0.8
            else:
                comunes = set(texto_plantilla) & set(texto_ocr)
                sim_texto = len(comunes) / max(len(texto_plantilla), len(texto_ocr)) if texto_plantilla and texto_ocr else 0.0

            if sim_texto > 0.3:
                distancia = math.sqrt(
                    (item_plantilla['Left'] - palabra_ocr['Left'])**2 +
                    (item_plantilla['Top'] - palabra_ocr['Top'])**2
                )
                sim_posicion = max(0, (200 - distancia) / 200) if distancia <= 200 else 0.0
                puntuacion = (sim_texto * 0.7) + (sim_posicion * 0.3)
                if puntuacion > mejor_puntuacion:
                    mejor_puntuacion = puntuacion
                    mejor_indice = i
        if mejor_puntuacion > 0.5:
            coincidencias += 1
            usadas.add(mejor_indice)
    return (coincidencias / len(plantilla)) * 100

@router.post("/ocr")
async def procesar_imagen(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, "Debe ser una imagen")

    temp_path = None
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(content)
            temp_path = temp.name
        data_ocr = ocr_api(temp_path)
        if not data_ocr:
            raise HTTPException(500, "Error en OCR")
        palabras = extraer_palabras(data_ocr)
        if not palabras:
            return {"porcentaje": 0}

        porcentaje1 = calcular_similitud(PLANTILLA1, palabras)
        porcentaje2 = calcular_similitud(PLANTILLA2, palabras)
        porcentaje = round(max(porcentaje1, porcentaje2), 2)

        advertencia = ""
        if porcentaje < 90:
            advertencia = "Alterado"
        elif porcentaje < 98:
            advertencia = "Sospechoso"
        else:
            advertencia = "Auténtico"

        return {"porcentaje": porcentaje, "advertencia": advertencia}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)