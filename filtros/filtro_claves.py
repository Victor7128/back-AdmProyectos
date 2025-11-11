import requests
import math
import tempfile
import os
from typing import Dict, List, Tuple
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

# ✅ PLANTILLA ACTUALIZADA - Basada en recibo Plin/Interbank real
PLANTILLA_PLIN_INTERBANK = [
    {"WordText": "Interbank", "Left": 338, "Top": 80, "peso": 1.5},  # Marca importante
    {"WordText": "exitoso", "Left": 404, "Top": 347, "peso": 1.5},   # Palabra clave
    {"WordText": "S/", "Left": 201, "Top": 428, "peso": 1.2},        # Símbolo de moneda
    {"WordText": "Enviado", "Left": 102, "Top": 615, "peso": 1.3},   # Acción
    {"WordText": "Destino", "Left": 102, "Top": 798, "peso": 1.3},   # Campo importante
    {"WordText": "Yape", "Left": 100, "Top": 852, "peso": 1.2},      # Destino común
    {"WordText": "Comision", "Left": 101, "Top": 928, "peso": 1.2},  # Sin tilde
    {"WordText": "GRATIS", "Left": 117, "Top": 991, "peso": 1.1},    # Comisión
    {"WordText": "Fecha", "Left": 102, "Top": 1078, "peso": 1.2},    # Campo temporal
    {"WordText": "hora", "Left": 248, "Top": 1078, "peso": 1.2},     # Campo temporal
    {"WordText": "Codigo", "Left": 101, "Top": 1207, "peso": 1.3},   # Sin tilde
    {"WordText": "operacion", "Left": 293, "Top": 1207, "peso": 1.3} # Sin tilde
]

# Plantilla alternativa para variaciones (envío vs recepción)
PLANTILLA_ALTERNATIVA = [
    {"WordText": "Interbank", "Left": 338, "Top": 80, "peso": 1.5},
    {"WordText": "Pago", "Left": 293, "Top": 348, "peso": 1.5},
    {"WordText": "exitoso", "Left": 404, "Top": 347, "peso": 1.5},
    {"WordText": "S/", "Left": 201, "Top": 428, "peso": 1.2},
    {"WordText": "Recibiste", "Left": 103, "Top": 426, "peso": 1.3},  # Alternativa
    {"WordText": "Destino", "Left": 102, "Top": 798, "peso": 1.3},
    {"WordText": "Fecha", "Left": 102, "Top": 1078, "peso": 1.2},
    {"WordText": "Codigo", "Left": 101, "Top": 1207, "peso": 1.3}
]

OCR_API_KEY = "e0b0a3ad7d88957"

def ocr_api(file_path: str) -> Dict:
    """Llama a la API de OCR.space para extraer texto de la imagen"""
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                "https://api.ocr.space/parse/image",
                data={
                    'apikey': OCR_API_KEY,
                    'language': 'spa',
                    'isOverlayRequired': 'true',
                    'scale': 'true',  # Mejora la detección
                    'OCREngine': '2'  # Motor más preciso
                },
                files={'file': f},
                timeout=30
            )
        return response.json() if response.status_code == 200 else {}
    except Exception as e:
        print(f"Error en OCR API: {e}")
        return {}

def normalizar_texto(texto: str) -> str:
    """Normaliza el texto eliminando tildes, espacios y convirtiendo a minúsculas"""
    if not texto:
        return ""
    texto = texto.lower().strip()
    
    # Mapeo más completo de caracteres especiales
    cambios = {
        'ö': 'o', 'ü': 'u', 'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ñ': 'n', 'ä': 'a', 'ë': 'e', 'ï': 'i', 'ô': 'o', 'û': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u'
    }
    
    for k, v in cambios.items():
        texto = texto.replace(k, v)
    
    # Eliminar signos de puntuación comunes
    texto = texto.replace(':', '').replace('¡', '').replace('!', '')
    
    return texto

def extraer_palabras(data_ocr: Dict) -> List[Dict]:
    """Extrae palabras con sus coordenadas del resultado del OCR"""
    palabras = []
    try:
        for resultado in data_ocr.get('ParsedResults', []):
            overlay = resultado.get('Overlay', {})
            if not overlay:
                # Fallback: intentar con TextOverlay
                overlay = resultado.get('TextOverlay', {})
            
            for linea in overlay.get('Lines', []):
                for palabra in linea.get('Words', []):
                    palabras.append({
                        'WordText': palabra.get('WordText', ''),
                        'Left': palabra.get('Left', 0),
                        'Top': palabra.get('Top', 0),
                        'Height': palabra.get('Height', 0),
                        'Width': palabra.get('Width', 0)
                    })
    except Exception as e:
        print(f"Error extrayendo palabras: {e}")
    
    return palabras

def calcular_similitud_texto(texto1: str, texto2: str) -> float:
    """Calcula similitud entre dos textos usando múltiples métricas"""
    t1 = normalizar_texto(texto1)
    t2 = normalizar_texto(texto2)
    
    if not t1 or not t2:
        return 0.0
    
    # Coincidencia exacta
    if t1 == t2:
        return 1.0
    
    # Contención (uno dentro del otro)
    if t1 in t2 or t2 in t1:
        return 0.85
    
    # Similitud por caracteres comunes
    comunes = set(t1) & set(t2)
    if not comunes:
        return 0.0
    
    # Jaccard similarity
    similitud_jaccard = len(comunes) / len(set(t1) | set(t2))
    
    # Similitud por longitud común
    len_comun = min(len(t1), len(t2))
    coincidencias = sum(1 for i in range(len_comun) if t1[i] == t2[i])
    similitud_posicion = coincidencias / max(len(t1), len(t2))
    
    # Promedio ponderado
    return (similitud_jaccard * 0.6) + (similitud_posicion * 0.4)

def calcular_distancia_normalizada(coord1: Tuple[int, int], coord2: Tuple[int, int], max_dist: int = 300) -> float:
    """
    Calcula la similitud basada en distancia entre coordenadas
    Retorna 1.0 si están muy cerca, 0.0 si están muy lejos
    """
    distancia = math.sqrt(
        (coord1[0] - coord2[0])**2 +
        (coord1[1] - coord2[1])**2
    )
    
    if distancia > max_dist:
        return 0.0
    
    return (max_dist - distancia) / max_dist

def calcular_similitud(plantilla: List[Dict], palabras_ocr: List[Dict]) -> Tuple[float, Dict]:
    """
    Calcula el porcentaje de similitud entre plantilla y palabras detectadas
    Retorna: (porcentaje, detalles_coincidencias)
    """
    if not plantilla or not palabras_ocr:
        return 0.0, {}

    coincidencias = 0.0
    peso_total = sum(item.get('peso', 1.0) for item in plantilla)
    usadas = set()
    detalles = {}

    for item_plantilla in plantilla:
        texto_plantilla = item_plantilla['WordText']
        pos_plantilla = (item_plantilla['Left'], item_plantilla['Top'])
        peso = item_plantilla.get('peso', 1.0)
        
        mejor_puntuacion = 0.0
        mejor_indice = -1
        mejor_palabra = ""

        for i, palabra_ocr in enumerate(palabras_ocr):
            if i in usadas:
                continue

            texto_ocr = palabra_ocr['WordText']
            pos_ocr = (palabra_ocr['Left'], palabra_ocr['Top'])

            # Calcular similitud de texto
            sim_texto = calcular_similitud_texto(texto_plantilla, texto_ocr)
            
            if sim_texto > 0.4:  # Umbral mínimo para considerar
                # Calcular similitud de posición
                sim_posicion = calcular_distancia_normalizada(pos_plantilla, pos_ocr, max_dist=250)
                
                # Puntuación combinada (texto más importante que posición)
                puntuacion = (sim_texto * 0.75) + (sim_posicion * 0.25)
                
                if puntuacion > mejor_puntuacion:
                    mejor_puntuacion = puntuacion
                    mejor_indice = i
                    mejor_palabra = texto_ocr

        # Considerar coincidencia si supera umbral
        if mejor_puntuacion > 0.55:
            coincidencias += peso * mejor_puntuacion
            usadas.add(mejor_indice)
            detalles[texto_plantilla] = {
                "encontrado": mejor_palabra,
                "confianza": round(mejor_puntuacion * 100, 2)
            }
        else:
            detalles[texto_plantilla] = {
                "encontrado": None,
                "confianza": 0
            }

    porcentaje = (coincidencias / peso_total) * 100
    return round(porcentaje, 2), detalles

@router.post("/ocr")
async def procesar_imagen(file: UploadFile = File(...)):
    """
    Endpoint para procesar imagen y verificar autenticidad mediante OCR
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, "❌ Debe ser una imagen válida (JPG, PNG, etc.)")

    temp_path = None
    try:
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(422, "❌ El archivo está vacío")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(content)
            temp_path = temp.name
        
        # Llamar al OCR
        data_ocr = ocr_api(temp_path)
        
        if not data_ocr or 'ParsedResults' not in data_ocr:
            raise HTTPException(500, "❌ Error al procesar OCR - No se pudo extraer texto")
        
        # Verificar si hay errores en el OCR
        if data_ocr.get('IsErroredOnProcessing'):
            error_msg = data_ocr.get('ErrorMessage', 'Error desconocido')
            raise HTTPException(500, f"❌ Error en OCR: {error_msg}")
        
        # Extraer palabras detectadas
        palabras = extraer_palabras(data_ocr)
        
        if not palabras:
            return {
                "porcentaje": 0.0,
                "advertencia": "Alterado",
                "es_valido": False,
                "mensaje": "⚠️ No se detectó texto en la imagen",
                "palabras_detectadas": 0
            }

        # Comparar con ambas plantillas
        porcentaje1, detalles1 = calcular_similitud(PLANTILLA_PLIN_INTERBANK, palabras)
        porcentaje2, detalles2 = calcular_similitud(PLANTILLA_ALTERNATIVA, palabras)
        
        # Usar la mejor coincidencia
        if porcentaje1 >= porcentaje2:
            porcentaje = porcentaje1
            detalles = detalles1
            plantilla_usada = "principal"
        else:
            porcentaje = porcentaje2
            detalles = detalles2
            plantilla_usada = "alternativa"

        # Clasificación de autenticidad (umbrales ajustados)
        if porcentaje >= 85:
            advertencia = "Auténtico"
            es_valido = True
        elif porcentaje >= 70:
            advertencia = "Sospechoso"
            es_valido = False
        else:
            advertencia = "Alterado"
            es_valido = False

        return {
            "porcentaje": porcentaje,
            "advertencia": advertencia,
            "es_valido": es_valido,
            "mensaje": f"{'✅' if es_valido else '⚠️'} Recibo clasificado como: {advertencia}",
            "plantilla_usada": plantilla_usada,
            "palabras_detectadas": len(palabras),
            "palabras_clave_encontradas": sum(1 for d in detalles.values() if d['encontrado']),
            "detalles_coincidencias": detalles
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error inesperado: {e}")
        raise HTTPException(500, f"❌ Error procesando imagen: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    """
    Endpoint de debug que muestra todas las palabras detectadas
    """
    temp_path = None
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(content)
            temp_path = temp.name
        
        data_ocr = ocr_api(temp_path)
        palabras = extraer_palabras(data_ocr)
        
        return {
            "total_palabras": len(palabras),
            "palabras_detectadas": palabras,
            "texto_completo": data_ocr.get('ParsedResults', [{}])[0].get('ParsedText', ''),
            "ocr_raw": data_ocr
        }
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)