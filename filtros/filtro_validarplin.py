# filtros/filtro_ocr.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import requests
import cv2
import numpy as np
import re
from datetime import datetime



router = APIRouter()

API_KEY = 'e0b0a3ad7d88957'
OCR_API_URL = 'https://api.ocr.space/parse/image'

destinos_validos = ["Plin", "BCP", "Interbank", "Scotiabank", "BBVA", "Yape"]
comisiones_permitidas_literal = ["GRATIS", "GRATIS.", "GRATIS "]
palabras_negra = ["PELIGRO", "BLOQUEADA", "ESTAFA", "FRAUDE", "ANULADO", "REEMBOLSO", "ERROR"]

nombre_regex_flexible = re.compile(
    r"^([A-ZÁÉÍÓÚÑ ]+|([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)( [A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)$"
)
telefono_regex_exact = re.compile(r"^\d{3} \d{3} \d{3}$")
monto_regex = re.compile(r"^[Ss]/\s*([\d]{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s*$")
fecha_regex = re.compile(r"^\d{2}\s[A-Za-z]{3}\s\d{4}\s\d{2}:\d{2}\s(?:AM|PM)$")

# --- Funciones auxiliares ---
def enviar_a_ocr_bytes(imagen_bytes):
    files = {'filename': ('imagen.jpg', imagen_bytes, 'image/jpeg')}
    payload = {
        'apikey': API_KEY,
        'language': 'spa',
        'isOverlayRequired': True,
        'OCREngine': 2
    }
    try:
        respuesta = requests.post(OCR_API_URL, files=files, data=payload, timeout=30)
        return respuesta.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enviando imagen al OCR: {e}")

def fecha_es_valida_y_no_futura(fecha_str):
    try:
        dt = datetime.strptime(fecha_str, "%d %b %Y %I:%M %p")
    except Exception:
        return False, "Formato inválido"
    if dt > datetime.now():
        return False, "Fecha en el futuro"
    return True, ""

def normalizar_decimal_monto(monto_str):
    s = monto_str.strip()
    if '.' in s and ',' in s:
        s = s.replace('.', '').replace(',', '.')
    else:
        s = s.replace(',', '.')
    try:
        return float(s)
    except:
        return None

def analizar_overlay(parsed_results):
    text_overlay = parsed_results.get('TextOverlay', {}) or {}
    lines_overlay = text_overlay.get('Lines', None)
    centers = []
    if lines_overlay:
        for linea in lines_overlay:
            tops, heights = [], []
            for w in linea.get('Words', []):
                try:
                    t = float(w.get('Top', 0))
                    h = float(w.get('Height', 0))
                except:
                    t = h = 0.0
                tops.append(t)
                heights.append(h)
            if tops:
                top_min = min(tops)
                avg_h = sum(heights)/len(heights) if heights else 0
                centers.append(top_min + avg_h/2.0)
    return centers

def analizar_distancias_verticales(centers):
    if len(centers) < 2:
        return True, ""
    diffs = [abs(centers[i+1] - centers[i]) for i in range(len(centers)-1)]
    diffs_sorted = sorted(diffs)
    mid = len(diffs_sorted)//2
    median = diffs_sorted[mid] if len(diffs_sorted) % 2 else (diffs_sorted[mid-1]+diffs_sorted[mid])/2
    outliers = [d for d in diffs if d > median * 3]
    if outliers:
        return False, f"Se detectaron saltos verticales anómalos: {outliers}, mediana={median:.1f}"
    return True, ""

# --- Función principal de validación ---
def validar_comprobante(parsed_results, imagen):
    resultado = {"valido": False, "errores": [], "advertencias": [], "campos": {}}
    texto_detectado = parsed_results.get('ParsedText', '')
    if isinstance(texto_detectado, bytes):
        texto_detectado = texto_detectado.decode('utf-8', errors='ignore')
    texto_detectado = str(texto_detectado)
    resultado['campos']['texto_detectado'] = texto_detectado
    lineas_texto = [l.strip() for l in texto_detectado.split('\n') if l.strip()]
    resultado['campos']['lineas'] = lineas_texto

    if len(lineas_texto) < 15:
        resultado['errores'].append(f"Se esperaban al menos 15 líneas; encontradas: {len(lineas_texto)}")

    def linea(n):
        return lineas_texto[n-1] if len(lineas_texto) >= n else None

    # Validaciones línea por línea
    l3 = linea(3)
    resultado['campos']['linea_3'] = l3
    if not (l3 and "pago" in l3.lower()):
        resultado['errores'].append(f"Línea 3 incorrecta: '{l3}'")

    l4 = linea(4) or ""
    resultado['campos']['linea_4'] = l4
    m_match = re.search(r"[Ss]/\s*([0-9\.,]+)", l4)
    if not m_match:
        resultado['errores'].append(f"No se detectó monto en línea 4: '{l4}'")
    else:
        monto_val = normalizar_decimal_monto(m_match.group(1))
        if monto_val is None:
            resultado['errores'].append(f"No se pudo parsear monto: '{m_match.group(1)}'")
        else:
            partes = m_match.group(1).replace('.', '').replace(',', '.').split('.')
            decimales = len(partes[1]) if len(partes) > 1 else 0
            if decimales > 2:
                resultado['errores'].append(f"Monto con más de 2 decimales: '{m_match.group(1)}'")
            if not (0 <= monto_val <= 500):
                resultado['errores'].append(f"Monto fuera de rango (0-500): S/{monto_val:.2f}")
            resultado['campos']['monto'] = monto_val

    l5 = linea(5)
    resultado['campos']['linea_5'] = l5
    if l5 != "Enviado a:":
        resultado['errores'].append(f"Línea 5 incorrecta: '{l5}'")

    l6 = linea(6) or ""
    resultado['campos']['linea_6'] = l6
    l6_norm = re.sub(r"\s+", " ", l6).strip()
    if not nombre_regex_flexible.match(l6_norm):
        resultado['errores'].append(f"Nombre inválido: '{l6}'")
    else:
        resultado['campos']['nombre'] = l6_norm

    l7 = linea(7) or ""
    resultado['campos']['linea_7'] = l7
    if not telefono_regex_exact.match(l7):
        resultado['errores'].append(f"Teléfono inválido: '{l7}'")
    else:
        resultado['campos']['telefono'] = l7

    l8 = linea(8)
    resultado['campos']['linea_8'] = l8
    if l8 != "Destino:":
        resultado['errores'].append(f"Línea 8 incorrecta: '{l8}'")

    l9 = linea(9) or ""
    resultado['campos']['linea_9'] = l9
    if l9 not in destinos_validos:
        resultado['errores'].append(f"Destino no válido: '{l9}'")

    l10 = linea(10)
    resultado['campos']['linea_10'] = l10
    if l10 != "Comisión:":
        resultado['errores'].append(f"Línea 10 incorrecta: '{l10}'")

    l11 = linea(11) or ""
    resultado['campos']['linea_11'] = l11
    l11_up = l11.strip().upper()
    if l11_up in comisiones_permitidas_literal:
        resultado['campos']['comision'] = l11_up
    else:
        cm = re.search(r"[Ss]/\s*([\d\.,]+)", l11)
        if cm:
            cm_val = normalizar_decimal_monto(cm.group(1))
            if cm_val is None:
                resultado['errores'].append(f"Comisión inválida: '{l11}'")
            else:
                resultado['campos']['comision'] = cm_val
        else:
            resultado['advertencias'].append(f"Tipo de comisión atípico: '{l11}'")

    l12 = linea(12)
    resultado['campos']['linea_12'] = l12
    if l12 != "Fecha y hora:":
        resultado['errores'].append(f"Línea 12 incorrecta: '{l12}'")

    l13 = linea(13) or ""
    resultado['campos']['linea_13'] = l13
    if not fecha_regex.match(l13):
        resultado['errores'].append(f"Formato de fecha inválido: '{l13}'")
    else:
        ok, motivo = fecha_es_valida_y_no_futura(l13)
        if not ok:
            resultado['errores'].append(f"Fecha inválida: {motivo} ('{l13}')")

    l14 = linea(14)
    resultado['campos']['linea_14'] = l14
    if l14 != "Código de operación:":
        resultado['errores'].append(f"Línea 14 incorrecta: '{l14}'")

    l15 = linea(15) or ""
    resultado['campos']['linea_15'] = l15
    corrected = l15.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
    if not re.fullmatch(r"\d{8}", corrected):
        resultado['errores'].append(f"Código inválido: '{l15}' (corregido: '{corrected}')")
    else:
        resultado['campos']['codigo_operacion'] = corrected

    # Palabras negras
    texto_upper = texto_detectado.upper()
    for p in palabras_negra:
        if p in texto_upper:
            resultado['errores'].append(f"Palabra sospechosa detectada: '{p}'")

    # Overlay
    centers = analizar_overlay(parsed_results)
    resultado['campos']['overlay_centers_count'] = len(centers)
    resultado['campos']['lineas_count'] = len(lineas_texto)
    if centers:
        if abs(len(centers) - len(lineas_texto)) > 3:
            resultado['advertencias'].append(
                f"Diferencia importante entre overlay ({len(centers)}) y líneas ({len(lineas_texto)})"
            )
        ok_dist, msg_dist = analizar_distancias_verticales(centers)
        if not ok_dist:
            resultado['advertencias'].append(msg_dist)

    resultado['valido'] = (len(resultado['errores']) == 0)
    return resultado

# --- Endpoint POST ---
@router.post("/validarplin")
async def filtro_ocr(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="El archivo debe ser una imagen")
    imagen_bytes = await file.read()
    if not imagen_bytes:
        raise HTTPException(status_code=422, detail="Archivo vacío")

    ocr_result = enviar_a_ocr_bytes(imagen_bytes)
    if not ocr_result or ocr_result.get('IsErroredOnProcessing'):
        raise HTTPException(status_code=422, detail=ocr_result.get('ErrorMessage', 'OCR falló'))

    parsed_results = ocr_result.get('ParsedResults')
    if not parsed_results:
        raise HTTPException(status_code=422, detail="No se encontraron resultados OCR")
    first = parsed_results[0]

    # Decodificar la imagen para OpenCV (opcional)
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    resultado = validar_comprobante(first, imagen)
    return JSONResponse(content=resultado)
