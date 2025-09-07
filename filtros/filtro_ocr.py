import cv2
import numpy as np
import re
import requests
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

API_KEY = 'e0b0a3ad7d88957'
OCR_API_URL = 'https://api.ocr.space/parse/image'

DESTINOS_VALIDOS = [
    'Yape', 'Plin', 'BCP', 'Interbank', 'BBVA', 'Scotiabank',
    'Caja Arequipa', 'Caja Huancayo', 'Caja Piura', 'Caja Cusco',
    'Caja Trujillo', 'MiBanco', 'Banco de la Nación', 'Caja Sullana',
    'Caja Tacna', 'Caja Metropolitana'
]

def enviar_imagen_ocr_bytes(imagen_bytes):
    try:
        response = requests.post(
            OCR_API_URL,
            files={'file': imagen_bytes},
            data={
                'apikey': API_KEY,
                'language': 'spa',
                'OCREngine': '2',
                'isOverlayRequired': True
            },
            timeout=30
        )
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="El OCR externo tardó demasiado (timeout).")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error llamando al OCR externo: {str(e)}")
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OCR externo respondió mal: {response.status_code}")
    resultado = response.json()
    if resultado.get("IsErroredOnProcessing"):
        return None, []
    parsed_result = resultado['ParsedResults'][0]
    return parsed_result['ParsedText'].strip(), parsed_result.get('TextOverlay', {}).get('Lines', [])

def recortar_cuadro_blanco_np(image_np):
    gris = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 240, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    contorno_mayor = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_mayor)
    recorte = image_np[y:y + h, x:x + w]
    return recorte

def extraer_codigo_destino(lineas):
    seguridad_y = None
    destino_y = None
    destino_x = None
    posibles_codigos_seguridad = []
    for linea in lineas:
        for palabra in linea['Words']:
            try:
                left = int(palabra['Left'])
                top = int(palabra['Top'])
                width = int(palabra['Width'])
                height = int(palabra['Height'])
                texto = palabra['WordText']
            except (KeyError, ValueError, TypeError):
                continue
            if texto.upper() == 'SEGURIDAD':
                seguridad_y = top + height // 2
            elif texto.upper() == 'DESTINO':
                destino_y = top + height // 2
                destino_x = left + width
    if seguridad_y:
        margen = 10
        for linea in lineas:
            for palabra in linea['Words']:
                try:
                    left = int(palabra['Left'])
                    top = int(palabra['Top'])
                    width = int(palabra['Width'])
                    height = int(palabra['Height'])
                    texto = palabra['WordText']
                except:
                    continue
                centro_y = top + height // 2
                if abs(centro_y - seguridad_y) <= margen and re.fullmatch(r'\d', texto):
                    posibles_codigos_seguridad.append(texto)
    codigo_concatenado = ''.join(posibles_codigos_seguridad)
    destino_texto = ""
    if destino_y and destino_x:
        margen_vertical = 10
        destino_colectado = []
        for linea in lineas:
            for palabra in linea['Words']:
                try:
                    left = int(palabra['Left'])
                    top = int(palabra['Top'])
                    height = int(palabra['Height'])
                    texto = palabra['WordText'].strip()
                    centro_y = top + height // 2
                    if abs(centro_y - destino_y) <= margen_vertical and left > destino_x:
                        destino_colectado.append(texto)
                except:
                    continue
        if destino_colectado:
            destino_texto = " ".join(destino_colectado)
    return [codigo_concatenado] if codigo_concatenado else [], destino_texto

def detectar_estructura(texto):
    if '¡Yapeaste!' in texto:
        return 1
    elif '¡Te Yapearon!' in texto:
        return 2
    return 0

def validar_estructura_1(texto, codigo_detectado=None, destino_detectado=None):
    resultado = {
        "estructura": "¡Yapeaste!",
        "monto": None,
        "receptor": None,
        "fecha": None,
        "hora": None,
        "codigo_seguridad": None,
        "destino": None,
        "numero_enmascarado": None,
        "codigo_operacion": None,
        "comentario": None
    }
    advertencias = []
    # 1. Buscar monto con símbolo
    match_monto = re.search(r'S/\s?(\d+(?:\.\d{2})?)', texto)
    monto_simbolo_encontrado = False
    if match_monto:
        monto_simbolo_encontrado = True
        try:
            monto = float(match_monto.group(1))
            resultado["monto"] = monto
            if monto <= 0 or monto > 500:
                advertencias.append("Monto incorrecto, debe estar entre 0 - 500 soles.")
        except ValueError:
            advertencias.append("Error al convertir el monto a número.")
    else:
        # 2. Buscar cualquier número decimal razonable
        match_monto_alt = re.search(r'\b(\d{1,4}(?:\.\d{2})?)\b', texto)
        if match_monto_alt:
            try:
                monto = float(match_monto_alt.group(1))
                resultado["monto"] = monto
                if monto <= 0 or monto > 500:
                    advertencias.append("Monto incorrecto, debe estar entre 0 - 500 soles.")
                advertencias.append("Monto encontrado pero símbolo de moneda no detectado (S/.)")
            except ValueError:
                advertencias.append("Número detectado como posible monto pero no se pudo convertir")
        else:
            advertencias.append("Monto no detectado")
    match_nombre = re.search(r'S/\s?\d+(?:\.\d{2})?\s*\n?([A-ZÁÉÍÓÚÑa-záéíóúñ ]+)', texto)
    if match_nombre:
        resultado["receptor"] = match_nombre.group(1).strip()
    match_fecha = re.search(r'(\d{1,2})\s+([a-zA-ZáéíóúñÑ]{3,9})\.?\s+(\d{4}).*?(\d{1,2}:\d{2}.*?m)', texto)
    if match_fecha:
        dia, mes_str, anio, hora = match_fecha.groups()
        resultado["fecha"] = f"{dia} {mes_str} {anio}"
        resultado["hora"] = hora
        meses = {
            'ene': 1, 'enero': 1,
            'feb': 2, 'febrero': 2,
            'mar': 3, 'marzo': 3,
            'abr': 4, 'abril': 4,
            'may': 5, 'mayo': 5,
            'jun': 6, 'junio': 6,
            'jul': 7, 'julio': 7,
            'ago': 8, 'agosto': 8,
            'sep': 9, 'sept': 9, 'septiembre': 9,
            'oct': 10, 'octubre': 10,
            'nov': 11, 'noviembre': 11,
            'dic': 12, 'diciembre': 12
        }
        mes_limpio = mes_str.lower().strip('.')
        mes_num = meses.get(mes_limpio, 0)
        if mes_num:
            try:
                fecha_obj = datetime(int(anio), mes_num, int(dia))
                hoy = datetime.now()
                if fecha_obj > hoy:
                    advertencias.append("Fecha inválida")
            except ValueError:
                advertencias.append("Fecha inválida (no existe en el calendario).")
        else:
            advertencias.append("Mes no reconocido")
        lineas = texto.splitlines()
        campos_fijos = ['Nro. de operación', 'Destino', 'DATOS DE LA TRANSACCIÓN', 'Yape']
        for i, linea in enumerate(lineas):
            if all(p in linea for p in [dia, mes_str, anio]):
                if i + 1 < len(lineas):
                    posible_comentario = lineas[i + 1].strip()
                    if posible_comentario and not any(campo.lower() in posible_comentario.lower() for campo in campos_fijos):
                        if len(posible_comentario.split()) >= 2:
                            resultado["comentario"] = posible_comentario
                break
    if codigo_detectado:
        resultado["codigo_seguridad"] = codigo_detectado
    if destino_detectado:
        resultado["destino"] = destino_detectado
    match_enmascarado = re.search(r'(\*{4,6}\s?\d{3})', texto)
    if match_enmascarado:
        resultado["numero_enmascarado"] = match_enmascarado.group(1)
    match_codigo_8 = re.search(r'\b\d{8}\b', texto)
    if match_codigo_8:
        resultado["codigo_operacion"] = match_codigo_8.group()
    validar_campos_obligatorios(resultado, advertencias, skip_monto=True)
    if resultado["monto"] is None:
        # Si después de todo no se encontró monto
        if not any("símbolo" in adv for adv in advertencias):
            advertencias.append("Monto no detectado")
    if advertencias:
        resultado["advertencias"] = advertencias
    return resultado

def validar_estructura_2(texto, codigo_detectado=None, destino_detectado=None):
    resultado = {
        "estructura": "¡Te Yapearon!",
        "monto": None,
        "receptor": None,
        "fecha": None,
        "hora": None,
        "codigo_seguridad": None,
        "destino": None,
        "codigo_operacion": None,
        "comentario": None
    }
    advertencias = []
    match_monto = re.search(r'S/\s?(\d+(?:\.\d{2})?)', texto)
    monto_simbolo_encontrado = False
    if match_monto:
        monto_simbolo_encontrado = True
        try:
            monto = float(match_monto.group(1))
            resultado["monto"] = monto
            if monto <= 0 or monto > 500:
                advertencias.append("Monto incorrecto")
        except ValueError:
            advertencias.append("Error al convertir el monto a número.")
    else:
        match_monto_alt = re.search(r'\b(\d{1,4}(?:\.\d{2})?)\b', texto)
        if match_monto_alt:
            try:
                monto = float(match_monto_alt.group(1))
                resultado["monto"] = monto
                if monto <= 0 or monto > 500:
                    advertencias.append("Monto incorrecto, debe estar entre 0 - 500 soles.")
                advertencias.append("Monto encontrado pero símbolo de moneda no detectado(S/.)")
            except ValueError:
                advertencias.append("Número detectado como posible monto pero no se pudo convertir")
        else:
            advertencias.append("Monto no detectado")
    nombre = None
    lineas = texto.splitlines()
    linea_monto_idx = -1
    for i, linea in enumerate(lineas):
        if re.search(r'S/\s?(\d+(?:\.\d{2})?)', linea):
            linea_monto_idx = i
            break
    if linea_monto_idx == -1:
        # Si no se encontró con símbolo, buscar con el alternativo
        for i, linea in enumerate(lineas):
            if re.search(r'\b(\d{1,4}(?:\.\d{2})?)\b', linea):
                linea_monto_idx = i
                break
    if linea_monto_idx != -1 and linea_monto_idx + 1 < len(lineas):
        posible_nombre = lineas[linea_monto_idx + 1].strip()
        patron_nombre = r'^([A-ZÁÉÍÓÚÑa-záéíóúñ]+|[A-ZÁÉÍÓÚÑ]\.?)(\s+([A-ZÁÉÍÓÚÑa-záéíóúñ]+|[A-ZÁÉÍÓÚÑ]\.?)){1,4}$'
        if re.fullmatch(patron_nombre, posible_nombre):
            nombre = posible_nombre
            resultado["receptor"] = nombre
    match_fecha = re.search(r'(\d{1,2})\s+([a-zA-ZáéíóúñÑ]{3,9})\.?\s+(\d{4}).*?(\d{1,2}:\d{2}.*?m)', texto)
    if match_fecha:
        dia, mes_str, anio, hora = match_fecha.groups()
        resultado["fecha"] = f"{dia} {mes_str} {anio}"
        resultado["hora"] = hora
        meses = {
            'ene': 1, 'enero': 1,
            'feb': 2, 'febrero': 2,
            'mar': 3, 'marzo': 3,
            'abr': 4, 'abril': 4,
            'may': 5, 'mayo': 5,
            'jun': 6, 'junio': 6,
            'jul': 7, 'julio': 7,
            'ago': 8, 'agosto': 8,
            'sep': 9, 'sept': 9, 'septiembre': 9,
            'oct': 10, 'octubre': 10,
            'nov': 11, 'noviembre': 11,
            'dic': 12, 'diciembre': 12
        }
        mes_limpio = mes_str.lower().strip('.')
        mes_num = meses.get(mes_limpio, 0)
        if mes_num:
            try:
                fecha_obj = datetime(int(anio), mes_num, int(dia))
                hoy = datetime.now()
                if fecha_obj > hoy:
                    advertencias.append("Fecha inválida")
            except ValueError:
                advertencias.append("Fecha inválida (no existe en el calendario).")
        else:
            advertencias.append("Mes no reconocido")
        campos_fijos = ['Nro. de operación', 'Destino', 'DATOS DE LA TRANSACCIÓN', 'Yape']
        for i, linea in enumerate(lineas):
            if all(p in linea for p in [dia, mes_str, anio]):
                if i + 1 < len(lineas):
                    posible_comentario = lineas[i + 1].strip()
                    if posible_comentario and not any(campo.lower() in posible_comentario.lower() for campo in campos_fijos):
                        if len(posible_comentario.split()) >= 2:
                            resultado["comentario"] = posible_comentario
                break
    if codigo_detectado:
        resultado["codigo_seguridad"] = codigo_detectado
    if destino_detectado:
        resultado["destino"] = destino_detectado
    match_codigo_8 = re.search(r'\b\d{8}\b', texto)
    if match_codigo_8:
        resultado["codigo_operacion"] = match_codigo_8.group()
    validar_campos_obligatorios(resultado, advertencias, skip_monto=True)
    if resultado["monto"] is None:
        if not any("símbolo" in adv for adv in advertencias):
            advertencias.append("Monto no detectado")
    if advertencias:
        resultado["advertencias"] = advertencias
    return resultado

def validar_campos_obligatorios(resultado, advertencias, skip_monto=False):
    if not resultado.get("codigo_operacion"):
        advertencias.append("Nro. de Operación no detectada")
    if not resultado.get("fecha"):
        advertencias.append("Fecha no detectada")
    if not resultado.get("hora"):
        advertencias.append("Hora no detectada")
    if not resultado.get("receptor"):
        advertencias.append("Receptor no detectado")
    if not skip_monto and resultado.get("monto") is None:
        advertencias.append("Monto no detectado")
    if not resultado.get("destino"):
        advertencias.append("Destino no detectado")
        
@router.post("/filtro_ocr")
async def filtro_ocr(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=422, detail="El archivo debe ser una imagen")
    content = await file.read()
    np_arr = np.frombuffer(content, np.uint8)
    imagen = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if imagen is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")
    if imagen.shape[0] > 2000 or imagen.shape[1] > 2000:
        imagen = cv2.resize(imagen, (0,0), fx=0.5, fy=0.5)
    recorte = recortar_cuadro_blanco_np(imagen)
    if recorte is None:
        raise HTTPException(status_code=400, detail="No se pudo recortar la imagen")
    _, buffer = cv2.imencode(".png", recorte)
    img_bytes = buffer.tobytes()
    texto, lineas_overlay = enviar_imagen_ocr_bytes(('recorte.png', img_bytes, 'image/png'))
    if not texto:
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del comprobante")
    tipo = detectar_estructura(texto)
    codigos_detectados, destino_detectado = extraer_codigo_destino(lineas_overlay)
    codigo_valido = codigos_detectados[0] if codigos_detectados else None
    if tipo == 1:
        resultado = validar_estructura_1(texto, codigo_valido, destino_detectado)
    elif tipo == 2:
        resultado = validar_estructura_2(texto, codigo_valido, destino_detectado)
    else:
        raise HTTPException(status_code=400, detail="Estructura desconocida en el comprobante")
    return resultado