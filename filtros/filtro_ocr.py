# Filtro OCR optimizado para estructura REAL de comprobantes Plin
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
    'Caja Tacna', 'Caja Metropolitana', 'Banco Pichincha'
]

def enviar_imagen_ocr_bytes(imagen_bytes):
    """Envía la imagen al servicio OCR externo"""
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
        raise HTTPException(
            status_code=504, 
            detail="❌ El OCR externo tardó demasiado (timeout)."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"❌ Error llamando al OCR externo: {str(e)}"
        )
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=502, 
            detail=f"❌ OCR externo respondió mal: {response.status_code}"
        )
    
    resultado = response.json()
    
    if resultado.get("IsErroredOnProcessing"):
        return None, []
    
    if not resultado.get('ParsedResults'):
        return None, []
    
    parsed_result = resultado['ParsedResults'][0]
    
    # Extraer overlay correctamente según estructura real
    overlay = parsed_result.get('Overlay', {})
    lineas = overlay.get('Lines', [])
    
    return (
        parsed_result.get('ParsedText', '').strip(), 
        lineas
    )

def recortar_cuadro_blanco_np(image_np):
    """Recorta el cuadro blanco del comprobante para mejor detección"""
    if image_np is None or image_np.size == 0:
        return None
    
    gris = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 240, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return None
    
    contorno_mayor = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_mayor)
    
    if w < 100 or h < 100:
        return None
    
    recorte = image_np[y:y + h, x:x + w]
    return recorte

def extraer_destino_texto(texto):
    """
    Extrae el destino directamente del texto
    Busca la línea siguiente a "Destino:"
    """
    lineas = texto.split('\n')
    
    for i, linea in enumerate(lineas):
        # Buscar "Destino:" (case insensitive)
        if 'destino' in linea.lower() and ':' in linea:
            # Verificar línea siguiente
            if i + 1 < len(lineas):
                posible_destino = lineas[i + 1].strip()
                
                # Verificar que esté en la lista de destinos válidos
                if posible_destino in DESTINOS_VALIDOS:
                    return posible_destino
                
                # Verificar case-insensitive
                for destino_valido in DESTINOS_VALIDOS:
                    if posible_destino.lower() == destino_valido.lower():
                        return destino_valido
    
    return None

def detectar_estructura(texto):
    """
    Detecta el tipo de transacción Plin basado en estructura REAL
    
    Retorna:
        1: Envío (¡Pago exitoso! / Enviado a:)
        0: No es un comprobante Plin válido
    """
    texto_lower = texto.lower()
    
    # Estructura 1: Envío
    if any(patron in texto_lower for patron in [
        '¡pago exitoso!',
        'pago exitoso',
        'enviado a:'
    ]):
        return 1
    
    return 0

def validar_estructura_plin(texto):
    """
    Valida comprobante Plin basado en estructura REAL
    """
    resultado = {
        "tipo_transaccion": "Envío",
        "monto": None,
        "receptor": None,
        "telefono": None,
        "fecha": None,
        "hora": None,
        "destino": None,
        "comision": None,
        "codigo_operacion": None,
        "comentario": None
    }
    advertencias = []
    
    # 1. Extraer monto (CASE INSENSITIVE: s/ o S/)
    match_monto = re.search(r'[sS]/\s*(\d+(?:[.,]\d{2})?)', texto)
    
    if match_monto:
        try:
            monto_str = match_monto.group(1)
            # Reemplazar coma por punto si existe
            monto_str = monto_str.replace(',', '.')
            monto = float(monto_str)
            resultado["monto"] = monto
            
            if monto <= 0:
                advertencias.append("❌ Monto debe ser mayor a 0")
            elif monto > 500:
                advertencias.append("⚠️ Monto muy alto (mayor a S/ 500)")
        except ValueError:
            advertencias.append("❌ Error al convertir el monto")
    else:
        advertencias.append("❌ Monto no detectado")
    
    # 2. Extraer receptor (línea después de "Enviado a:")
    match_nombre = re.search(
        r'Enviado a:\s*[\r\n]+\s*([A-ZÁÉÍÓÚÑ][A-Za-záéíóúñ\s\.]+)',
        texto,
        re.IGNORECASE
    )
    
    if match_nombre:
        nombre_raw = match_nombre.group(1).strip()
        # Limpiar múltiples espacios y saltos de línea
        nombre_raw = nombre_raw.split('\n')[0].strip()
        resultado["receptor"] = re.sub(r'\s+', ' ', nombre_raw)
    else:
        advertencias.append("❌ Nombre del receptor no detectado")
    
    # 3. Extraer teléfono (formato: XXX XXX XXX)
    match_telefono = re.search(r'(\d{3})\s+(\d{3})\s+(\d{3})', texto)
    
    if match_telefono:
        resultado["telefono"] = f"{match_telefono.group(1)} {match_telefono.group(2)} {match_telefono.group(3)}"
    else:
        advertencias.append("⚠️ Número de teléfono no detectado")
    
    # 4. Extraer fecha y hora (formato: 01 Oct 2025 10:51 PM)
    match_fecha = re.search(
        r'(\d{1,2})\s+([A-Za-záéíóúñÑ]{3,10})\.?\s+(\d{4})\s+(\d{1,2}:\d{2})\s+([APap][Mm])',
        texto
    )
    
    if match_fecha:
        dia, mes_str, anio, hora, ampm = match_fecha.groups()
        resultado["fecha"] = f"{dia} {mes_str} {anio}"
        resultado["hora"] = f"{hora} {ampm.upper()}"
        
        # Validar fecha
        meses = {
            'ene': 1, 'enero': 1, 'jan': 1,
            'feb': 2, 'febrero': 2,
            'mar': 3, 'marzo': 3,
            'abr': 4, 'abril': 4, 'apr': 4,
            'may': 5, 'mayo': 5,
            'jun': 6, 'junio': 6,
            'jul': 7, 'julio': 7,
            'ago': 8, 'agosto': 8, 'aug': 8,
            'sep': 9, 'sept': 9, 'septiembre': 9,
            'oct': 10, 'octubre': 10,
            'nov': 11, 'noviembre': 11,
            'dic': 12, 'diciembre': 12, 'dec': 12
        }
        
        mes_limpio = mes_str.lower().strip('.')
        mes_num = meses.get(mes_limpio)
        
        if mes_num:
            try:
                fecha_obj = datetime(int(anio), mes_num, int(dia))
                hoy = datetime.now()
                
                if fecha_obj > hoy:
                    advertencias.append("⚠️ Fecha es futura")
                
                diferencia_dias = (hoy - fecha_obj).days
                if diferencia_dias > 365:
                    advertencias.append("⚠️ Fecha muy antigua (más de 1 año)")
            except ValueError:
                advertencias.append("❌ Fecha inválida (no existe en el calendario)")
        else:
            advertencias.append(f"❌ Mes no reconocido: {mes_str}")
    else:
        advertencias.append("❌ Fecha u hora no detectada")
    
    # 5. Extraer destino del texto directamente
    destino = extraer_destino_texto(texto)
    
    if destino:
        resultado["destino"] = destino
    else:
        advertencias.append("⚠️ Destino no detectado")
    
    # 6. Comisión (formato: GRATIS o S/ X.XX)
    match_comision = re.search(
        r'Comisi[oó]n:\s*[\r\n]+\s*([A-Z]+|[sS]/\s*\d+(?:[.,]\d{2})?)', 
        texto, 
        re.IGNORECASE
    )
    
    if match_comision:
        comision_text = match_comision.group(1).strip().upper()
        resultado["comision"] = comision_text
    else:
        advertencias.append("⚠️ Comisión no detectada")
    
    # 7. Código de operación (8 dígitos)
    match_codigo = re.search(
        r'(?:C[oó]digo de operaci[oó]n:|operaci[eé]n:)\s*[\r\n]+\s*(\d{8})', 
        texto, 
        re.IGNORECASE
    )
    
    if match_codigo:
        resultado["codigo_operacion"] = match_codigo.group(1)
    else:
        # Fallback: buscar cualquier secuencia de 8 dígitos
        match_codigo_fallback = re.search(r'\b(\d{8})\b', texto)
        if match_codigo_fallback:
            resultado["codigo_operacion"] = match_codigo_fallback.group(1)
        else:
            advertencias.append("❌ Código de operación no detectado")
    
    # 8. Extraer comentario (línea después de la fecha, si existe)
    if resultado["fecha"]:
        lineas = texto.split('\n')
        campos_sistema = [
            'código', 'codigo', 'operación', 'operacion',
            'destino', 'comisión', 'comision', 'gratis', 'interbank', 'plin'
        ]
        
        for i, linea in enumerate(lineas):
            # Buscar línea que contiene la fecha
            if any(parte in linea for parte in resultado["fecha"].split()):
                if i + 1 < len(lineas):
                    posible_comentario = lineas[i + 1].strip()
                    
                    # Verificar que no sea un campo del sistema
                    if posible_comentario and len(posible_comentario) > 3:
                        es_campo_sistema = any(
                            campo in posible_comentario.lower() 
                            for campo in campos_sistema
                        )
                        
                        if not es_campo_sistema:
                            resultado["comentario"] = posible_comentario
                break
    
    if advertencias:
        resultado["advertencias"] = advertencias
    else:
        resultado["advertencias"] = ["✅ Sin observaciones"]
    
    return resultado

@router.post("/filtro_ocr")
async def filtro_ocr_plin(file: UploadFile = File(...)):
    """
    Endpoint para validar comprobantes Plin mediante OCR
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=422, 
            detail="❌ El archivo debe ser una imagen válida"
        )
    
    try:
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=422,
                detail="❌ El archivo está vacío"
            )
        
        # Decodificar imagen
        np_arr = np.frombuffer(content, np.uint8)
        imagen = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if imagen is None:
            raise HTTPException(
                status_code=422,
                detail="❌ No se pudo decodificar la imagen"
            )
        
        # Redimensionar si es muy grande
        if imagen.shape[0] > 2000 or imagen.shape[1] > 2000:
            factor = min(2000 / imagen.shape[0], 2000 / imagen.shape[1])
            imagen = cv2.resize(imagen, (0, 0), fx=factor, fy=factor)
        
        # Recortar cuadro blanco
        recorte = recortar_cuadro_blanco_np(imagen)
        
        if recorte is None:
            recorte = imagen
        
        # Convertir a bytes para OCR
        _, buffer = cv2.imencode(".png", recorte)
        img_bytes = buffer.tobytes()
        
        # Enviar a OCR
        texto, lineas_overlay = enviar_imagen_ocr_bytes(
            ('comprobante.png', img_bytes, 'image/png')
        )
        
        if not texto:
            raise HTTPException(
                status_code=422,
                detail="❌ No se pudo extraer texto del comprobante"
            )
        
        # Detectar tipo de estructura
        tipo = detectar_estructura(texto)
        
        if tipo == 0:
            raise HTTPException(
                status_code=422,
                detail="❌ No se reconoce como un comprobante Plin válido. "
                       "Verifica que la imagen sea clara y contenga '¡Pago exitoso!' o 'Enviado a:'"
            )
        
        # Validar estructura
        resultado = validar_estructura_plin(texto)
        
        # Agregar texto completo para debug
        resultado["texto_ocr"] = texto
        
        return resultado
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"❌ Error inesperado: {str(e)}"
        )