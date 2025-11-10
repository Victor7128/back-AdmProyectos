# Filtro OCR para validación de comprobantes Plin
import cv2
import numpy as np
import re
import requests
import os
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Optional, List, Tuple, Dict

router = APIRouter()

API_KEY = os.getenv('OCR_API_KEY', 'e0b0a3ad7d88957')
OCR_API_URL = 'https://api.ocr.space/parse/image'

DESTINOS_VALIDOS = [
    'Yape', 'Plin', 'BCP', 'Interbank', 'BBVA', 'Scotiabank',
    'Caja Arequipa', 'Caja Huancayo', 'Caja Piura', 'Caja Cusco',
    'Caja Trujillo', 'MiBanco', 'Banco de la Nación', 'Caja Sullana',
    'Caja Tacna', 'Caja Metropolitana', 'Banco Pichincha'
]

MESES = {
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

def enviar_imagen_ocr_bytes(imagen_bytes) -> Tuple[Optional[str], List]:
    try:
        response = requests.post(
            OCR_API_URL,
            files={'file': imagen_bytes},
            data={
                'apikey': API_KEY,
                'language': 'spa',
                'OCREngine': '2',
                'isOverlayRequired': True,
                'scale': True,
                'detectOrientation': True
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
            detail=f"❌ OCR externo respondió con código: {response.status_code}"
        )
    
    try:
        resultado = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="❌ Respuesta inválida del OCR externo"
        )
    
    if resultado.get("IsErroredOnProcessing"):
        error_msg = resultado.get("ParsedResults", [{}])[0].get("ErrorMessage", "Error desconocido")
        raise HTTPException(
            status_code=422,
            detail=f"❌ Error en el procesamiento OCR: {error_msg}"
        )
    
    if not resultado.get('ParsedResults'):
        raise HTTPException(
            status_code=422,
            detail="❌ No se pudieron obtener resultados del OCR"
        )
    
    parsed_result = resultado['ParsedResults'][0]
    texto = parsed_result.get('ParsedText', '').strip()
    lineas = parsed_result.get('Overlay', {}).get('Lines', [])
    
    return texto, lineas

def recortar_cuadro_blanco_np(image_np: np.ndarray) -> Optional[np.ndarray]:
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

def detectar_estructura(texto: str) -> int:
    texto_lower = texto.lower()
    
    if any(patron in texto_lower for patron in [
        'enviaste con plin',
        'pago exitoso',
        '¡pago exitoso!',
        'enviado a:'
    ]):
        return 1
    
    if any(patron in texto_lower for patron in [
        'recibiste con plin',
        'recibido de:'
    ]):
        return 2
    
    return 0

def extraer_monto(texto: str) -> Tuple[Optional[float], List[str]]:
    advertencias = []
    match_monto = re.search(r'S/\s*(\d{1,6}(?:\.\d{2})?)', texto)
    
    if match_monto:
        try:
            monto_str = match_monto.group(1)
            monto = float(monto_str)
            
            if monto <= 0:
                advertencias.append("❌ Monto debe ser mayor a 0")
                return None, advertencias
            elif monto > 500:
                advertencias.append("⚠️ Monto muy alto (mayor a S/ 500)")
            
            return monto, advertencias
        except ValueError:
            advertencias.append("❌ Error al convertir el monto")
            return None, advertencias

    match_alt = re.search(r'\b(\d{1,6}(?:\.\d{2})?)\b', texto)
    if match_alt:
        try:
            monto = float(match_alt.group(1))
            if 0 < monto <= 2000:
                advertencias.append("⚠️ Monto encontrado sin símbolo S/")
                return monto, advertencias
        except ValueError:
            pass
    
    advertencias.append("❌ Monto no detectado")
    return None, advertencias

def extraer_fecha_hora(texto: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    advertencias = []
    match = re.search(
        r'(\d{1,2})\s+([a-zA-ZáéíóúñÑ]{3,10})\.?\s+(\d{4})\s+(\d{1,2}:\d{2}\s*[APap][Mm])',
        texto
    )
    
    if not match:
        advertencias.append("❌ Fecha u hora no detectada")
        return None, None, advertencias
    
    dia, mes_str, anio, hora = match.groups()
    
    mes_limpio = mes_str.lower().strip('.')
    mes_num = MESES.get(mes_limpio)
    
    if not mes_num:
        advertencias.append(f"❌ Mes no reconocido: {mes_str}")
        return None, None, advertencias
    try:
        fecha_obj = datetime(int(anio), mes_num, int(dia))
        hoy = datetime.now()
        
        if fecha_obj > hoy:
            advertencias.append("⚠️ Fecha es futura")
        diferencia_dias = (hoy - fecha_obj).days
        if diferencia_dias > 365:
            advertencias.append("⚠️ Fecha muy antigua (más de 1 año)")
        
        fecha_formateada = f"{dia} {mes_str} {anio}"
        return fecha_formateada, hora.strip(), advertencias
        
    except ValueError:
        advertencias.append("❌ Fecha inválida (no existe en el calendario)")
        return None, None, advertencias

def extraer_destinatario(texto: str, lineas: List) -> Tuple[Optional[str], Optional[str], List[str]]:
    advertencias = []
    nombre = None
    numero = None
    
    match_nombre = re.search(
        r'(?:Enviado a:|Recibido de:)\s*\n?\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ\s\.]{2,50})',
        texto,
        re.IGNORECASE
    )
    
    if match_nombre:
        nombre_raw = match_nombre.group(1).strip()
        nombre = re.sub(r'\s+', ' ', nombre_raw)
    match_numero = re.search(r'(\d{3})\s*(\d{3})\s*(\d{3})', texto)
    if match_numero:
        numero = f"{match_numero.group(1)} {match_numero.group(2)} {match_numero.group(3)}"
    
    if not nombre:
        advertencias.append("❌ Nombre del destinatario no detectado")
    
    if not numero:
        advertencias.append("⚠️ Número de teléfono no detectado")
    
    return nombre, numero, advertencias

def extraer_destino(lineas: List) -> Tuple[Optional[str], List[str]]:
    advertencias = []
    destino = None
    destino_y = None
    destino_x = None
    
    for linea in lineas:
        for palabra in linea.get('Words', []):
            try:
                texto = palabra['WordText']
                if 'destino' in texto.lower():
                    destino_y = palabra['Top'] + palabra['Height'] // 2
                    destino_x = palabra['Left'] + palabra['Width']
                    break
            except (KeyError, TypeError):
                continue
    
    if destino_y and destino_x:
        margen = 50
        posibles_destinos = []
        
        for linea in lineas:
            for palabra in linea.get('Words', []):
                try:
                    left = palabra['Left']
                    top = palabra['Top']
                    texto = palabra['WordText'].strip()
                    if (abs(top - destino_y) <= margen or top > destino_y) and left >= destino_x - 10:
                        if texto in DESTINOS_VALIDOS:
                            posibles_destinos.append(texto)
                        elif texto.lower() in [d.lower() for d in DESTINOS_VALIDOS]:
                            for dest in DESTINOS_VALIDOS:
                                if dest.lower() == texto.lower():
                                    posibles_destinos.append(dest)
                                    break
                except (KeyError, TypeError):
                    continue
        
        if posibles_destinos:
            destino = posibles_destinos[0]
    
    if not destino:
        advertencias.append("❌ Destino no detectado")
    
    return destino, advertencias

def extraer_codigo_operacion(texto: str) -> Tuple[Optional[str], List[str]]:
    advertencias = []
    match = re.search(r'\b(\d{8})\b', texto)
    
    if match:
        return match.group(1), advertencias
    
    advertencias.append("❌ Código de operación no detectado")
    return None, advertencias

def extraer_comentario(texto: str, fecha_str: Optional[str]) -> Optional[str]:
    if not fecha_str:
        return None
    
    lineas = texto.splitlines()
    campos_sistema = [
        'nro. de operación', 'código de operación', 'destino', 
        'datos de la transacción', 'plin', 'comisión', 'gratis',
        'fecha y hora', 'interbank'
    ]
    
    for i, linea in enumerate(lineas):
        if any(parte in linea for parte in fecha_str.split()):
            if i + 1 < len(lineas):
                posible_comentario = lineas[i + 1].strip()
                if posible_comentario and len(posible_comentario) > 3:
                    es_campo_sistema = any(
                        campo in posible_comentario.lower() 
                        for campo in campos_sistema
                    )
                    
                    if not es_campo_sistema:
                        return posible_comentario
    
    return None

def validar_comprobante_plin(
    texto: str,
    lineas: List,
    tipo_estructura: int
) -> Dict:
    resultado = {
        "tipo_transaccion": "Envío" if tipo_estructura == 1 else "Recepción",
        "monto": None,
        "destinatario": None,
        "telefono": None,
        "fecha": None,
        "hora": None,
        "codigo_operacion": None,
        "destino": None,
        "comentario": None,
        "es_valido": True,
        "advertencias": []
    }
    
    todas_advertencias = []
    
    monto, adv_monto = extraer_monto(texto)
    resultado["monto"] = monto
    todas_advertencias.extend(adv_monto)
    
    fecha, hora, adv_fecha = extraer_fecha_hora(texto)
    resultado["fecha"] = fecha
    resultado["hora"] = hora
    todas_advertencias.extend(adv_fecha)
    
    nombre, telefono, adv_dest = extraer_destinatario(texto, lineas)
    resultado["destinatario"] = nombre
    resultado["telefono"] = telefono
    todas_advertencias.extend(adv_dest)
    
    destino, adv_destino = extraer_destino(lineas)
    resultado["destino"] = destino
    todas_advertencias.extend(adv_destino)
    
    codigo, adv_codigo = extraer_codigo_operacion(texto)
    resultado["codigo_operacion"] = codigo
    todas_advertencias.extend(adv_codigo)

    comentario = extraer_comentario(texto, fecha)
    resultado["comentario"] = comentario
    campos_criticos = [
        resultado["monto"],
        resultado["fecha"],
        resultado["codigo_operacion"],
        resultado["destinatario"]
    ]
    
    if not all(campos_criticos):
        resultado["es_valido"] = False
    advertencias_criticas = [
        adv for adv in todas_advertencias if adv.startswith("❌")
    ]
    
    if advertencias_criticas:
        resultado["es_valido"] = False
    
    resultado["advertencias"] = todas_advertencias if todas_advertencias else ["✅ Sin observaciones"]
    
    return resultado

@router.post("/filtro_ocr")
async def filtro_ocr(file: UploadFile = File(...)):
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
        
        np_arr = np.frombuffer(content, np.uint8)
        imagen = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if imagen is None:
            raise HTTPException(
                status_code=422,
                detail="❌ No se pudo decodificar la imagen"
            )
        if imagen.shape[0] > 2000 or imagen.shape[1] > 2000:
            factor = min(2000 / imagen.shape[0], 2000 / imagen.shape[1])
            imagen = cv2.resize(imagen, (0, 0), fx=factor, fy=factor)
        recorte = recortar_cuadro_blanco_np(imagen)
        
        if recorte is None:
            recorte = imagen
        _, buffer = cv2.imencode(".png", recorte)
        img_bytes = buffer.tobytes()
        texto, lineas_overlay = enviar_imagen_ocr_bytes(
            ('comprobante.png', img_bytes, 'image/png')
        )
        
        if not texto:
            raise HTTPException(
                status_code=422,
                detail="❌ No se pudo extraer texto del comprobante"
            )
        tipo = detectar_estructura(texto)
        
        if tipo == 0:
            raise HTTPException(
                status_code=422,
                detail="❌ No se reconoce como un comprobante Plin válido. "
                       "Verifica que la imagen sea clara y esté completa."
            )
        resultado = validar_comprobante_plin(texto, lineas_overlay, tipo)
        resultado["texto_ocr"] = texto
        
        return resultado
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"❌ Error inesperado: {str(e)}"
        )