# Filtro Exif
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
from PIL.ExifTags import TAGS
import io
import piexif

router = APIRouter()

def _convert_gps_to_decimal(coord, ref):
    """Convierte coordenadas GPS EXIF a grados decimales"""
    try:
        d, m, s = coord
        decimal = d[0] / d[1] + (m[0] / m[1]) / 60 + (s[0] / s[1]) / 3600
        if ref in ["S", "W"]:
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None

def extraer_exif(imagen_bytes: bytes):
    try:
        imagen = Image.open(io.BytesIO(imagen_bytes))
        
        # Información básica de la imagen
        info_basica = {
            "format": imagen.format,
            "mode": imagen.mode,
            "size": imagen.size,
            "width": imagen.width,
            "height": imagen.height
        }
        
        # Intentar extraer EXIF con PIL primero (más compatible)
        exif_data = imagen.getexif()
        datos_utiles = {}
        sospechoso = False
        
        if exif_data:
            # Usar PIL para extraer datos básicos
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='ignore')
                    except:
                        value = str(value)
                datos_utiles[tag] = value
            
            # Detección de edición
            software = str(datos_utiles.get("Software", "")).lower()
            processing_software = str(datos_utiles.get("ProcessingSoftware", "")).lower()
            
            if any(s in software + processing_software for s in [
                "photoshop", "gimp", "snapseed", "lightroom", "canva", 
                "screenshot", "capture", "paint", "editor"
            ]):
                sospechoso = True
        
        # Si PIL no encuentra EXIF, intentar con piexif
        if not datos_utiles:
            try:
                exif_raw = imagen.info.get('exif')
                if exif_raw:
                    exif_dict = piexif.load(exif_raw)
                    
                    campos_interes = {
                        "0th": ["Make", "Model", "Software", "Orientation"],
                        "Exif": ["DateTimeOriginal", "LensModel", "ISOSpeedRatings"],
                        "GPS": ["GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef"]
                    }

                    for ifd in campos_interes:
                        if ifd in exif_dict:
                            for campo in campos_interes[ifd]:
                                tag = piexif.TAGS[ifd]
                                tag_id = next((tid for tid, val in tag.items() if val["name"] == campo), None)
                                if tag_id and tag_id in exif_dict[ifd]:
                                    valor = exif_dict[ifd][tag_id]
                                    if isinstance(valor, bytes):
                                        valor = valor.decode(errors="ignore")
                                    datos_utiles[campo] = valor

                    # Procesar coordenadas GPS
                    if "GPSLatitude" in datos_utiles and "GPSLongitude" in datos_utiles:
                        lat = _convert_gps_to_decimal(
                            datos_utiles["GPSLatitude"], 
                            datos_utiles.get("GPSLatitudeRef", "N")
                        )
                        lon = _convert_gps_to_decimal(
                            datos_utiles["GPSLongitude"], 
                            datos_utiles.get("GPSLongitudeRef", "E")
                        )
                        if lat is not None and lon is not None:
                            datos_utiles["GPS"] = {"latitud": lat, "longitud": lon}
                        # Limpiar campos GPS individuales
                        for gps_field in ["GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef"]:
                            datos_utiles.pop(gps_field, None)
            except Exception as e:
                print(f"Error con piexif: {e}")

        # Solo mostrar advertencia si el formato NO es JPEG/JPG o PNG
        formato = (info_basica.get("format") or "").lower()
        if formato not in ["jpeg", "jpg", "png"]:
            advertencia_formato = "Sospechoso"
        else:
            advertencia_formato = ""

        if not datos_utiles:
            mensaje = "Sospechoso"
        else:
            mensaje = "Metadatos EXIF extraídos correctamente"

        return {
            "info_imagen": info_basica,
            "datos": datos_utiles,
            "editado": sospechoso,
            "tiene_gps": "GPS" in datos_utiles,
            "advertencia_formato": advertencia_formato
        }, mensaje

    except Exception as e:
        return None, f"❌ Error procesando imagen: {e}"

@router.post("/filtro_exif")
async def filtro_exif(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=422, 
                detail="El archivo debe ser una imagen válida"
            )

        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413, 
                detail="El archivo es demasiado grande (máximo 10MB)"
            )

        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=422, 
                detail="El archivo está vacío"
            )

        resultado, mensaje = extraer_exif(content)

        if resultado is None:
            raise HTTPException(status_code=500, detail=mensaje)

        return {
            "archivo": file.filename,
            "mensaje": mensaje,
            "info_imagen": resultado["info_imagen"],
            "editado": resultado["editado"],
            "tiene_gps": resultado["tiene_gps"],
            "exif": resultado["datos"],
            "advertencia_formato": resultado["advertencia_formato"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error interno: {e}") 