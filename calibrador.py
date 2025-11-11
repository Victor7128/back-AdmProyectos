import cv2
import numpy as np
from pathlib import Path

def analyze_plin_image(image_path: str):
    """
    Analiza una imagen de Plin y extrae estadísticas de colores
    para calibrar el filtro
    """
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"❌ No se pudo cargar la imagen: {image_path}")
        return
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_pixels = img.shape[0] * img.shape[1]
    
    print(f"\n{'='*60}")
    print(f"📊 ANÁLISIS DE IMAGEN PLIN")
    print(f"{'='*60}")
    print(f"📁 Archivo: {Path(image_path).name}")
    print(f"📐 Dimensiones: {img.shape[1]}x{img.shape[0]} ({total_pixels:,} pixels)")
    print(f"{'='*60}\n")
    
    # Definir rangos a probar
    rangos = {
        "🔷 Turquesa Estricto": {
            "lower": [80, 30, 40],
            "upper": [100, 255, 255],
            "descripcion": "Rango original restrictivo"
        },
        "🔷 Turquesa Amplio": {
            "lower": [70, 25, 35],
            "upper": [110, 255, 255],
            "descripcion": "Más tolerante a variaciones"
        },
        "🔷 Turquesa Muy Amplio": {
            "lower": [75, 20, 30],
            "upper": [105, 255, 255],
            "descripcion": "Máxima flexibilidad"
        },
        "🔵 Azul": {
            "lower": [100, 50, 50],
            "upper": [130, 255, 255],
            "descripcion": "Azul de UI"
        },
        "💠 Cyan": {
            "lower": [85, 50, 50],
            "upper": [95, 255, 255],
            "descripcion": "Cyan específico"
        },
        "⚪ Blanco Estricto": {
            "lower": [0, 0, 200],
            "upper": [180, 30, 255],
            "descripcion": "Blanco puro"
        },
        "⚪ Blanco Flexible": {
            "lower": [0, 0, 180],
            "upper": [180, 40, 255],
            "descripcion": "Blanco con tolerancia"
        },
        "⚪ Blanco Muy Flexible": {
            "lower": [0, 0, 160],
            "upper": [180, 50, 255],
            "descripcion": "Incluye grises claros"
        }
    }
    
    resultados = []
    
    print("🎨 DETECCIÓN DE COLORES:\n")
    
    for nombre, config in rangos.items():
        lower = np.array(config["lower"])
        upper = np.array(config["upper"])
        
        mask = cv2.inRange(hsv, lower, upper)
        pixels_detectados = cv2.countNonZero(mask)
        ratio = (pixels_detectados / total_pixels) * 100
        
        resultados.append({
            "nombre": nombre,
            "ratio": ratio,
            "pixels": pixels_detectados,
            "lower": config["lower"],
            "upper": config["upper"]
        })
        
        print(f"{nombre}")
        print(f"  📝 {config['descripcion']}")
        print(f"  📊 Ratio: {ratio:.2f}%")
        print(f"  🔢 Pixels: {pixels_detectados:,}")
        print(f"  🎨 HSV Range: H[{config['lower'][0]}-{config['upper'][0]}] "
              f"S[{config['lower'][1]}-{config['upper'][1]}] "
              f"V[{config['lower'][2]}-{config['upper'][2]}]")
        print()
    
    # Análisis de combinaciones
    print(f"{'='*60}")
    print("🔗 COMBINACIONES RECOMENDADAS:\n")
    
    turquesa_amplio = resultados[1]["ratio"]
    azul = resultados[3]["ratio"]
    cyan = resultados[4]["ratio"]
    blanco_flexible = resultados[6]["ratio"]
    
    colores_plin_total = turquesa_amplio + azul + cyan
    
    print(f"💡 Opción 1 - Colores Plin Combinados:")
    print(f"   Turquesa Amplio + Azul + Cyan = {colores_plin_total:.2f}%")
    print(f"   Blanco Flexible = {blanco_flexible:.2f}%")
    print()
    
    # Generar recomendación de umbrales
    print(f"{'='*60}")
    print("⚙️  CONFIGURACIÓN RECOMENDADA:\n")
    
    # Reducir umbral al 70% del valor detectado para dar margen
    umbral_turquesa_recomendado = max(0.01, (colores_plin_total * 0.7) / 100)
    umbral_blanco_recomendado = max(0.05, (blanco_flexible * 0.7) / 100)
    
    print(f"turquoise_ratio_thresh = {umbral_turquesa_recomendado:.3f}  # {umbral_turquesa_recomendado*100:.1f}%")
    print(f"white_ratio_thresh = {umbral_blanco_recomendado:.3f}  # {umbral_blanco_recomendado*100:.1f}%")
    print()
    
    # Código sugerido
    print(f"{'='*60}")
    print("💻 CÓDIGO PYTHON SUGERIDO:\n")
    
    codigo = f'''
# Rangos HSV optimizados para tu imagen Plin
lower_turquoise = np.array({resultados[1]["lower"]})
upper_turquoise = np.array({resultados[1]["upper"]})

lower_white = np.array({resultados[6]["lower"]})
upper_white = np.array({resultados[6]["upper"]})

# Umbrales calibrados
turquoise_ratio_thresh = {umbral_turquesa_recomendado:.3f}
white_ratio_thresh = {umbral_blanco_recomendado:.3f}
'''
    print(codigo)
    
    print(f"{'='*60}\n")
    
    # Guardar resultados
    return {
        "dimensiones": (img.shape[1], img.shape[0]),
        "total_pixels": total_pixels,
        "detecciones": resultados,
        "umbrales_recomendados": {
            "turquoise": umbral_turquesa_recomendado,
            "white": umbral_blanco_recomendado
        }
    }


def analyze_multiple_images(image_folder: str):
    """
    Analiza múltiples imágenes de Plin para encontrar
    valores óptimos que funcionen para todas
    """
    folder = Path(image_folder)
    
    if not folder.exists():
        print(f"❌ La carpeta no existe: {image_folder}")
        return
    
    # Buscar imágenes
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(folder.glob(ext))
    
    if not image_files:
        print(f"❌ No se encontraron imágenes en: {image_folder}")
        return
    
    print(f"\n🔍 Encontradas {len(image_files)} imágenes para analizar\n")
    
    todos_resultados = []
    
    for img_path in image_files:
        resultado = analyze_plin_image(str(img_path))
        if resultado:
            todos_resultados.append(resultado)
        print("\n" + "="*60 + "\n")
    
    # Análisis consolidado
    if len(todos_resultados) > 1:
        print(f"\n{'#'*60}")
        print(f"📈 ANÁLISIS CONSOLIDADO DE {len(todos_resultados)} IMÁGENES")
        print(f"{'#'*60}\n")
        
        umbrales_turquesa = [r["umbrales_recomendados"]["turquoise"] for r in todos_resultados]
        umbrales_blanco = [r["umbrales_recomendados"]["white"] for r in todos_resultados]
        
        # Usar el mínimo para que funcione con todas las imágenes
        umbral_turquesa_final = min(umbrales_turquesa)
        umbral_blanco_final = min(umbrales_blanco)
        
        print(f"🎯 UMBRALES FINALES (mínimos para cubrir todas las imágenes):\n")
        print(f"turquoise_ratio_thresh = {umbral_turquesa_final:.3f}  # {umbral_turquesa_final*100:.1f}%")
        print(f"white_ratio_thresh = {umbral_blanco_final:.3f}  # {umbral_blanco_final*100:.1f}%")
        print()
        
        print(f"📊 Rango detectado:")
        print(f"  Turquesa: {min(umbrales_turquesa)*100:.1f}% - {max(umbrales_turquesa)*100:.1f}%")
        print(f"  Blanco: {min(umbrales_blanco)*100:.1f}% - {max(umbrales_blanco)*100:.1f}%")
        print(f"\n{'#'*60}\n")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("🎨 CALIBRADOR DE FILTRO PLIN")
    print("="*60 + "\n")
    
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python calibrate_plin_filter.py <imagen.jpg>")
        print("  python calibrate_plin_filter.py <carpeta_con_imagenes/>")
        print("\nEjemplo:")
        print("  python calibrate_plin_filter.py plin_screenshot.jpg")
        print("  python calibrate_plin_filter.py ./imagenes_plin/")
        sys.exit(1)
    
    ruta = sys.argv[1]
    
    if Path(ruta).is_file():
        analyze_plin_image(ruta)
    elif Path(ruta).is_dir():
        analyze_multiple_images(ruta)
    else:
        print(f"❌ Ruta no válida: {ruta}")