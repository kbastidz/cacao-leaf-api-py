from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import traceback
import logging

# Configuración inicial
app = FastAPI(
    title="Cacao Leaf Analyzer API",
    description="API para analizar hojas de cacao y detectar enfermedades",
    version="3.0"
)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analizar_hoja(image_bytes, debug=False):
    """
    Analiza una imagen de hoja de cacao y detecta posibles enfermedades
    """
    try:
        logger.info(f"Procesando imagen de {len(image_bytes)} bytes")

        image = Image.open(io.BytesIO(image_bytes))
        logger.info(f"Imagen abierta: {image.format}, {image.size}, {image.mode}")

        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info("Convertido a RGB")

        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Reducir ruido
        img_blur = cv2.GaussianBlur(img_cv, (5, 5), 0)

        # Color promedio
        mean_color = cv2.mean(img_blur)[:3]
        b_mean, g_mean, r_mean = mean_color
        logger.info(f"Color promedio - R:{r_mean:.2f}, G:{g_mean:.2f}, B:{b_mean:.2f}")

        # Conversión a HSV
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        h_mean, s_mean, v_mean, _ = cv2.mean(hsv)
        logger.info(f"HSV promedio - H:{h_mean:.2f}, S:{s_mean:.2f}, V:{v_mean:.2f}")

        # Análisis de textura
        gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        texture_score = np.mean(edges) / 255
        logger.info(f"Texture score: {texture_score:.4f}")

        # === DETECCIÓN MEJORADA DE MANCHAS ===
        # Detectar manchas marrones/oscuras (hongos/necrosis)
        # Rango HSV para marrones: H en 10-25, S moderado-alto, V bajo-medio
        mask_brown_spots = cv2.inRange(hsv, (5, 40, 20), (25, 255, 120))
        brown_spot_area = np.sum(mask_brown_spots > 0) / mask_brown_spots.size
        
        # Detectar manchas muy oscuras (necrosis severa)
        mask_dark_spots = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        dark_spot_area = np.sum(mask_dark_spots > 0) / mask_dark_spots.size
        
        # Detectar áreas amarillas (clorosis/deficiencias)
        mask_yellow = cv2.inRange(hsv, (20, 40, 100), (35, 255, 255))
        yellow_area = np.sum(mask_yellow > 0) / mask_yellow.size
        
        logger.info(f"Brown spots: {brown_spot_area:.4f}, Dark spots: {dark_spot_area:.4f}, Yellow: {yellow_area:.4f}")

        # === ANÁLISIS DE SALUD DE LA HOJA ===
        # Verde saludable: H entre 35-85, S > 30, V > 60
        mask_healthy_green = cv2.inRange(hsv, (35, 30, 60), (85, 255, 255))
        healthy_green_ratio = np.sum(mask_healthy_green > 0) / mask_healthy_green.size
        logger.info(f"Verde saludable: {healthy_green_ratio:.4f}")

        # Cálculo de saturación promedio (indica vitalidad)
        saturation_avg = s_mean / 255
        
        # === MÉTRICAS DE DECISIÓN ===
        total_damage = brown_spot_area + dark_spot_area + yellow_area
        
        # Inicializar valores
        estado_general = ""
        posible_enfermedad = ""
        color_principal = ""
        confianza = 0.0
        
        # === LÓGICA DE DECISIÓN MEJORADA ===
        
        # 1. HOJA SANA
        if (healthy_green_ratio > 0.65 and 
            brown_spot_area < 0.08 and 
            dark_spot_area < 0.05 and 
            yellow_area < 0.15 and
            saturation_avg > 0.25):
            
            estado_general = "Sana"
            posible_enfermedad = "Ninguna"
            color_principal = "verde"
            confianza = min(0.95, 0.70 + (healthy_green_ratio * 0.25))
        
        # 2. MANCHAS OSCURAS/MARRONES SIGNIFICATIVAS (Hongos)
        elif brown_spot_area > 0.12 or dark_spot_area > 0.10:
            estado_general = "Enfermedad fúngica"
            posible_enfermedad = "Posible Moniliasis, Phytophthora o Antracnosis"
            color_principal = "verde con manchas marrones"
            confianza = min(0.92, 0.65 + (total_damage * 1.5))
        
        # 3. MANCHAS MODERADAS
        elif brown_spot_area > 0.08 or dark_spot_area > 0.06:
            estado_general = "Posible infección fúngica inicial"
            posible_enfermedad = "Etapa temprana de hongo foliar o Cercospora"
            color_principal = "verde con manchas leves"
            confianza = 0.70 + (brown_spot_area * 2)
        
        # 4. ALTA PRESENCIA DE AMARILLO (Deficiencia nutricional)
        elif yellow_area > 0.25:
            estado_general = "Deficiencia nutricional"
            posible_enfermedad = "Posible falta de nitrógeno o magnesio"
            color_principal = "amarillento"
            confianza = 0.75 + (yellow_area * 0.6)
        
        # 5. TEXTURA IRREGULAR (Daño físico o plagas)
        elif texture_score > 0.30:
            estado_general = "Daño físico o plaga"
            posible_enfermedad = "Posible ataque de insectos masticadores"
            color_principal = "verde con bordes irregulares"
            confianza = 0.68 + (texture_score * 0.6)
        
        # 6. BAJA SATURACIÓN (Estrés o senescencia)
        elif saturation_avg < 0.20 and v_mean < 100:
            estado_general = "Estrés o envejecimiento"
            posible_enfermedad = "Senescencia natural o estrés hídrico"
            color_principal = "verde pálido"
            confianza = 0.70
        
        # 7. CASOS INTERMEDIOS (ligeros indicios)
        elif total_damage > 0.10 or healthy_green_ratio < 0.50:
            estado_general = "Posible estrés leve"
            posible_enfermedad = "Monitorear evolución - síntomas no concluyentes"
            color_principal = "verde variable"
            confianza = 0.60 + (total_damage * 0.8)
        
        # 8. DEFAULT - SANA
        else:
            estado_general = "Sana"
            posible_enfermedad = "Ninguna"
            color_principal = "verde"
            confianza = 0.85
        
        # Limitar confianza entre 0.50 y 0.98
        probabilidad = round(min(0.98, max(0.50, confianza)), 2)
        
        # === CARACTERÍSTICAS DETECTADAS ===
        caracteristicas = {
            "color_principal": color_principal,
            "manchas": "circulares marrones" if brown_spot_area > 0.08 else "no significativas",
            "borde": "irregular" if texture_score > 0.25 else "regular",
            "textura": "seca/necrótica" if v_mean < 80 else "normal",
            "deformaciones": bool(texture_score > 0.30)
        }

        # Resultado final
        resultado = {
            "estado_general": estado_general,
            "probabilidad": probabilidad,
            "caracteristicas_detectadas": caracteristicas,
            "posible_enfermedad": posible_enfermedad
        }

        # Modo debug opcional
        if debug:
            resultado["debug"] = {
                "mean_rgb": {"r": round(r_mean, 2), "g": round(g_mean, 2), "b": round(b_mean, 2)},
                "brown_spot_area": round(brown_spot_area, 4),
                "dark_spot_area": round(dark_spot_area, 4),
                "yellow_area": round(yellow_area, 4),
                "healthy_green_ratio": round(healthy_green_ratio, 4),
                "texture_score": round(texture_score, 4),
                "saturation_avg": round(saturation_avg, 4),
                "hsv_mean": {"h": round(h_mean, 2), "s": round(s_mean, 2), "v": round(v_mean, 2)}
            }

        logger.info(f"Análisis completado: {estado_general} (conf: {probabilidad})")
        return resultado

    except Exception as e:
        logger.error(f"Error en analizar_hoja: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@app.get("/")
async def root():
    return {
        "message": "API de Análisis de Hojas de Cacao",
        "status": "activa",
        "version": "3.0",
        "endpoints": {
            "POST /analizar-hoja": "Analiza una imagen de hoja de cacao (agregar ?debug=true para métricas)",
            "GET /": "Información de la API",
            "GET /health": "Health check"
        }
    }


@app.post("/analizar-hoja")
async def analizar_hoja_endpoint(file: UploadFile = File(...), debug: bool = False):
    logger.info("=== Nueva petición POST /analizar-hoja ===")
    logger.info(f"Content-Type: {file.content_type}")
    logger.info(f"Filename: {file.filename}")

    try:
        if file.content_type and not file.content_type.startswith('image/'):
            return JSONResponse(
                content={"error": "El archivo debe ser una imagen"},
                status_code=400
            )

        image_bytes = await file.read()
        if len(image_bytes) == 0:
            return JSONResponse(
                content={"error": "El archivo está vacío"},
                status_code=400
            )

        resultado = analizar_hoja(image_bytes, debug)
        return JSONResponse(content=resultado)

    except UnidentifiedImageError:
        return JSONResponse(
            content={"error": "No se pudo procesar la imagen. Asegúrate de que sea un archivo de imagen válido"},
            status_code=400
        )
    except Exception as e:
        logger.error(f"Error al procesar: {str(e)}")
        return JSONResponse(
            content={"error": f"Error al procesar la imagen: {str(e)}"},
            status_code=500
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cacao-api", "version": "3.0"}


@app.options("/analizar-hoja")
async def options_analizar_hoja():
    return JSONResponse(content={"message": "OK"})
