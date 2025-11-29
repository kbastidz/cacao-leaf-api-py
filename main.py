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
    version="4.0"
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

        # Color promedio RGB
        mean_color = cv2.mean(img_blur)[:3]
        b_mean, g_mean, r_mean = mean_color

        # Conversión a HSV
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        h_mean, s_mean, v_mean, _ = cv2.mean(hsv)

        # Análisis de textura
        gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        texture_score = np.mean(edges) / 255

        # === DETECCIÓN MEJORADA ===

        # Manchas marrones (hongos/necrosis)
        mask_brown_spots = cv2.inRange(hsv, (5, 40, 20), (25, 255, 120))
        brown_spot_area = np.sum(mask_brown_spots > 0) / mask_brown_spots.size

        # Manchas oscuras severas
        mask_dark_spots = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        dark_spot_area = np.sum(mask_dark_spots > 0) / mask_dark_spots.size

        # Clorosis fuerte (amarillo intenso)
        mask_yellow = cv2.inRange(hsv, (20, 40, 100), (35, 255, 255))
        yellow_area = np.sum(mask_yellow > 0) / mask_yellow.size

        # NECROSIS DE BORDE (nuevo)
        mask_edge_necrosis = cv2.inRange(hsv, (0, 0, 20), (25, 180, 90))
        edge_necrosis_area = np.sum(mask_edge_necrosis > 0) / mask_edge_necrosis.size

        # CLOROSIS SUAVE / estrés leve (nuevo)
        mask_soft_chlorosis = cv2.inRange(hsv, (18, 20, 120), (40, 150, 255))
        soft_chlorosis_area = np.sum(mask_soft_chlorosis > 0) / mask_soft_chlorosis.size

        # Áreas verdes saludables
        mask_healthy_green = cv2.inRange(hsv, (35, 30, 60), (85, 255, 255))
        healthy_green_ratio = np.sum(mask_healthy_green > 0) / mask_healthy_green.size

        saturation_avg = s_mean / 255
        total_damage = brown_spot_area + dark_spot_area + yellow_area

        logger.info(f"Brown: {brown_spot_area:.4f}, Dark: {dark_spot_area:.4f}, Yellow: {yellow_area:.4f}")
        logger.info(f"Edge necrosis: {edge_necrosis_area:.4f}, Soft chlorosis: {soft_chlorosis_area:.4f}")
        logger.info(f"Healthy green: {healthy_green_ratio:.4f}")

        # === LÓGICA DE DECISIÓN MEJORADA ===

        estado_general = ""
        posible_enfermedad = ""
        color_principal = ""
        confianza = 0.0

        # NECROSIS DE BORDE (nuevo - coincide con la hoja que me enviaste)
        if edge_necrosis_area > 0.05:
            estado_general = "Estrés moderado"
            posible_enfermedad = "Necrosis en bordes (posible deficiencia de potasio o estrés hídrico)"
            color_principal = "verde con bordes oscurecidos"
            confianza = 0.75 + (edge_necrosis_area * 0.8)

        # Enfermedad fúngica avanzada
        elif brown_spot_area > 0.12 or dark_spot_area > 0.10:
            estado_general = "Enfermedad fúngica"
            posible_enfermedad = "Posible Antracnosis o daño fúngico avanzado"
            color_principal = "verde con manchas marrones"
            confianza = min(0.92, 0.70 + total_damage)

        # Infección inicial
        elif brown_spot_area > 0.06 or dark_spot_area > 0.05:
            estado_general = "Posible infección fúngica inicial"
            posible_enfermedad = "Manchas iniciales por Cercospora"
            color_principal = "verde con manchas leves"
            confianza = 0.65 + (brown_spot_area * 1.2)

        # Deficiencia nutricional
        elif yellow_area > 0.20 or soft_chlorosis_area > 0.15:
            estado_general = "Deficiencia nutricional"
            posible_enfermedad = "Posible desbalance de nitrógeno o magnesio"
            color_principal = "verde amarillento"
            confianza = 0.70 + (yellow_area + soft_chlorosis_area)

        # Estrés leve
        elif total_damage > 0.10:
            estado_general = "Posible estrés leve"
            posible_enfermedad = "Síntomas leves, requiere monitoreo"
            color_principal = "verde irregular"
            confianza = 0.60 + (total_damage * 0.6)

        # Hoja sana
        elif healthy_green_ratio > 0.65:
            estado_general = "Sana"
            posible_enfermedad = "Ninguna"
            color_principal = "verde uniforme"
            confianza = 0.80 + (healthy_green_ratio * 0.15)

        # Default
        else:
            estado_general = "Sana"
            posible_enfermedad = "Ninguna"
            color_principal = "verde"
            confianza = 0.75

        probabilidad = round(min(0.98, max(0.50, confianza)), 2)

        # === CARACTERÍSTICAS DETECTADAS MEJORADAS ===
        caracteristicas = {
            "color_principal": color_principal,
            "manchas": (
                "manchas marrones dispersas" if brown_spot_area > 0.06
                else "manchas oscuras" if dark_spot_area > 0.05
                else "sin manchas relevantes"
            ),
            "borde": (
                "necrosis en bordes" if edge_necrosis_area > 0.05
                else "desgaste leve" if texture_score > 0.25
                else "regular"
            ),
            "textura": (
                "opaca con signos de estrés" if soft_chlorosis_area > 0.15
                else "normal"
            ),
            "deformaciones": texture_score > 0.30
        }

        # Resultado final
        resultado = {
            "estado_general": estado_general,
            "probabilidad": probabilidad,
            "caracteristicas_detectadas": caracteristicas,
            "posible_enfermedad": posible_enfermedad
        }

        if debug:
            resultado["debug"] = {
                "mean_rgb": {"r": round(r_mean, 2), "g": round(g_mean, 2), "b": round(b_mean, 2)},
                "brown_spot_area": round(brown_spot_area, 4),
                "dark_spot_area": round(dark_spot_area, 4),
                "yellow_area": round(yellow_area, 4),
                "edge_necrosis_area": round(edge_necrosis_area, 4),
                "soft_chlorosis_area": round(soft_chlorosis_area, 4),
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
        "version": "4.0",
        "endpoints": {
            "POST /analizar-hoja": "Analiza una imagen de hoja de cacao (?debug=true para detalles)",
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
        return JSONResponse(
            content={"error": f"Error al procesar la imagen: {str(e)}"},
            status_code=500
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cacao-api", "version": "4.0"}


@app.options("/analizar-hoja")
async def options_analizar_hoja():
    return JSONResponse(content={"message": "OK"})
