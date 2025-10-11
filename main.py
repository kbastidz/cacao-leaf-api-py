from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import random
import traceback
import logging

# Configuración inicial
app = FastAPI(
    title="Cacao Leaf Analyzer API",
    description="API para analizar hojas de cacao y detectar enfermedades",
    version="2.1"
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
        r, g, b = mean_color
        logger.info(f"Color promedio - R:{r:.2f}, G:{g:.2f}, B:{b:.2f}")

        # Conversión a HSV
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        h, s, v, _ = cv2.mean(hsv)
        logger.info(f"HSV promedio - H:{h:.2f}, S:{s:.2f}, V:{v:.2f}")

        # Análisis de textura
        gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        texture_score = np.mean(edges) / 255
        logger.info(f"Texture score: {texture_score:.4f}")

        # Detección de manchas
        mask_spots = cv2.inRange(hsv, (0, 30, 0), (180, 255, 80))
        spot_area = np.sum(mask_spots > 0) / mask_spots.size
        logger.info(f"Spot area: {spot_area:.4f}")

        # Probabilidad simulada
        probabilidad = round(random.uniform(0.85, 0.97), 2)

        # === NUEVA LÓGICA DE DECISIÓN (más sensible) ===
        if spot_area > 0.035 or texture_score > 0.18:
            color_principal = "verde amarillento"
            estado_general = "Hongo foliar"
            posible_enfermedad = "Posible Cercospora o Phytophthora"
        elif r > g and r > b:
            color_principal = "amarillento"
            estado_general = "Deficiencia nutricional"
            posible_enfermedad = "Posible falta de nitrógeno"
        elif texture_score > 0.25:
            color_principal = "verde oscuro"
            estado_general = "Daño físico o plaga"
            posible_enfermedad = "Posible ataque de insectos"
        elif g > r and g > b:
            color_principal = "verde"
            estado_general = "Sana"
            posible_enfermedad = "Ninguna"
        else:
            color_principal = "indeterminado"
            estado_general = "Desconocido"
            posible_enfermedad = "Requiere análisis avanzado"

        # === Características detectadas (ajustadas) ===
        caracteristicas = {
            "color_principal": color_principal,
            "manchas": "circulares, marrones" if spot_area > 0.035 else "ninguna visible",
            "borde": "irregular" if texture_score > 0.18 else "regular",
            "textura": "seca" if v < 80 else "normal",
            "deformaciones": bool(texture_score > 0.25)
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
                "mean_rgb": {"r": round(r, 2), "g": round(g, 2), "b": round(b, 2)},
                "spot_area": round(spot_area, 4),
                "texture_score": round(texture_score, 4),
                "hsv_mean": {"h": round(h, 2), "s": round(s, 2), "v": round(v, 2)}
            }

        logger.info(f"Análisis completado: {estado_general}")
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
        "version": "2.1",
        "endpoints": {
            "POST /analizar-hoja": "Analiza una imagen de hoja de cacao",
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
    return {"status": "healthy", "service": "cacao-api", "version": "2.1"}


@app.options("/analizar-hoja")
async def options_analizar_hoja():
    return JSONResponse(content={"message": "OK"})
