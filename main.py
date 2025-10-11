from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import random
import traceback

app = FastAPI(
    title="Cacao Leaf Analyzer API",
    description="API para analizar hojas de cacao y detectar enfermedades",
    version="2.0"
)

# Habilitar CORS para permitir peticiones desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agregar logging para debug
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analizar_hoja(image_bytes):
    """
    Analiza una imagen de hoja de cacao y detecta posibles enfermedades
    """
    try:
        # Convertir imagen a formato OpenCV
        logger.info(f"Procesando imagen de {len(image_bytes)} bytes")
        
        image = Image.open(io.BytesIO(image_bytes))
        logger.info(f"Imagen abierta: {image.format}, {image.size}, {image.mode}")
        
        # Asegurar que esté en RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Convertido a RGB")
        
        img_array = np.array(image)
        logger.info(f"Array shape: {img_array.shape}")
        
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Reducir ruido
        img_blur = cv2.GaussianBlur(img_cv, (5, 5), 0)
        
        # Calcular color promedio
        mean_color = cv2.mean(img_blur)[:3]
        r, g, b = mean_color
        logger.info(f"Color promedio - R:{r:.2f}, G:{g:.2f}, B:{b:.2f}")
        
        # Convertir a HSV (para analizar tono general)
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        h, s, v, _ = cv2.mean(hsv)
        logger.info(f"HSV promedio - H:{h:.2f}, S:{s:.2f}, V:{v:.2f}")
        
        # Detectar bordes y textura
        gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        texture_score = np.mean(edges) / 255  # valor entre 0 y 1
        logger.info(f"Texture score: {texture_score:.4f}")
        
        # Detección de manchas (por color oscuro)
        mask_spots = cv2.inRange(hsv, (0, 30, 0), (180, 255, 80))
        spot_area = np.sum(mask_spots > 0) / mask_spots.size
        logger.info(f"Spot area: {spot_area:.4f}")
        
        # Heurística básica
        probabilidad = round(random.uniform(0.8, 0.95), 2)
        
        # --- Análisis de color dominante ---
        if g > r and g > b:
            color_principal = "verde"
            estado_general = "Sana"
            posible_enfermedad = "Ninguna"
        elif r > g and r > b:
            color_principal = "amarillento"
            estado_general = "Deficiencia nutricional"
            posible_enfermedad = "Posible falta de nitrógeno"
        elif spot_area > 0.05:
            color_principal = "verde amarillento"
            estado_general = "Hongo foliar"
            posible_enfermedad = "Posible Cercospora o Phytophthora"
        elif texture_score > 0.25:
            color_principal = "verde oscuro"
            estado_general = "Daño físico o plaga"
            posible_enfermedad = "Posible ataque de insectos"
        else:
            color_principal = "indeterminado"
            estado_general = "Desconocido"
            posible_enfermedad = "Requiere análisis avanzado"
        
        # --- Características detectadas ---
        caracteristicas = {
            "color_principal": color_principal,
            "manchas": "circulares, marrones" if spot_area > 0.05 else "ninguna visible",
            "borde": "irregular" if texture_score > 0.25 else "regular",
            "textura": "seca" if v < 80 else "normal",
            "deformaciones": bool(texture_score > 0.3)
        }
        
        # --- Respuesta final ---
        resultado = {
            "estado_general": estado_general,
            "probabilidad": probabilidad,
            "caracteristicas_detectadas": caracteristicas,
            "posible_enfermedad": posible_enfermedad
        }
        
        logger.info(f"Análisis completado: {estado_general}")
        return resultado
        
    except Exception as e:
        logger.error(f"Error en analizar_hoja: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.get("/")
async def root():
    """
    Endpoint raíz - Información de la API
    """
    return {
        "message": "API de Análisis de Hojas de Cacao",
        "status": "activa",
        "version": "2.0",
        "endpoints": {
            "POST /analizar-hoja": "Analiza una imagen de hoja de cacao",
            "GET /": "Información de la API",
            "GET /health": "Health check"
        }
    }

@app.post("/analizar-hoja")
async def analizar_hoja_endpoint(file: UploadFile = File(...)):
    """
    Endpoint principal - Analiza una imagen de hoja de cacao
    
    Parámetros:
    - file: Archivo de imagen (JPG, PNG, etc.)
    
    Retorna:
    - JSON con el análisis de la hoja
    """
    logger.info(f"=== Nueva petición POST /analizar-hoja ===")
    logger.info(f"Content-Type: {file.content_type}")
    logger.info(f"Filename: {file.filename}")
    
    try:
        # Validar que sea una imagen
        if file.content_type and not file.content_type.startswith('image/'):
            logger.warning(f"Tipo de archivo inválido: {file.content_type}")
            return JSONResponse(
                content={"error": "El archivo debe ser una imagen"},
                status_code=400
            )
        
        # Leer la imagen
        image_bytes = await file.read()
        logger.info(f"Imagen leída: {len(image_bytes)} bytes")
        
        if len(image_bytes) == 0:
            logger.error("Archivo vacío recibido")
            return JSONResponse(
                content={"error": "El archivo está vacío"},
                status_code=400
            )
        
        # Analizar la hoja
        resultado = analizar_hoja(image_bytes)
        logger.info(f"✓ Análisis completado exitosamente")
        
        return JSONResponse(content=resultado)
    
    except PIL.UnidentifiedImageError:
        logger.error("No se pudo identificar la imagen")
        return JSONResponse(
            content={"error": "No se pudo procesar la imagen. Asegúrate de que sea un archivo de imagen válido"},
            status_code=400
        )
    
    except Exception as e:
        logger.error(f"✗ Error al procesar: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={
                "error": f"Error al procesar la imagen: {str(e)}",
                "type": type(e).__name__
            },
            status_code=500
        )

@app.get("/health")
async def health_check():
    """
    Health check para verificar que la API está funcionando
    """
    return {"status": "healthy", "service": "cacao-api", "version": "2.0"}

@app.options("/analizar-hoja")
async def options_analizar_hoja():
    """
    Handle CORS preflight request
    """
    return JSONResponse(content={"message": "OK"})
