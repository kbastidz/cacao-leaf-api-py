from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import random

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

def analizar_hoja(image_bytes):
    """
    Analiza una imagen de hoja de cacao y detecta posibles enfermedades
    """
    # Convertir imagen a formato OpenCV
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Reducir ruido
    img_blur = cv2.GaussianBlur(img_cv, (5, 5), 0)
    
    # Calcular color promedio
    mean_color = cv2.mean(img_blur)[:3]
    r, g, b = mean_color
    
    # Convertir a HSV (para analizar tono general)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    h, s, v, _ = cv2.mean(hsv)
    
    # Detectar bordes y textura
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    texture_score = np.mean(edges) / 255  # valor entre 0 y 1
    
    # Detección de manchas (por color oscuro)
    mask_spots = cv2.inRange(hsv, (0, 30, 0), (180, 255, 80))
    spot_area = np.sum(mask_spots > 0) / mask_spots.size
    
    # Heurística básica
    color_principal = ""
    estado_general = ""
    posible_enfermedad = ""
    probabilidad = round(random.uniform(0.8, 0.95), 2)  # valor simulado
    
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
        "deformaciones": texture_score > 0.3
    }
    
    # --- Respuesta final ---
    resultado = {
        "estado_general": estado_general,
        "probabilidad": probabilidad,
        "caracteristicas_detectadas": caracteristicas,
        "posible_enfermedad": posible_enfermedad
    }
    
    return resultado

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
            "GET /": "Información de la API"
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
    try:
        # Validar que sea una imagen
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                content={"error": "El archivo debe ser una imagen"},
                status_code=400
            )
        
        # Leer la imagen
        image_bytes = await file.read()
        
        # Analizar la hoja
        resultado = analizar_hoja(image_bytes)
        
        return JSONResponse(content=resultado)
    
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error al procesar la imagen: {str(e)}"},
            status_code=500
        )

@app.get("/health")
async def health_check():
    """
    Health check para verificar que la API está funcionando
    """
    return {"status": "healthy", "service": "cacao-api"}
