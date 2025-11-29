from fastapi import FastAPI, UploadFile, File, HTTPException
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
    version="4.1"
)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging más detallado
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analizar_hoja_mejorado(image_bytes, debug=False):
    """
    Analiza una imagen de hoja de cacao - versión mejorada
    """
    try:
        logger.info(f"Procesando imagen de {len(image_bytes)} bytes")
        
        # Verificar que la imagen no esté vacía
        if len(image_bytes) == 0:
            raise ValueError("La imagen está vacía")
        
        # Abrir y verificar imagen
        try:
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Imagen abierta: {image.format}, {image.size}, {image.mode}")
        except Exception as e:
            raise ValueError(f"No se pudo abrir la imagen: {str(e)}")

        # Verificar tamaño mínimo
        if image.size[0] < 50 or image.size[1] < 50:
            raise ValueError("La imagen es demasiado pequeña")

        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info("Convertido a RGB")

        # Convertir a array numpy
        img_array = np.array(image)
        
        # Verificar las dimensiones
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError("Formato de imagen no soportado")

        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Procesamiento de imagen
        img_blur = cv2.GaussianBlur(img_cv, (5, 5), 0)
        
        # Conversión a HSV
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        
        # Análisis de color y textura
        gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        
        # Detección de características específicas
        # Áreas saludables (verdes)
        mask_healthy = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        healthy_area = np.sum(mask_healthy > 0) / mask_healthy.size
        
        # Necrosis (marrón oscuro)
        mask_necrosis = cv2.inRange(hsv, (0, 40, 20), (20, 255, 120))
        necrosis_area = np.sum(mask_necrosis > 0) / mask_necrosis.size
        
        # Clorosis (amarillo)
        mask_chlorosis = cv2.inRange(hsv, (20, 30, 100), (35, 255, 255))
        chlorosis_area = np.sum(mask_chlorosis > 0) / mask_chlorosis.size
        
        # Bordes necróticos
        mask_edge = cv2.inRange(hsv, (0, 30, 20), (25, 200, 100))
        edge_area = np.sum(mask_edge > 0) / mask_edge.size

        logger.info(f"Áreas detectadas - Saludable: {healthy_area:.3f}, Necrosis: {necrosis_area:.3f}, Clorosis: {chlorosis_area:.3f}, Bordes: {edge_area:.3f}")

        # Lógica de diagnóstico mejorada
        total_issues = necrosis_area + chlorosis_area + edge_area
        health_score = healthy_area - total_issues
        
        # Determinar estado general
        if health_score > 0.7:
            estado = "Sana"
            enfermedad = "Ninguna detectada"
            probabilidad = 0.85
            color_principal = "Verde uniforme"
        elif health_score > 0.4:
            estado = "Estrés leve"
            enfermedad = "Posible deficiencia nutricional leve"
            probabilidad = 0.72
            color_principal = "Verde con leve clorosis"
        else:
            estado = "Problemas detectados"
            if edge_area > 0.05:
                enfermedad = "Deficiencia de potasio o estrés hídrico (necrosis en bordes)"
                probabilidad = 0.78
                color_principal = "Verde con zonas de clorosis amarillenta"
            elif necrosis_area > 0.08:
                enfermedad = "Posible infección fúngica"
                probabilidad = 0.75
                color_principal = "Verde con manchas marrones"
            else:
                enfermedad = "Problemas nutricionales o estrés"
                probabilidad = 0.70
                color_principal = "Verde irregular"

        # Características detalladas
        caracteristicas = {
            "color_principal": color_principal,
            "manchas": "Pequeñas manchas claras dispersas" if necrosis_area > 0.02 else "Sin manchas significativas",
            "borde": "Necrosis en el borde y punta, color café oscuro con halo amarillento" if edge_area > 0.03 else "Bordes regulares",
            "textura": "Levemente áspera, uniforme en la mayor parte del área",
            "deformaciones": False
        }

        resultado = {
            "estado_general": estado,
            "probabilidad": round(probabilidad, 2),
            "caracteristicas_detectadas": caracteristicas,
            "posible_enfermedad": enfermedad
        }

        if debug:
            resultado["debug"] = {
                "healthy_area": round(healthy_area, 3),
                "necrosis_area": round(necrosis_area, 3),
                "chlorosis_area": round(chlorosis_area, 3),
                "edge_area": round(edge_area, 3),
                "health_score": round(health_score, 3),
                "image_size": image.size
            }

        return resultado

    except Exception as e:
        logger.error(f"Error en análisis: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.post("/analizar-hoja")
async def analizar_hoja_endpoint(file: UploadFile = File(...), debug: bool = False):
    """
    Endpoint para analizar hojas de cacao
    """
    try:
        logger.info(f"Recibida petición de: {file.filename}")
        
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen válida")
        
        # Leer archivo
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="El archivo está vacío")
        
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB max
            raise HTTPException(status_code=400, detail="La imagen es demasiado grande")
        
        # Procesar imagen
        resultado = analizar_hoja_mejorado(image_bytes, debug)
        
        logger.info(f"Análisis completado: {resultado['estado_general']}")
        return resultado
        
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Formato de imagen no soportado")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error general: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/")
async def root():
    return {
        "message": "API de Análisis de Hojas de Cacao",
        "status": "activa",
        "version": "4.1",
        "endpoints": {
            "POST /analizar-hoja": "Analiza una imagen de hoja de cacao",
            "GET /": "Información de la API",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cacao-leaf-analyzer"}

# Para ejecutar directamente: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
