from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import traceback
import logging

app = FastAPI(
    title="Cacao Leaf Analyzer API",
    description="API para analizar hojas de cacao y detectar deficiencias nutricionales",
    version="5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# 🔍 FUNCIÓN PRINCIPAL DE ANÁLISIS
# ----------------------------------------------------------------
def analizar_hoja_mejorado(image_bytes, debug=False):

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = np.array(image)
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Filtros suaves
        blur = cv2.GaussianBlur(img_cv, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # ---------------------------------------------------------
        # 🎯 DETECCIÓN POR COLORES (HSV)
        # ---------------------------------------------------------
        mask_green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        mask_yellow = cv2.inRange(hsv, (20, 30, 80), (35, 255, 255))
        mask_brown = cv2.inRange(hsv, (5, 30, 20), (25, 200, 150))
        mask_dark_brown = cv2.inRange(hsv, (0, 20, 0), (20, 255, 80))

        mask_purple = cv2.inRange(hsv, (125, 20, 20), (155, 255, 255))  # fósforo avanzado

        # Áreas normalizadas
        total_px = mask_green.size
        green_area = np.sum(mask_green > 0) / total_px
        yellow_area = np.sum(mask_yellow > 0) / total_px
        brown_area = np.sum(mask_brown > 0) / total_px
        darkbrown_area = np.sum(mask_dark_brown > 0) / total_px
        purple_area = np.sum(mask_purple > 0) / total_px

        # ---------------------------------------------------------
        # 🧠 CLASIFICACIÓN POR REGLAS
        # ---------------------------------------------------------

        # 🟢 1. HOJA SANA
        if green_area > 0.75 and yellow_area < 0.05 and brown_area < 0.05:
            return generar_respuesta(
                estado="Sana",
                prob=0.90,
                color="Verde uniforme",
                manchas="Sin manchas",
                borde="Regular",
                textura="Normal",
                deform=False,
                enfermedad="Ninguna"
            )

        # 🟡 2. DEFICIENCIA DE NITRÓGENO
        if yellow_area > 0.35 and green_area < 0.40:
            return generar_respuesta(
                estado="Deficiencia nutricional",
                prob=0.88,
                color="Verde pálido a amarillo uniforme",
                manchas="Sin manchas marcadas",
                borde="Regular",
                textura="Suave",
                deform=False,
                enfermedad="Deficiencia de Nitrógeno (N)"
            )

        # 🟣 3. DEFICIENCIA DE FÓSFORO
        if purple_area > 0.03 or (green_area < 0.50 and brown_area < 0.10 and yellow_area < 0.10):
            return generar_respuesta(
                estado="Problema nutricional",
                prob=0.80,
                color="Verde oscuro-opaco con tonos bronce/púrpura",
                manchas="Manchas oscuras leves",
                borde="Oscurecido",
                textura="Más rígida",
                deform=False,
                enfermedad="Deficiencia de Fósforo (P)"
            )

        # 🔥 4. DEFICIENCIA DE POTASIO (K)
        if brown_area > 0.07 and yellow_area > 0.05:
            return generar_respuesta(
                estado="Problema nutricional severo",
                prob=0.91,
                color="Verde con bordes amarillos y necrosis café",
                manchas="Pequeñas manchas cloróticas",
                borde="Necrosis marginal (borde quemado)",
                textura="Áspera en el borde",
                deform=False,
                enfermedad="Deficiencia de Potasio (K)"
            )

        # 🌱 5. DEFICIENCIA DE MAGNESIO
        if yellow_area > 0.20 and green_area > 0.40:
            return generar_respuesta(
                estado="Deficiencia nutricional",
                prob=0.86,
                color="Clorosis internerval (venas verdes, fondo amarillo)",
                manchas="Leves",
                borde="Regular",
                textura="Normal",
                deform=False,
                enfermedad="Deficiencia de Magnesio (Mg)"
            )

        # 🌧 6. ESTRÉS HÍDRICO
        if darkbrown_area > 0.05 and yellow_area < 0.10:
            return generar_respuesta(
                estado="Estrés fisiológico",
                prob=0.78,
                color="Café seco",
                manchas="Manchas secas",
                borde="Seco",
                textura="Quebradiza",
                deform=False,
                enfermedad="Estrés hídrico"
            )

        # 🍂 7. INFECCIÓN FÚNGICA
        if brown_area > 0.12 and yellow_area < 0.05:
            return generar_respuesta(
                estado="Problema fitosanitario",
                prob=0.75,
                color="Manchas marrones",
                manchas="Lesiones irregulares",
                borde="Irregular",
                textura="Rugosa",
                deform=False,
                enfermedad="Infección fúngica"
            )

        # 🔘 Si nada coincide
        return generar_respuesta(
            estado="No concluyente",
            prob=0.55,
            color="Mixto",
            manchas="Irregulares",
            borde="Variable",
            textura="Variable",
            deform=False,
            enfermedad="No identificado"
        )

    except Exception as e:
        logger.error(str(e))
        raise


# ----------------------------------------------------------------
# 📝 GENERADOR DE RESPUESTA UNIFICADO
# ----------------------------------------------------------------
def generar_respuesta(estado, prob, color, manchas, borde, textura, deform, enfermedad):
    return {
        "estado_general": estado,
        "probabilidad": prob,
        "caracteristicas_detectadas": {
            "color_principal": color,
            "manchas": manchas,
            "borde": borde,
            "textura": textura,
            "deformaciones": deform
        },
        "posible_enfermedad": enfermedad
    }


# ----------------------------------------------------------------
# 📤 ENDPOINT
# ----------------------------------------------------------------
@app.post("/analizar-hoja")
async def analizar_hoja_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        return analizar_hoja_mejorado(image_bytes)
    except Exception:
        raise HTTPException(status_code=500, detail="Error procesando la imagen")


@app.get("/")
def root():
    return {"msg": "Cacao Leaf Analyzer v5.0", "status": "activo"}
