from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from io import BytesIO
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

credit_model = None
try:
    credit_model = joblib.load("model/credit_model.pkl")
except FileNotFoundError:
    print("⚠️ Advertencia: El modelo credit_model.pkl no se encontró. Ruta: model/credit_model.pkl")

required_cols = [
    'promedio_util_tc', 'max_atraso_6m', 'deuda_mes_vs_max12m',
    'num_decrementos', 'tipo_ingresos', 'num_incrementos_efectivo',
    'max_calificacion', 'prop_deuda_revolvente'
]

@app.post("/predict_excel")
async def predict_excel(file: UploadFile = File(...)):
    print(f"Archivo recibido: {file.filename}")

    if credit_model is None:
        raise HTTPException(status_code=503, detail="Modelo de regresión no cargado")

    if not file.filename.lower().endswith(('.xls', '.xlsx')):
        print("Archivo no es Excel")
        raise HTTPException(status_code=400, detail="Archivo no es Excel")

    contents = await file.read()
    try:
        df = pd.read_excel(BytesIO(contents))
        df.columns = df.columns.str.strip().str.lower()
        print("Columnas leídas del Excel (normalizadas):", df.columns.tolist())
    except Exception as e:
        print("Error leyendo archivo Excel:", e)
        raise HTTPException(status_code=400, detail="Error leyendo archivo Excel")

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Faltan columnas: {missing}")
        raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")

    try:
        probs = credit_model.predict_proba(df[required_cols])[:, 1]
        
        recom_labels = np.array(["✅ Recomendable", "⚠️ Riesgo medio", "❌ No recomendable"])
        porc = probs * 100
        indices = np.floor(porc / 30).astype(int)
        indices = np.clip(indices, 0, 2)
        recomendaciones = recom_labels[indices]

        results = [
            {
                "index": i,
                "probabilidad_default": round(float(p), 4),
                "recomendacion": rec
            }
            for i, (p, rec) in enumerate(zip(probs, recomendaciones))
        ]
    except Exception as e:
        print("Error al predecir:", e)
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")

    return {"predicciones": results}

@app.get("/")
def home():
    return {
        "mensaje": "API lista",
        "endpoints": [
            "/predict_excel → predicción de riesgo con Excel"
        ]
    }
