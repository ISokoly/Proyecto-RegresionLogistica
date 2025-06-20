import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os
import numpy as np

# Crear 50 registros con variabilidad controlada
np.random.seed(42)
n_samples = 50

df = pd.DataFrame({
    'promedio_util_tc': np.random.uniform(0, 1.5, n_samples),
    'max_atraso_6m': np.random.randint(0, 5, n_samples),
    'deuda_mes_vs_max12m': np.random.uniform(0, 2, n_samples),
    'num_decrementos': np.random.randint(0, 6, n_samples),
    'tipo_ingresos': np.random.randint(0, 3, n_samples),  # 0 = informal, 1 = formal, 2 = independiente
    'num_incrementos_efectivo': np.random.randint(0, 4, n_samples),
    'max_calificacion': np.random.randint(0, 6, n_samples),  # 0 = excelente, 5 = muy mala
    'prop_deuda_revolvente': np.random.uniform(0, 1, n_samples),
})

# Score ponderado ficticio para simular riesgo de default
score = (
    0.25 * df['deuda_mes_vs_max12m'] +
    0.2 * df['max_atraso_6m'] +
    0.2 * df['promedio_util_tc'] +
    0.1 * df['num_decrementos'] +
    0.05 * df['tipo_ingresos'] +
    0.05 * df['num_incrementos_efectivo'] +
    0.1 * df['max_calificacion'] +
    0.05 * df['prop_deuda_revolvente']
)

# Etiqueta de default basada en si el score está por encima del promedio
df['default'] = (score > score.mean()).astype(int)

# Entrenar el modelo
x = df.drop('default', axis=1)
y = df['default']

model = LogisticRegression()
model.fit(x, y)

# Guardar modelo
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/credit_model.pkl')
print("✅ Modelo guardado correctamente en 'model/credit_model.pkl'")

# Guardar Excel para usarlo desde el frontend
df.to_excel("data_for_demo.xlsx", index=False)
print("📊 Datos guardados en 'data_for_demo.xlsx' para pruebas o recomendaciones")
