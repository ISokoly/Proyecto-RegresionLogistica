import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime

n_samples = 100
df = pd.DataFrame({
    'promedio_util_tc': np.random.uniform(0, 1.5, n_samples),
    'max_atraso_6m': np.random.randint(0, 5, n_samples),
    'deuda_mes_vs_max12m': np.random.uniform(0, 2, n_samples),
    'num_decrementos': np.random.randint(0, 6, n_samples),
    'tipo_ingresos': np.random.randint(0, 3, n_samples),
    'num_incrementos_efectivo': np.random.randint(0, 4, n_samples),
    'max_calificacion': np.random.randint(0, 6, n_samples),
    'prop_deuda_revolvente': np.random.uniform(0, 1, n_samples),
})
df['default'] = np.random.randint(0, 2, n_samples)

X = df.drop('default', axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print(f"✅ Entrenamiento (80%): {len(X_train)} muestras → Precisión: {train_acc:.2f}")
print(f"✅ Prueba (20%): {len(X_test)} muestras → Precisión: {test_acc:.2f}")

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/credit_model.pkl')
print("✅ Modelo guardado en 'model/credit_model.pkl'")

new_samples = 20
new_data = pd.DataFrame({
    'promedio_util_tc': np.random.uniform(0, 1.5, new_samples),
    'max_atraso_6m': np.random.randint(0, 5, new_samples),
    'deuda_mes_vs_max12m': np.random.uniform(0, 2, new_samples),
    'num_decrementos': np.random.randint(0, 6, new_samples),
    'tipo_ingresos': np.random.randint(0, 3, new_samples),
    'num_incrementos_efectivo': np.random.randint(0, 4, new_samples),
    'max_calificacion': np.random.randint(0, 6, new_samples),
    'prop_deuda_revolvente': np.random.uniform(0, 1, new_samples),
})

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"data_for_demo_{timestamp}.xlsx"
new_data.to_excel(filename, index=False)
print(f"📊 Nuevos datos guardados en '{filename}'")