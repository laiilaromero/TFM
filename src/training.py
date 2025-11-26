import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model
from pathlib import Path


# -----------------------------------------
#  Generar datos sintéticos para entrenamiento
# -----------------------------------------
EPD_factors = {
    "A1": 120,   # kgCO2 / m3
    "A2": 15,
    "A3": 40
}

N = 50
volumen = np.random.uniform(0.1, 400, N)
espesor_mm = np.random.uniform(50, 400, N)
area = volumen / (espesor_mm/1000 + 1e-9)
types = np.random.choice(["IfcSlab","IfcWall","IfcBeam","IfcColumn"], size=N, p=[0.4,0.2,0.2,0.2])

def cal_c02(vol, factors=EPD_factors):
    a1 = factors["A1"]*vol
    a2 = factors["A2"]*vol
    a3 = factors["A3"]*vol
    return {"A1":a1, "A2":a2, "A3":a3, "total":a1+a2+a3}

rows = []
for i in range(N):
    co2 = cal_c02(volumen[i])
    rows.append({
        "type": types[i],
        "material_norm": "concrete",
        "Espesor_mm": espesor_mm[i],
        "area_m2": area[i],
        "volume_m3": volumen[i],
        "A1_CO2": co2["A1"],
        "A2_CO2": co2["A2"],
        "A3_CO2": co2["A3"],
        "total_A1A3_CO2": co2["total"]
    })

df_sintetico = pd.DataFrame(rows)

# -----------------------------------------
# Preparar datos para entrenamiento
# -----------------------------------------
X = df_sintetico[['type','Espesor_mm','area_m2','volume_m3']]
y = df_sintetico[['A1_CO2','A2_CO2','A3_CO2','total_A1A3_CO2']]
X_encoded = pd.get_dummies(X, columns=["type"])
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Escalado para redes neuronales
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# Entrenar modelos
# -----------------------------------------
# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Linear Regression
rl = LinearRegression()
rl.fit(X_train, y_train)

# NN
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(4, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=8, verbose=1)

# -----------------------------------------
#  Guardar modelos entrenados
# -----------------------------------------
ROOT= Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT/"outputs"/"modelos"

joblib.dump(rf, MODEL_DIR / "rf_model.pkl")
joblib.dump(rl, MODEL_DIR / "lr_model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

with open(MODEL_DIR / "train_columns.json", "w") as f:
    json.dump(X_encoded.columns.tolist(), f)

model.save(MODEL_DIR / "nn_model.h5")

