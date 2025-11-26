import ifcopenshell
import ifcopenshell.geom
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
import pandas as pd
import os
import joblib
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import joblib
from pathlib import Path
# # -----------------------------------------
# # Leer Dataframe de EDA----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RES_DIR =ROOT/"outputs"/"resultados"
df_ifc = pd.read_pickle(RES_DIR/"df_ifc_limpio.pkl")
epd_subset = pd.read_pickle(RES_DIR/"epd_subset_limpio.pkl")
epd_subset.loc[epd_subset["Name (en)"].str.contains("1 Ready-mixed concrete mixtures: Multibeton R30C3D16S4XC2XC1", na=False), "material_norm"] = "concrete"

#------------------------Unir Tablas-------------------------------------------
df_merged = df_ifc.merge(epd_subset, on="material_norm", how="left")
df_merged2= df_merged[df_merged["Module"].isin(["A1","A2","A3"])].copy()
#print(df_merged2.head(10))
# #Calcular el C02 de Concrete-------------------------------------------------
df_merged2["CO2_Kg"]= df_merged2["volume_m3"]*df_merged2["GWP"]
df_total=df_merged2.pivot_table(
    index="type",
    columns="Module",
    values="CO2_Kg",
    aggfunc="sum"
)
df_total =df_total.rename(columns={"A1":"A1_CO2","A2":"A2_CO2","A3":"A3_CO2"})
df_total["total_A1A3_CO2"]=df_total["A1_CO2"]+ df_total["A2_CO2"] + df_total["A3_CO2"]
print(df_total.head(10))

# # ------------------------- Cargar modelos entrenados -------------------------
MODEL_DIR =ROOT/"outputs"/"modelos"
rf = joblib.load(MODEL_DIR/"rf_model.pkl")
rl = joblib.load(MODEL_DIR/"lr_model.pkl")
scaler = joblib.load(MODEL_DIR/"scaler.pkl")
nn_model = load_model(MODEL_DIR/"nn_model.h5", compile=False)
nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# # ------------------------- Cargar columnas del training ---------------------
with open(MODEL_DIR/"train_columns.json","r") as f:
    train_cols = json.load(f)

# # ------------------------- Recalcular CO2 en df_merged2 ----------------------
A1 = df_merged2[df_merged2["Module"] == "A1"]["GWP"].unique()[0]
A2 = df_merged2[df_merged2["Module"] == "A2"]["GWP"].unique()[0]
A3 = df_merged2[df_merged2["Module"] == "A3"]["GWP"].unique()[0]

df_merged2["A1_CO2"] = df_merged2["volume_m3"] * A1
df_merged2["A2_CO2"] = df_merged2["volume_m3"] * A2
df_merged2["A3_CO2"] = df_merged2["volume_m3"] * A3
df_merged2["total_A1A3_CO2"] = df_merged2["A1_CO2"] + df_merged2["A2_CO2"] + df_merged2["A3_CO2"]

# # ------------------------- Preparar dataset real ----------------------------
X_real = df_merged2[["type","Espesor_mm","area_m2","volume_m3"]].copy()
X_real_encoded = pd.get_dummies(X_real, columns=["type"])
X_real_encoded = X_real_encoded.reindex(columns=train_cols, fill_value=0)
y_real = df_merged2[['A1_CO2','A2_CO2','A3_CO2','total_A1A3_CO2']]

# # ------------------------- Predicciones iniciales --------------------------
# # Random Forest
y_pred_rf = rf.predict(X_real_encoded)
# Linear Regression
y_pred_rl = rl.predict(X_real_encoded)
# Neural Network (escalar)
X_real_scaled = scaler.transform(X_real_encoded)
y_pred_nn = nn_model.predict(X_real_scaled)

# # ------------------------- Métricas iniciales ------------------------------
print("-----Resultados Iniciales Real Data-----------")
print("RF MAE:", mean_absolute_error(y_real, y_pred_rf))
print("RF R2:", r2_score(y_real, y_pred_rf))
print("LR MAE:", mean_absolute_error(y_real, y_pred_rl))
print("LR R2 :", r2_score(y_real, y_pred_rl))
print("NN MAE:", mean_absolute_error(y_real, y_pred_nn))
print("NN R2 :", r2_score(y_real, y_pred_nn))

# # ------------------------- Fine-tuning de la NN ----------------------------
# # Ajustar la NN con los datos reales 
nn_model.fit(
    X_real_scaled,
    y_real,
    epochs=50,      
    batch_size=8,
    verbose=1
)

# Guardar modelo fine-tuned
nn_model.save(MODEL_DIR/"nn_model_finetuned.h5")

# ------------------------- Predicciones después de fine-tuning -------------
y_pred_nn_finetuned = nn_model.predict(X_real_scaled)

print("-----NN Fine-Tuned Real Data-----------")
print("NN MAE (finetuned):", mean_absolute_error(y_real, y_pred_nn_finetuned))
print("NN R2 (finetuned):", r2_score(y_real, y_pred_nn_finetuned))
#----------------Representacion Gráfica--------------------------------------------
import matplotlib.pyplot as plt
IMG_DIR= ROOT/"outputs"/"imágen"

plt.figure(figsize=(8,6))
plt.scatter(y_real['total_A1A3_CO2'], y_pred_rf[:,3], alpha=0.7, color='blue', label='Random Forest')
plt.scatter(y_real['total_A1A3_CO2'], y_pred_rl[:,3], alpha=0.7, color='green', label='Regresión Lineal')
plt.scatter(y_real['total_A1A3_CO2'], y_pred_nn_finetuned[:,3], alpha=0.7, color='red', label='NN Fine-Tuned')
plt.plot([y_real['total_A1A3_CO2'].min(), y_real['total_A1A3_CO2'].max()],
         [y_real['total_A1A3_CO2'].min(), y_real['total_A1A3_CO2'].max()],
         color='black', linestyle='--', label='Perfecta Predicción')
plt.xlabel('GWP real [kgCO2]')
plt.ylabel('GWP predicho [kgCO2]')
plt.title('Comparación Predicción vs Valores Reales - Total A1-A3')
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR/'pred_vs_real.png', dpi=300)
plt.show()
