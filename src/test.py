import pandas as pd
from openpyxl import workbook
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import joblib
from joblib import load
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# # -----------------------------------------
# # Leer Dataframe----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RES_DIR =ROOT/"outputs"/"resultados"
df_real=pd.read_pickle(RES_DIR/"df_real.pkl")
df_synth=pd.read_pickle(RES_DIR/"df_synth.pkl")
#Cargar modelos-------------------------------------------------------------
MODEL_DIR=ROOT/"outputs"/"modelos"
rl_real=joblib.load(MODEL_DIR/"rl_real.pkl")
rl_synth=joblib.load(MODEL_DIR/"rl_synth.pkl")
rf_real=joblib.load(MODEL_DIR/"rf_real.pkl")
rf_synth=joblib.load(MODEL_DIR/"rf_synth.pkl")
scaler_real=joblib.load(MODEL_DIR/"scaler_real.pkl")
scaler_synth=joblib.load(MODEL_DIR/"scaler_synth.pkl")
nn_real=load_model(MODEL_DIR/"nn_real.keras")
nn_synth=load_model(MODEL_DIR/"nn_synth.keras")
cols_real=joblib.load(MODEL_DIR/"columns_real.pkl")
cols_mix=joblib.load(MODEL_DIR/"columns_mix.pkl")
#------Prepara datos de Test------------------------------------------------
X_real = df_real[["type", "volume_m3", "area_m2"]]
X_mix  = df_synth[["type", "volume_m3", "area_m2"]]
y_real = df_real["GWP_total"]
y_mix  = df_synth["GWP_total"]
#one-hot encoding
X_real = pd.get_dummies(X_real, columns=["type"], drop_first=True)
X_mix = pd.get_dummies(X_mix, columns=["type"], drop_first=True)
#Reindexar para igual columnas con el training
X_real = X_real.reindex(columns=cols_real, fill_value=0)
X_mix  = X_mix.reindex(columns=cols_mix, fill_value=0)
#Split de test
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_real,y_real, test_size=0.2, random_state=42)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(X_mix,y_mix, test_size=0.2, random_state=42)
Xr_test_scaled=scaler_real.transform(Xr_test)
Xm_test_scaled=scaler_synth.transform(Xm_test)
#Predicciones--------------------------
#RL
y_pred_rl_real = rl_real.predict(Xr_test)
y_pred_rl_synth = rl_synth.predict(Xm_test)
# Métricas REAL
mae_real  = mean_absolute_error(yr_test, y_pred_rl_real)
rmse_real = np.sqrt(mean_squared_error(yr_test, y_pred_rl_real))
r2_real   = r2_score(yr_test, y_pred_rl_real)
#Métrica Sintetico
mae_synth  = mean_absolute_error(ym_test, y_pred_rl_synth)
rmse_synth = np.sqrt(mean_squared_error(ym_test, y_pred_rl_synth))
r2_synth   = r2_score(ym_test, y_pred_rl_synth)

print("Linear Regression")
print("REAL:", mae_real, rmse_real, r2_real)
print("REAL + SYNTH:", mae_synth, rmse_synth, r2_synth)
# Random Forest
y_pred_rf_real  = rf_real.predict(Xr_test)
y_pred_rf_synth = rf_synth.predict(Xm_test)
# # Métricas REAL
mae_real_rf  = mean_absolute_error(yr_test, y_pred_rf_real)
rmse_real_rf = np.sqrt(mean_squared_error(yr_test, y_pred_rf_real))
r2_real_rf  = r2_score(yr_test, y_pred_rf_real)
#Métrica Sintetico
mae_synth_rf = mean_absolute_error(ym_test,y_pred_rf_synth)
rmse_synth_rf= np.sqrt(mean_squared_error(ym_test,y_pred_rf_synth))
r2_synth_rf = r2_score(ym_test, y_pred_rf_synth)
print("Random Forest")
print("REAL:", mae_real_rf, rmse_real_rf, r2_real_rf)
print("REAL + SYNTH:", mae_synth_rf, rmse_synth_rf, r2_synth_rf)

# # Redes Neuronales
y_pred_nn_real  = nn_real.predict(Xr_test_scaled).ravel()
y_pred_nn_synth = nn_synth.predict(Xm_test_scaled).ravel()
# # Métricas REAL
mae_real_nn  = mean_absolute_error(yr_test, y_pred_nn_real)
rmse_real_nn = np.sqrt(mean_squared_error(yr_test, y_pred_nn_real))
r2_real_nn  = r2_score(yr_test, y_pred_nn_real)
#Métrica Sintetico
mae_synth_nn = mean_absolute_error(ym_test,y_pred_nn_synth)
rmse_synth_nn= np.sqrt(mean_squared_error(ym_test, y_pred_nn_synth))
r2_synth_nn = r2_score(ym_test, y_pred_nn_synth)
print("Redes Neuronales ")
print("REAL:", mae_real_nn, rmse_real_nn, r2_real_nn)
print("REAL + SYNTH:", mae_synth_nn, rmse_synth_nn, r2_synth_nn)
#Graficos de valores por Modelo
#RL
# scenarios = ["Real", "Real+Sintético"]
# mae_values=[96.46,93.19]
# plt.figure(figsize=(7,7))
# plt.bar(scenarios,mae_values)
# plt.ylabel("MAE [kg CO₂ eq]")
# plt.title("Comparación MAE – Regresión Lineal")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()
#RF
# plt.figure(figsize=(12, 5))
# plt.subplot(1,2,1)
# plt.scatter(yr_test, y_pred_rf_real, alpha=0.7)
# plt.plot(
#     [yr_test.min(), yr_test.max()],
#     [yr_test.min(), yr_test.max()],
#     linestyle='--'
# )
# plt.xlabel("GWP real")
# plt.ylabel("GWP predicho")
# plt.title("Datos Reales")
# plt.subplot(1,2,1)
# plt.scatter(ym_test, y_pred_rf_synth, alpha=0.7)
# plt.plot(
#     [ym_test.min(), ym_test.max()],
#     [ym_test.min(), ym_test.max()],
#     linestyle='--'
# )
# plt.xlabel("GWP real")
# plt.ylabel("GWP predicho")
# plt.title("Datos Reales + sintéticos")

# plt.tight_layout()
# plt.show()
#RN
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.scatter(yr_test, y_pred_nn_real, alpha=0.7)
plt.plot(
    [yr_test.min(), yr_test.max()],
    [yr_test.min(), yr_test.max()],
    linestyle='--'
)
plt.xlabel("GWP real")
plt.ylabel("GWP predicho")
plt.title("Datos Reales")
plt.subplot(1,2,1)
plt.scatter(ym_test, y_pred_nn_synth, alpha=0.7)
plt.plot(
    [ym_test.min(), ym_test.max()],
    [ym_test.min(), ym_test.max()],
    linestyle='--'
)
plt.xlabel("GWP real")
plt.ylabel("GWP predicho")
plt.title("Datos Reales + sintéticos")

plt.tight_layout()
plt.show()
#Tabla final comparativa
import pandas as pd

# Crear diccionario con resultados
results = {
    "Modelo": [
        "Linear Regression",
        "Linear Regression",
        "Random Forest",
        "Random Forest",
        "Neural Network",
        "Neural Network"
    ],
    "Dataset": [
        "Real",
        "Real + Sintético",
        "Real",
        "Real + Sintético",
        "Real",
        "Real + Sintético"
    ],
    "MAE": [
        96.46,
        93.19,
        105.87,
        45.90,
        95.87,
        93.40
    ],
    "RMSE": [
        102.75,
        99.75,
        114.84,
        61.46,
        102.70,
        99.88
    ],
    "R2": [
        -0.02,
        0.02,
        -0.28,
        0.63,
        -0.02,
        0.02
    ]
}

# Crear DataFrame
df_results = pd.DataFrame(results)
df_results.to_excel(RES_DIR/"tabla de resultados.xlsx", engine="openpyxl")
