from pathlib import Path
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Cargar dataset Real Y Mix------------------------------------------------
ROOT= Path(__file__).resolve().parents[1]
DATASETS_ROOT= ROOT/"outputs"/"resultados"
MODEL_DIR=ROOT/"outputs"/"modelos"
df_real = pd.read_pickle(DATASETS_ROOT/"df_real.pkl")
df_mix = pd.read_pickle(DATASETS_ROOT/"df_mix.pkl")
# print(df_mix.dtypes)
# print(df_real.dtypes)
#print(df_real["Espesor_mm"].isna().mean())
#--------------Hipotesis 1- Datos Reales---------------------------------------------------------
df_h1= df_real.copy()
#Elección de entrada y salida
X1 = df_h1[["type", "volume_m3", "area_m2"]]
y1 = df_h1["GWP"]
#one-hot encoding para type
X1 = pd.get_dummies(X1, columns=["type"], drop_first=True)
#Separamos datos para entrenar y evaluar
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1, test_size=0.2, random_state=42)
#---------------Hipotesis 2 - Datos R+ Datos S ----------------------------------------------------
df_h2=df_mix.copy()
X2 = df_h2[["type", "volume_m3", "area_m2"]]
y2 = df_h2["GWP"]
X2 = pd.get_dummies(X2, columns=["type"], drop_first=True)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
joblib.dump(X1_train.columns, MODEL_DIR / "columns_real.pkl")
joblib.dump(X2_train.columns, MODEL_DIR / "columns_mix.pkl")

#----------------Entrenar Modelos-------------------------------------------------------------
rl_h1= LinearRegression()
rl_h1.fit(X1_train,y1_train)
joblib.dump(rl_h1,MODEL_DIR/"rl_real.pkl")
rl_h2 = LinearRegression()
rl_h2.fit(X2_train,y2_train)
joblib.dump(rl_h2,MODEL_DIR/"rl_synth.pkl")
#---RandomF-------------------
rf_h1=RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
rf_h1.fit(X1_train,y1_train)
joblib.dump(rf_h1,MODEL_DIR/"rf_real.pkl")
rf_h2=RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
rf_h2.fit(X2_train,y2_train)
joblib.dump(rf_h2,MODEL_DIR/"rf_synth.pkl")
#----Scaler-------------------------
scaler_h1 = StandardScaler()
X1_train_scaled = scaler_h1.fit_transform(X1_train)
X1_test_scaled = scaler_h1.transform(X1_test)
joblib.dump(scaler_h1, MODEL_DIR / "scaler_real.pkl")
scaler_h2 = StandardScaler()
X2_train_scaled = scaler_h2.fit_transform(X2_train)
X2_test_scaled = scaler_h2.transform(X2_test)
joblib.dump(scaler_h2, MODEL_DIR / "scaler_synth.pkl")
#-------------RN-------------------------------
model_h1= Sequential([
    Input(shape=(X1_train_scaled.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])
model_h1.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
model_h1.fit(
    X1_train_scaled,
    y1_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=0
)
model_h1.save(MODEL_DIR/"nn_real.keras")
model_h2= Sequential([
    Input(shape=(X2_train_scaled.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])
model_h2.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
model_h2.fit(
    X2_train_scaled,
    y2_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=0
)
model_h2.save(MODEL_DIR/"nn_synth.keras")
