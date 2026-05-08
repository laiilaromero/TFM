import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model
from pathlib import Path

#--------------------------------------------
#Ingenrir datos IFC + Epd
# -----------------------------------------
ROOT= Path(__file__).resolve().parents[1]
RES_DIR =ROOT/"outputs"/"csv"
RES_DIR2 = ROOT/"outputs"/"resultados"
df_ifc= pd.read_csv(RES_DIR/"df_ifc_clean.csv", sep=";")
epd_subset = pd.read_pickle(RES_DIR2/"epd_subset_limpio.pkl")
#Inspeccionar IFC-------------------
print(df_ifc.columns)
print(df_ifc.head())
print(df_ifc["material_name"].value_counts(dropna=False))
#Normalizar material_name
df_ifc["material_norm"] = "concrete"
print(df_ifc["material_norm"].value_counts())
#Normalizar EPD-------------------------
print(epd_subset.columns.to_list())
print(epd_subset.head())
epd_subset["material_norm"]="concrete"
epd_final = epd_subset[epd_subset["Module"].isin(["A1","A2","A3"])].copy()
print(epd_final.head())
#----------Merge IFC+EPD------------------------------
df_real = df_ifc.merge(
    epd_final[["material_norm", "GWP"]],
    on="material_norm",
    how="left"
)
#target
df_real["GWP_total"]= df_real["volume_m3"]* df_real["GWP"]
print(df_real.head())
features = ["type", "volume_m3", "area_m2", "Espesor_mm"]
target = "GWP_total"
df_listo = df_real[features + [target]].copy()
df_listo = df_listo[
    (df_listo["volume_m3"]> 0) &
    (df_listo["GWP_total"].notna())
].copy()
print(df_listo.head())
print(df_listo.info())
#---------------------------------------------------
#  Generar datos sintéticos para entrenamiento
# -------------------------------------------------
N_SYNTH = 500
df_synth = df_listo.sample(
    n= N_SYNTH,
    replace=True,
    random_state=42
).reset_index(drop=True)
np.random.seed(42)
for col in ["volume_m3", "area_m2"]:
    df_synth[col] *=np.random.normal(
        loc=1.0,
        scale=0.05,
        size=len(df_synth)
    )
    df_synth[col]=df_synth[col].clip(lower=0.001)
# distinguir real de sintetico y guardarlos-------
df_listo["source"]="real"
df_synth["source"]="synthetic"
df_listo.to_pickle(RES_DIR2/"df_real.pkl")
df_synth.to_pickle(RES_DIR2/"df_synth.pkl")
df_mix = pd.concat([df_listo, df_synth], ignore_index=True)
df_mix.to_pickle(RES_DIR2/"df_mix.pkl")
print(df_mix.info())
# Comparación estadistica----------------
stats = pd.DataFrame({
    "real_mean": df_listo[["volume_m3","area_m2"]].mean(),
    "synth_mean": df_synth[["volume_m3","area_m2"]].mean(),
    "real_std": df_listo[["volume_m3","area_m2"]].std(),
    "synth_std": df_synth[["volume_m3","area_m2"]].std()
})
print(stats.head())
#-------- Graficos-------------------------
IMG_DIR = ROOT/"outputs"/"imágen"
plt.figure(figsize=(7,5))

plt.scatter(
    df_listo["volume_m3"],
    df_listo["GWP_total"],
    alpha=0.7,
    label="Real"
)

plt.scatter(
    df_synth["volume_m3"],
    df_synth["GWP_total"],
    alpha=0.4,
    label="Synthetic"
)

plt.xlabel("Volume (m³)")
plt.ylabel("GWP (kg CO₂ eq)")
plt.legend()
plt.title("Distribución Real vs Sintética")
plt.grid(alpha=0.3)
plt.savefig(IMG_DIR/"Distribución Real vs Sintética", dpi=300)
plt.show()
