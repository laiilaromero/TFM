import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#--------------------Leer archivo-----------------------------------------------------
ROOT= Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT/"outputs"
CSV_DIR =OUTPUTS_DIR/"csv"
ifc_path = CSV_DIR/"ifc_target_elements.csv"
df_ifc = pd.read_csv(ifc_path, sep=";")
print(df_ifc.columns)
DATA_DIR = ROOT/"data"
epd_path = DATA_DIR/ "EPD2.csv"
epd = pd.read_csv(epd_path, sep=";", on_bad_lines="skip", encoding="latin-1") 
#print(epd.columns)
epd2_path =DATA_DIR/"EPD1.CSV"
epd2 = pd.read_csv(epd2_path, sep=";", on_bad_lines="skip", encoding="latin-1")
#print(epd2.head())
#-----------------------EDA----------------------------------------------------------
IMG_DIR= OUTPUTS_DIR/"imágen"
print(df_ifc.describe())
print(df_ifc.info())
print(df_ifc.isnull().sum())
df_ifc = df_ifc[df_ifc["volume_m3"] > 0].copy()
plt.figure(figsize=(8,6))
plt.hist(np.log10(df_ifc["area_m2"][df_ifc["area_m2"]>0]), bins=50)
plt.title(" Distribución log10 del área")
plt.xlabel ("area_m2")
plt.ylabel("frecuencia")
plt.tight_layout()
plt.savefig(IMG_DIR/"area_hist.png")
plt.show()
plt.figure(figsize=(8,6))
plt.hist(np.log10(df_ifc["volume_m3"]), bins=30)
plt.title("Distribución de volumen (log10)")
plt.xlabel("log10(volume_m3)")
plt.ylabel("frecuencia")
plt.tight_layout()
plt.savefig(IMG_DIR/"volume_hist_log.png")
plt.show()
#print(df_ifc["volume_m3"].min(), df_ifc["volume_m3"].max(), df_ifc["volume_m3"].quantile([0.5,0.9,0.99]))
#Mapear los nombres de materiales
df_ifc["material_norm"] = None
# # Diccionario para mapear IFC -> EPD---------------------------------------------------------------------
df_ifc.loc[df_ifc["material_name"].str.contains("concrete", case=False, na=False), "material_norm"] = "concrete"
# df_ifc.loc[df["material_name"].str.contains("Metal - Decking", na=False), "material_norm"] = "Steel sheets"
# df_ifc.loc[df["material_name"].str.contains("Structure - Steel Bar Joist Layer", na=False), "material_norm"] = "Steel reinforcing bar"
###---------------------------------EPD--------------------------------------------------------------------------------
# Columnas de interés
print(epd.describe())
print(epd2.describe())
# seleccionar columnas relevantes y renombrar
epd_subset = epd[['Name (en)', 'Ref. unit', 'Bulk Density (kg/m3)', 'GWPtotal (A2)', 'Module']].copy()
epd_subset = epd_subset.rename(columns={'GWPtotal (A2)': 'GWP'})
epd2_subset = epd2[['Name (en)', 'Ref. unit', 'Bulk Density (kg/m3)', 'GWPtotal (A2)', 'Module']].copy()
epd2_subset = epd2_subset.rename(columns={'GWPtotal (A2)': 'GWP'})
# verificar unidades
print(epd_subset['Ref. unit'].unique())
print(epd2_subset['Ref. unit'].unique())

summary = epd_subset.drop_duplicates()
summary2= epd2_subset.drop_duplicates()
print("=== Tipos de concreto y valores ===")
print(summary)
print(summary2)
##Ver filas con valores faltantes
missing_gwp = summary[summary[["GWP","Module"]].isna().any(axis=1)]
missing_gwp2 = summary2[summary2[["GWP","Module"]].isna().any(axis=1)]
print("\n=== Filas con GWP faltante ===")
print(missing_gwp)

##Ver si hay duplicados exactos
duplicates = summary[summary.duplicated(subset=["Name (en)"], keep=False)]
duplicates2= summary2[summary2.duplicated(subset=["Name (en)"], keep=False)]
print("\n=== Duplicados por nombre y unidad ===")
print(duplicates)

# Información general
print("\n=== Información general ===")
print(f"Total tipos de concreto: {summary['Name (en)'].nunique()}")
print(f"Unidades de referencia disponibles: {summary['Ref. unit'].unique()}")
print(f"Cantidad de filas con GWP disponible: {summary.dropna(subset=['GWP']).shape[0]}")
#Unificar
epd_subset["source"]="Italy"
epd2_subset["source"]="UK"
epd_all=pd.concat([epd_subset, epd2_subset], ignore_index=True)
plt.figure(figsize=(8,6))
sns.boxplot(x="source", y="GWP", data=epd_all)
plt.title("Comparación EPD UK vs Italy")
plt.xlabel("Fuente")
plt.ylabel("GWP (kg CO₂ eq / m³)")
plt.tight_layout()
plt.savefig(IMG_DIR/"comparación epd")
plt.show()
RES_DIR = OUTPUTS_DIR/"resultados"
df_ifc.to_pickle(RES_DIR/"df_ifc_limpio.pkl")
epd_subset.to_pickle(RES_DIR/"epd_subset_limpio.pkl")