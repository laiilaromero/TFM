#Importar librerias necesarias
from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
#Primero preguntar tipo de archivo
ROOT= Path(__file__).resolve().parents[1]
DATA_DIR = ROOT/"Data"
OUT_DIR= ROOT/"outputs"
RES_DIR= OUT_DIR/"resultados"
archivo= input("Ingresa el archivo de datos: ").strip()
ruta_archivo = DATA_DIR/ archivo
#obtener la extension
ext = ruta_archivo.suffix.lower()
nombre = ruta_archivo.stem

#Cargar datos segun extension
if ext == ".csv":
    try:
        df= pd.read_csv(ruta_archivo, sep= ";", on_bad_lines="skip", encoding="utf-8")
    except UnicodeDecodeError:
        df= pd.read_csv(ruta_archivo, sep= ";", on_bad_lines="skip", encoding="latin-1")
elif ext in [".xls", ".xlsx"]:
    df= pd.read_excel(ruta_archivo)
elif ext == ".json":
    df= pd.read_json(ruta_archivo, lines=True)
elif ext == ".parquet":
    df=pd.read_parquet(ruta_archivo)


#Generar reporte
profile = ProfileReport(
    df,
    title = f'Reporte EDA automatico -{nombre}',
    explorative = True,
    minimal = False
)

#Exportar el repporte HTML
profile.to_file(RES_DIR/f"{nombre}_reporte.html")
print(f"✅ Reporte generado: {nombre}reporte.html")