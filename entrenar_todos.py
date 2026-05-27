from modules.modulo1_carga import cargar_csv
from modules.modulo2_clasificacion import entrenar_modelo
from modules.modulo3_anomalias import actualizar_modelo_incremental
import os
import glob

archivos = glob.glob('data/*.csv')
print(f"Archivos encontrados: {len(archivos)}")

df_total = None

for archivo in archivos:
    print(f"\nProcesando: {archivo}")
    try:
        df = cargar_csv(archivo)
        if df_total is None:
            df_total = df
        else:
            import pandas as pd
            df_total = pd.concat([df_total, df], ignore_index=True)
    except Exception as e:
        print(f"Error en {archivo}: {e}")

print(f"\nTotal de registros combinados: {len(df_total)}")
print("Entrenando modelo con todos los datos...")
modelo_clf, df_total = entrenar_modelo(df_total)
actualizar_modelo_incremental(df_total)
print("\nEntrenamiento completo con todos los CSV.")