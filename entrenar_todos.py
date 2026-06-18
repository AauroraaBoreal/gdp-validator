from modules.modulo1_carga import cargar_csv
from modules.modulo3_anomalias import entrenar_detector
from modules.modulo6_evaluacion_no_supervisada import evaluar_detector_no_supervisado

import os
import glob
import pandas as pd

CSV_EVALUACION = 'data/30948.csv'

def validar_sistema():
    print("\n" + "=" * 60)
    print("VALIDACIÓN DEL SISTEMA — QA")
    print("=" * 60)
    errores = []

    print("ℹ️ Modelo clasificador omitido: el enfoque principal es no supervisado")

    if not os.path.exists('modelo_anomalias.pkl'):
        errores.append("❌ modelo_anomalias.pkl no encontrado")
    else:
        print("✅ Modelo de anomalías encontrado")

    try:
        from modules.modulo4_base_datos import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(__import__('sqlalchemy').text("SELECT 1"))
        print("✅ Conexión a Supabase correcta")
    except Exception as e:
        errores.append(f"❌ Error de conexión a base de datos: {e}")

    archivos = glob.glob('data/*.csv')
    if not archivos:
        errores.append("❌ No hay archivos CSV en la carpeta data/")
    else:
        print(f"✅ {len(archivos)} archivos CSV disponibles para entrenamiento")

    try:
        from modules.modulo1_carga import cargar_csv
        from modules.modulo3_anomalias import detectar_anomalias
        from modules.modulo5_reportes import mostrar_reportes
        print("✅ Todos los módulos importan correctamente")
    except Exception as e:
        errores.append(f"❌ Error al importar módulos: {e}")

    print("\n" + "-" * 60)
    if errores:
        print("RESULTADO: ⚠️  Se encontraron problemas:")
        for e in errores:
            print(f"  {e}")
    else:
        print("RESULTADO: ✅ Sistema validado correctamente — sin problemas")
    print("=" * 60 + "\n")
    return len(errores) == 0



# --- ENTRENAMIENTO ---
archivos = glob.glob('data/*.csv')
archivos_entrenamiento = [a for a in archivos if os.path.basename(a) != '30948.csv']
print(f"Archivos para entrenamiento: {len(archivos_entrenamiento)}")
print(f"Archivo reservado para evaluación: {CSV_EVALUACION}\n")

df_total = None

for archivo in archivos_entrenamiento:
    print(f"Procesando: {archivo}")
    try:
        df = cargar_csv(archivo)
        if df_total is None:
            df_total = df
        else:
            df_total = pd.concat([df_total, df], ignore_index=True)
    except Exception as e:
        print(f"Error en {archivo}: {e}")

print(f"\nTotal de registros para entrenamiento: {len(df_total)}")

print("\nEntrenando detector de anomalías no supervisado...")
modelo_anomalias = entrenar_detector(df_total)

print("\nEntrenamiento completo.")

# --- EVALUACIÓN NO SUPERVISADA ---
print("\nCargando CSV de evaluación independiente...")
df_prueba = cargar_csv(CSV_EVALUACION)

df_resultado = evaluar_detector_no_supervisado(
    df_prueba,
    modelo_anomalias,
    salida_csv="resultados_evaluacion_no_supervisada.csv"
)

# --- VALIDACIÓN QA ---
validar_sistema()