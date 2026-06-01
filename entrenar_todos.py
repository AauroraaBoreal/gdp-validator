from modules.modulo1_carga import cargar_csv
from modules.modulo2_clasificacion import entrenar_modelo, clasificar_eventos
from modules.modulo3_anomalias import actualizar_modelo_incremental, entrenar_detector
from sklearn.metrics import classification_report, accuracy_score
import os
import glob
import pandas as pd

CSV_EVALUACION = 'data/30948.csv'

def validar_sistema():
    print("\n" + "=" * 60)
    print("VALIDACIÓN DEL SISTEMA — QA")
    print("=" * 60)
    errores = []

    if not os.path.exists('modelo_clasificador.pkl'):
        errores.append("❌ modelo_clasificador.pkl no encontrado")
    else:
        print("✅ Modelo clasificador encontrado")

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
        from modules.modulo2_clasificacion import clasificar_eventos
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

def evaluar_modelo(modelo, df_prueba):
    """Evalúa el modelo con un CSV que no participó en el entrenamiento."""
    from modules.modulo2_clasificacion import preparar_features, etiquetar_evento
    import pickle

    print("\n" + "=" * 60)
    print("EVALUACIÓN DEL MODELO — CSV INDEPENDIENTE")
    print(f"Archivo de prueba: {CSV_EVALUACION}")
    print(f"Total registros de prueba: {len(df_prueba)}")
    print("=" * 60)

    # Generar etiquetas reales
    df_prueba['etiqueta_real'] = df_prueba.apply(etiquetar_evento, axis=1)

    # Predecir con el modelo
    X_prueba = preparar_features(df_prueba)
    with open('modelo_clasificador.pkl', 'rb') as f:
        modelo_clf = pickle.load(f)

    df_prueba['etiqueta_predicha'] = modelo_clf.predict(X_prueba)

    # Métricas
    print("\nDistribución de etiquetas reales en CSV de prueba:")
    print(df_prueba['etiqueta_real'].value_counts())

    print("\nReporte de clasificación:")
    print(classification_report(
        df_prueba['etiqueta_real'],
        df_prueba['etiqueta_predicha'],
        zero_division=0
    ))

    acc = accuracy_score(df_prueba['etiqueta_real'], df_prueba['etiqueta_predicha'])
    print(f"Accuracy general: {acc:.4f} ({acc*100:.2f}%)")
    print("=" * 60)

    return acc

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
print("Entrenando modelo clasificador...")
modelo_clf, df_total = entrenar_modelo(df_total)

print("\nEntrenando detector de anomalías...")
entrenar_detector(df_total)

actualizar_modelo_incremental(df_total)
print("\nEntrenamiento completo.")

# --- EVALUACIÓN ---
print("\nCargando CSV de evaluación independiente...")
df_prueba = cargar_csv(CSV_EVALUACION)
evaluar_modelo(modelo_clf, df_prueba)

# --- VALIDACIÓN QA ---
validar_sistema()