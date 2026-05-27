from modules.modulo1_carga import cargar_csv
from modules.modulo2_clasificacion import entrenar_modelo, clasificar_eventos
from modules.modulo3_anomalias import entrenar_detector, detectar_anomalias
from modules.modulo4_base_datos import crear_tablas, guardar_jugador, guardar_validacion, guardar_anomalias

# Crear tablas en Supabase
crear_tablas()

# Cargar CSV
df = cargar_csv('data/10069.csv')

# Clasificación
modelo_clf, df = entrenar_modelo(df)
df = clasificar_eventos(df)

# Detección de anomalías
entrenar_detector(df)
df, modelo_anomalias = detectar_anomalias(df)

# Guardar en base de datos
guardar_jugador(df)
validacion_id = guardar_validacion(df, 'normal', 'Reporte de prueba')
anomalias = df[df['es_anomalia'] == True]
guardar_anomalias(anomalias, validacion_id)