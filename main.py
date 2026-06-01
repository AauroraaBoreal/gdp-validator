import sys
from modules.modulo1_carga import cargar_csv
from modules.modulo2_clasificacion import clasificar_eventos
from modules.modulo3_anomalias import detectar_anomalias, actualizar_modelo_incremental
from modules.modulo4_base_datos import crear_tablas, guardar_jugador, guardar_validacion, guardar_anomalias
from modules.modulo5_reportes import mostrar_reportes

def procesar_csv(ruta_archivo):
    print("\n" + "=" * 60)
    print("GDP VALIDATOR — SISTEMA DE VALIDACIÓN INTELIGENTE")
    print("=" * 60)

    # Paso 1: Cargar y preprocesar
    df = cargar_csv(ruta_archivo)

    # Paso 2: Clasificar eventos
    df = clasificar_eventos(df)

    # Paso 3: Detectar anomalías
    df, _ = detectar_anomalias(df)

    # Paso 4: Mostrar reportes
    resultado, reporte_whatsapp, reporte_qa = mostrar_reportes(df)

    # Paso 5: Guardar en base de datos
    crear_tablas()
    guardar_jugador(df)
    validacion_id = guardar_validacion(df, resultado, reporte_whatsapp)
    anomalias = df[df['es_anomalia'] == True]
    guardar_anomalias(anomalias, validacion_id)

    print("\n✅ Proceso completado.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 main.py <ruta_del_csv>")
        print("Ejemplo: python3 main.py data/10069.csv")
        sys.exit(1)

    ruta = sys.argv[1]
    procesar_csv(ruta)