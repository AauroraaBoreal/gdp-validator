import pandas as pd
import os

COLUMNAS_REQUERIDAS = [
    'PlayerId', 'BalanceStart', 'BalanceChange', 'TotalBet',
    'TotalWin', 'TotalJPWin', 'EventTime', 'GameInstanceId'
]

COLUMNAS_MONTO = [
    'TotalBet', 'BalanceStart', 'TotalWin',
    'TotalJPWin', 'BalanceChange', 'Balance'
]

def cargar_csv(ruta_archivo):
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

    # Validar que sea un archivo CSV
    if not ruta_archivo.endswith('.csv'):
        raise ValueError("El archivo debe ser de formato CSV (.csv)")

    # Leer con quoting para manejar arrays en columnas como Jackpot y JackpotId
    df = pd.read_csv(ruta_archivo, on_bad_lines='skip', quotechar='"', engine='python')
    print(f"Archivo cargado: {len(df)} registros")

    # Validar columnas requeridas
    faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
    if faltantes:
        raise ValueError(f"El archivo no tiene el formato esperado. Columnas faltantes: {faltantes}")

    # Validar que PlayerId tenga valores
    if df['PlayerId'].isna().all():
        raise ValueError("El archivo no contiene registros válidos de jugador")

    # Convertir EventTime a datetime
    df['EventTime'] = pd.to_datetime(df['EventTime'], errors='coerce')

    # Eliminar filas sin fecha válida
    filas_sin_fecha = df['EventTime'].isna().sum()
    if filas_sin_fecha > 0:
        print(f"Se eliminaron {filas_sin_fecha} filas sin fecha válida")
    df = df.dropna(subset=['EventTime'])

    if len(df) == 0:
        raise ValueError("El archivo no contiene registros con fecha válida")

    # Ordenar cronológicamente
    df = df.sort_values('EventTime').reset_index(drop=True)

    # Convertir montos a escala real (dividir entre 100)
    for col in COLUMNAS_MONTO:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100

    # Eliminar filas donde TotalBet sea nulo
    df = df.dropna(subset=['TotalBet'])

    # Calcular ratio ganancia/apuesta
    df['ratio_ganancia'] = df.apply(
        lambda row: row['TotalWin'] / row['TotalBet'] if row['TotalBet'] > 0 else 0,
        axis=1
    )

    # Verificar si el archivo ya fue procesado
    player_id = str(df['PlayerId'].iloc[0])
    fecha_inicio = df['EventTime'].min()
    fecha_fin = df['EventTime'].max()

    print(f"Registros válidos tras limpieza: {len(df)}")
    print(f"Periodo: {fecha_inicio} → {fecha_fin}")
    print(f"Jugador: {player_id}")

    return df