import pandas as pd
import os

COLUMNAS_REQUERIDAS = [
    'PlayerId', 'BalanceStart', 'BalanceChange', 'TotalBet',
    'TotalWin', 'TotalJPWin', 'EventTime', 'GameInstanceId', 'EventId'
]

COLUMNAS_MONTO = [
    'TotalBet', 'BalanceStart', 'TotalWin',
    'TotalJPWin', 'BalanceChange', 'Balance'
]

def cargar_csv(ruta_archivo):
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

    if not ruta_archivo.endswith('.csv'):
        raise ValueError("El archivo debe ser de formato CSV (.csv)")

    import csv
    data = []
    with open(ruta_archivo, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("El archivo CSV está vacío")
        
        for i, row in enumerate(reader):
            if len(row) < len(header):
                row.extend([''] * (len(header) - len(row)))
            elif len(row) > len(header):
                row = row[:len(header)]
            row.append(i + 2)  # Excel line number (1-based index + header)
            data.append(row)

    header.append('_fila_csv')
    df = pd.DataFrame(data, columns=header)
    df = df.replace('', pd.NA)
    print(f"Archivo cargado: {len(df)} registros")

    faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
    if faltantes:
        raise ValueError(f"El archivo no tiene el formato esperado. Columnas faltantes: {faltantes}")

    if df['PlayerId'].isna().all():
        raise ValueError("El archivo no contiene registros válidos de jugador")

    df['EventTime'] = pd.to_datetime(df['EventTime'], errors='coerce')
    # Convertir de UTC a horario de Perú (UTC-5)
    if df['EventTime'].dt.tz is None:
        df['EventTime'] = df['EventTime'].dt.tz_localize('UTC')
    df['EventTime'] = df['EventTime'].dt.tz_convert('America/Lima').dt.tz_localize(None)

    filas_sin_fecha = df['EventTime'].isna().sum()
    if filas_sin_fecha > 0:
        print(f"Se eliminaron {filas_sin_fecha} filas sin fecha válida")
    df = df.dropna(subset=['EventTime'])

    if len(df) == 0:
        raise ValueError("El archivo no contiene registros con fecha válida")

    df = df.sort_values('EventTime').reset_index(drop=True)

    for col in COLUMNAS_MONTO:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) / 100

    df['ratio_ganancia'] = df.apply(
        lambda row: row['TotalWin'] / row['TotalBet'] if row['TotalBet'] > 0 else 0,
        axis=1
    )

    player_id = str(df['PlayerId'].iloc[0])
    fecha_inicio = df['EventTime'].min()
    fecha_fin = df['EventTime'].max()

    print(f"Registros válidos tras limpieza: {len(df)}")
    print(f"Periodo: {fecha_inicio} → {fecha_fin}")
    print(f"Jugador: {player_id}")

    return df