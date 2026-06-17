import pandas as pd
import numpy as np
import csv

ruta_archivo = 'data/132425.csv'
data = []
with open(ruta_archivo, 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f)
    header = next(reader)
    for i, row in enumerate(reader):
        if len(row) < len(header):
            row.extend([''] * (len(header) - len(row)))
        elif len(row) > len(header):
            row = row[:len(header)]
        row.append(i + 2)
        data.append(row)

header.append('_fila_csv')
df = pd.DataFrame(data, columns=header)
df.replace('', np.nan, inplace=True)

df['EventTime'] = pd.to_datetime(df['EventTime'], errors='coerce')
df = df.dropna(subset=['EventTime'])
df = df.sort_values('EventTime').reset_index(drop=True)

COLUMNAS_MONTO = ['TotalBet', 'BalanceStart', 'TotalWin', 'TotalJPWin', 'BalanceChange', 'Balance']
for col in COLUMNAS_MONTO:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) / 100

balance = df['Balance'].values
balance_change = df['BalanceChange'].values
diferencias = -(balance[:-1] - balance[1:] + balance_change[1:])

mask_retiros = diferencias < 0
indices_cercanos = np.where(
    mask_retiros &
    (np.abs(diferencias + 17800) <= 100)
)[0]

print("Indices cercanos:", indices_cercanos)
for idx in indices_cercanos:
    print(f"Index {idx}, Monto real: {abs(diferencias[idx])}, Fila CSV: {df.loc[idx, '_fila_csv']}")
