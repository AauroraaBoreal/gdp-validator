import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from modules.modulo1_carga import cargar_csv

df = cargar_csv('data/132425.csv')
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

print("Valores de diferencias negativos que se acerquen a 17800:")
# print some closest ones
diff_abs = np.abs(diferencias + 17800)
idx_min = np.argmin(diff_abs)
print(f"El más cercano es en idx {idx_min}: diferencia = {diferencias[idx_min]}, fila CSV = {df.loc[idx_min, '_fila_csv']}")
