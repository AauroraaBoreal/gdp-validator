import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import pickle
import os

MODELO_ANOMALIAS_PATH = 'modelo_anomalias.pkl'

FEATURES_ANOMALIAS = [
    'TotalBet', 'TotalWin', 'TotalJPWin',
    'BalanceChange', 'ratio_ganancia'
]

def preparar_features_anomalias(df):
    X = df[FEATURES_ANOMALIAS].copy()
    X['TotalJPWin'] = X['TotalJPWin'].fillna(0)
    return X

def entrenar_detector(df):
    print("Entrenando Isolation Forest...")
    X = preparar_features_anomalias(df)

    modelo = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        random_state=42
    )
    modelo.fit(X)

    with open(MODELO_ANOMALIAS_PATH, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"Modelo de anomalías guardado en {MODELO_ANOMALIAS_PATH}")

    return modelo

def analizar_patron_ganancias_altas(df, umbral_ratio=26):
    """
    Analiza si las ganancias altas vienen juntas (< 1h entre sí) o chispeadas (> 1h).
    Retorna el conteo junto y chispeado por separado.
    """
    ganancias_altas = df[df['ratio_ganancia'] >= umbral_ratio].copy()
    ganancias_altas = ganancias_altas.sort_values('EventTime')

    if len(ganancias_altas) == 0:
        return 0, 0

    conteo_junto = 0
    conteo_chispeado = 0
    tiempos = ganancias_altas['EventTime'].tolist()

    for i in range(1, len(tiempos)):
        diferencia = (tiempos[i] - tiempos[i-1]).total_seconds() / 3600
        if diferencia < 1:
            conteo_junto += 1
        else:
            conteo_chispeado += 1

    return conteo_junto, conteo_chispeado

def evaluar_anomalia(row, modelo, x_row, umbral_ratio, umbral_bet, conteo_junto, conteo_chispeado):
    # Jackpot siempre se marca
    if pd.notna(row.get('TotalJPWin')) and row['TotalJPWin'] > 0:
        return True

    # Ratio >= 100: siempre anómalo
    if row['ratio_ganancia'] >= 100:
        return True

    # Ratio entre 36 y 99
    if 36 <= row['ratio_ganancia'] < 100:
        if conteo_junto >= 3:
            return True
        if conteo_chispeado >= 5:
            return True

    # Ratio entre 26 y 35
    if 26 <= row['ratio_ganancia'] < 36:
        if conteo_junto >= 3:
            return True
        if conteo_chispeado >= 5:
            return True

    return False

def clasificar_tipo_anomalia(row):
    if not row['es_anomalia']:
        return None

    if pd.notna(row.get('TotalJPWin')) and row['TotalJPWin'] > 0:
        return 'jackpot_sospechoso'

    if row['ratio_ganancia'] >= 100:
        return 'ganancia_anomala'

    if row['ratio_ganancia'] >= 36:
        return 'ganancia_media_sospechosa'

    if row['ratio_ganancia'] >= 26:
        return 'ganancia_alta_repetitiva'

    return 'comportamiento_atipico'

def detectar_anomalias(df):
    X = preparar_features_anomalias(df)

    if os.path.exists(MODELO_ANOMALIAS_PATH):
        with open(MODELO_ANOMALIAS_PATH, 'rb') as f:
            modelo = pickle.load(f)
        print("Modelo de anomalías cargado desde archivo")
    else:
        modelo = entrenar_detector(df)

    df['anomalia_score'] = modelo.decision_function(X)

    # Calcular umbrales personalizados por jugador
    ratio_mean = df['ratio_ganancia'].mean()
    ratio_std = df['ratio_ganancia'].std()
    bet_mean = df['TotalBet'].mean()
    bet_std = df['TotalBet'].std()

    umbral_ratio = ratio_mean + (3 * ratio_std)
    umbral_bet = bet_mean + (3 * bet_std)

    # Analizar patrón de ganancias altas
    conteo_junto, conteo_chispeado = analizar_patron_ganancias_altas(df)
    print(f"Patrón ganancias altas — Juntas (<1h): {conteo_junto} | Chispeadas (>1h): {conteo_chispeado}")

    df['es_anomalia'] = df.apply(
        lambda row: evaluar_anomalia(
            row, modelo, X.loc[row.name],
            umbral_ratio, umbral_bet,
            conteo_junto, conteo_chispeado
        ),
        axis=1
    )

    # Free games inusual: observación sin marcar como anomalía
    df['es_free_game_inusual'] = df.apply(
        lambda row: row['TotalBet'] == 0 and row['TotalWin'] > 50000,
        axis=1
    )

    df['tipo_anomalia'] = df.apply(
        lambda row: clasificar_tipo_anomalia(row) if row['es_anomalia'] else None,
        axis=1
    )

    total_anomalias = df['es_anomalia'].sum()
    total_free_games = df['es_free_game_inusual'].sum()
    print(f"Anomalías detectadas: {total_anomalias} de {len(df)} registros ({total_anomalias/len(df)*100:.2f}%)")
    if total_free_games > 0:
        print(f"Observaciones free games inusuales: {total_free_games} (no marcados como anomalía)")

    return df, modelo

def actualizar_modelo_incremental(df_nuevo):
    """
    Aprendizaje incremental: reentrena el modelo incluyendo los nuevos datos.
    """
    X_nuevo = preparar_features_anomalias(df_nuevo)

    if os.path.exists(MODELO_ANOMALIAS_PATH):
        print("Actualizando modelo con nuevos datos...")
        # En lugar de combinar estimadores, reentrenamos con los nuevos datos
        modelo_nuevo = IsolationForest(
            n_estimators=100,
            contamination=0.02,
            random_state=42
        )
        modelo_nuevo.fit(X_nuevo)

        with open(MODELO_ANOMALIAS_PATH, 'wb') as f:
            pickle.dump(modelo_nuevo, f)
        print(f"Modelo actualizado con {len(df_nuevo)} registros nuevos.")
        return modelo_nuevo
    else:
        return entrenar_detector(df_nuevo)

def obtener_observaciones_free_games(df):
    free_games = df[df['es_free_game_inusual'] == True]
    if free_games.empty:
        return None
    total_ganancia = free_games['TotalWin'].sum()
    conteo = len(free_games)
    return f"Se observaron {conteo} jugada(s) de free games con ganancia acumulada de ${total_ganancia:,.2f} MXN sin apuesta asociada."