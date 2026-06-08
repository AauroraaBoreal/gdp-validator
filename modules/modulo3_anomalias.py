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

def analizar_patron_jackpots(df):
    """Analiza si los jackpots tienen un patrón sospechoso por juego."""
    jackpots = df[pd.notna(df['TotalJPWin']) & (df['TotalJPWin'] > 0)].copy()
    jackpots = jackpots.sort_values('EventTime')

    if len(jackpots) == 0:
        return {}, 0, 0

    # Analizar por juego
    patron_por_juego = {}
    for juego, grupo in jackpots.groupby('GameId'):
        grupo = grupo.sort_values('EventTime')
        tiempos = grupo['EventTime'].tolist()
        juntos = 0
        chispeados = 0
        for i in range(1, len(tiempos)):
            diff = (tiempos[i] - tiempos[i-1]).total_seconds() / 3600
            if diff < 1:
                juntos += 1
            else:
                chispeados += 1
        patron_por_juego[juego] = {
            'total': len(grupo),
            'juntos': juntos,
            'chispeados': chispeados,
            'montos': grupo['TotalJPWin'].tolist(),
            'tiempos': tiempos
        }

    total_juntos = sum(p['juntos'] for p in patron_por_juego.values())
    total_chispeados = sum(p['chispeados'] for p in patron_por_juego.values())
    return patron_por_juego, total_juntos, total_chispeados

def es_jackpot_sospechoso(row, patron_por_juego):
    """Determina si un jackpot específico es sospechoso según contexto."""
    juego = row.get('GameId', '')
    if juego not in patron_por_juego:
        return False, None

    patron = patron_por_juego[juego]
    total = patron['total']
    juntos = patron['juntos']
    chispeados = patron['chispeados']

    # Un jackpot aislado no es sospechoso
    if total == 1:
        return False, None

    # 3+ jackpots juntos en el mismo juego
    if juntos >= 3:
        duracion_horas = (patron['tiempos'][-1] - patron['tiempos'][0]).total_seconds() / 3600
        razon = (
            f"El jugador registró {total} jackpots en el juego {juego} "
            f"en un lapso de {duracion_horas:.1f} horas, con {juntos} ocurrencias "
            f"en intervalos menores a 1 hora. La probabilidad de obtener jackpots "
            f"repetidamente en tan poco tiempo es estadísticamente inusual."
        )
        return True, razon

    # 5+ jackpots chispeados en el mismo juego
    if chispeados >= 5:
        duracion_horas = (patron['tiempos'][-1] - patron['tiempos'][0]).total_seconds() / 3600
        razon = (
            f"El jugador acumuló {total} jackpots en el juego {juego} "
            f"a lo largo de {duracion_horas:.1f} horas. Aunque espaciados, "
            f"la frecuencia de {total} jackpots en el mismo juego durante el periodo "
            f"es inusualmente alta para el patrón esperado."
        )
        return True, razon

    return False, None

def generar_razon_anomalia(row, conteo_junto, conteo_chispeado, patron_por_juego):
    """Genera una explicación en lenguaje natural de por qué la jugada es anómala."""
    if pd.notna(row.get('TotalJPWin')) and row['TotalJPWin'] > 0:
        juego = row.get('GameId', '')
        if juego in patron_por_juego:
            patron = patron_por_juego[juego]
            total = patron['total']
            juntos = patron['juntos']
            duracion = (patron['tiempos'][-1] - patron['tiempos'][0]).total_seconds() / 3600
            if juntos >= 3:
                return (
                    f"Se detectaron {total} jackpots en el juego {juego} "
                    f"en {duracion:.1f} horas, con {juntos} ocurrencias en menos de 1 hora entre sí. "
                    f"Esta frecuencia es estadísticamente inusual."
                )
            else:
                return (
                    f"Se detectaron {total} jackpots en el juego {juego} "
                    f"distribuidos en {duracion:.1f} horas. La acumulación de jackpots "
                    f"en el mismo juego durante el periodo supera el umbral esperado."
                )

    if row['ratio_ganancia'] >= 100:
        return (
            f"La ganancia de ${row['TotalWin']:,.2f} MXN con una apuesta de "
            f"${row['TotalBet']:,.2f} MXN representa un ratio de x{row['ratio_ganancia']:.1f}, "
            f"extremadamente alejado del patrón habitual del jugador. "
            f"Este nivel de ganancia requiere verificación inmediata."
        )

    if row['ratio_ganancia'] >= 36:
        return (
            f"La ganancia de ${row['TotalWin']:,.2f} MXN con una apuesta de "
            f"${row['TotalBet']:,.2f} MXN genera un ratio de x{row['ratio_ganancia']:.1f}. "
            f"El jugador registró {conteo_junto} ganancias altas juntas y "
            f"{conteo_chispeado} distribuidas en el periodo, superando el umbral definido."
        )

    if row['ratio_ganancia'] >= 26:
        return (
            f"La ganancia de ${row['TotalWin']:,.2f} MXN con apuesta de "
            f"${row['TotalBet']:,.2f} MXN genera un ratio de x{row['ratio_ganancia']:.1f}. "
            f"Se detectaron {conteo_junto + conteo_chispeado} ganancias con ratio ≥ x26 "
            f"en el periodo ({conteo_junto} juntas, {conteo_chispeado} distribuidas), "
            f"lo que indica un patrón de ganancias altas recurrente."
        )

    return "Comportamiento transaccional que se desvía del patrón habitual del jugador según el modelo de detección."

def evaluar_anomalia(row, umbral_ratio, umbral_bet, conteo_junto, conteo_chispeado, patron_por_juego):
    # Jackpot: evaluar con criterio contextual
    if pd.notna(row.get('TotalJPWin')) and row['TotalJPWin'] > 0:
        sospechoso, _ = es_jackpot_sospechoso(row, patron_por_juego)
        return sospechoso

    # Ratio >= 100: siempre anómalo
    if row['ratio_ganancia'] >= 100:
        return True

    # Ratio entre 36 y 99
    if 36 <= row['ratio_ganancia'] < 100:
        if conteo_junto >= 3 or conteo_chispeado >= 5:
            return True

    # Ratio entre 26 y 35
    if 26 <= row['ratio_ganancia'] < 36:
        if conteo_junto >= 3 or conteo_chispeado >= 5:
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

    ratio_mean = df['ratio_ganancia'].mean()
    ratio_std = df['ratio_ganancia'].std()
    bet_mean = df['TotalBet'].mean()
    bet_std = df['TotalBet'].std()
    umbral_ratio = ratio_mean + (3 * ratio_std)
    umbral_bet = bet_mean + (3 * bet_std)

    conteo_junto, conteo_chispeado = analizar_patron_ganancias_altas(df)
    patron_por_juego, jp_juntos, jp_chispeados = analizar_patron_jackpots(df)

    print(f"Patrón ganancias altas — Juntas (<1h): {conteo_junto} | Chispeadas (>1h): {conteo_chispeado}")
    print(f"Patrón jackpots — Juntos (<1h): {jp_juntos} | Chispeados (>1h): {jp_chispeados}")

    df['es_anomalia'] = df.apply(
        lambda row: evaluar_anomalia(
            row, umbral_ratio, umbral_bet,
            conteo_junto, conteo_chispeado, patron_por_juego
        ),
        axis=1
    )

    df['es_free_game_inusual'] = df.apply(
        lambda row: row['TotalBet'] == 0 and row['TotalWin'] > 50000,
        axis=1
    )

    df['tipo_anomalia'] = df.apply(
        lambda row: clasificar_tipo_anomalia(row) if row['es_anomalia'] else None,
        axis=1
    )

    df['razon_anomalia'] = df.apply(
        lambda row: generar_razon_anomalia(
            row, conteo_junto, conteo_chispeado, patron_por_juego
        ) if row['es_anomalia'] else None,
        axis=1
    )

    total_anomalias = df['es_anomalia'].sum()
    total_free_games = df['es_free_game_inusual'].sum()
    print(f"Anomalías detectadas: {total_anomalias} de {len(df)} registros ({total_anomalias/len(df)*100:.2f}%)")
    if total_free_games > 0:
        print(f"Observaciones free games inusuales: {total_free_games} (no marcados como anomalía)")

    return df, modelo

def obtener_observaciones_free_games(df):
    free_games = df[df['es_free_game_inusual'] == True]
    if free_games.empty:
        return None
    total_ganancia = free_games['TotalWin'].sum()
    conteo = len(free_games)
    return f"Se observaron {conteo} jugada(s) de free games con ganancia acumulada de ${total_ganancia:,.2f} MXN sin apuesta asociada."

def actualizar_modelo_incremental(df_nuevo):
    X_nuevo = preparar_features_anomalias(df_nuevo)
    if os.path.exists(MODELO_ANOMALIAS_PATH):
        print("Actualizando modelo con nuevos datos...")
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