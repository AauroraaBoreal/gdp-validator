import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

from modules.modulo3_anomalias import (
    FEATURES_ANOMALIAS,
    preparar_features_anomalias,
    marcar_free_games,
    evaluar_anomalia,
    analizar_patron_ganancias_altas,
    analizar_patron_jackpots,
    clasificar_tipo_anomalia,
    generar_razon_anomalia
)

MODELO_RANDOM_FOREST_PATH = 'modelo_random_forest.pkl'

def entrenar_random_forest(df):
    print("Entrenando Random Forest Classifier...")
    df = marcar_free_games(df)
    X = preparar_features_anomalias(df)
    
    # Importar localmente para evitar importación circular
    from modules.modulo6_evaluacion_no_supervisada import marcar_anomalia_por_reglas
    y = df.apply(marcar_anomalia_por_reglas, axis=1).astype(int)
    
    # Random Forest con balanceo de clases debido al fuerte desbalanceo
    modelo_rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    modelo_rf.fit(X, y)
    print(f"Random Forest entrenado con éxito.")
    
    with open(MODELO_RANDOM_FOREST_PATH, 'wb') as f:
        pickle.dump(modelo_rf, f)
    print(f"Modelo Random Forest guardado en {MODELO_RANDOM_FOREST_PATH}")
    return modelo_rf

def detectar_anomalias_random_forest(df):
    df = marcar_free_games(df)
    X = preparar_features_anomalias(df)

    if os.path.exists(MODELO_RANDOM_FOREST_PATH):
        with open(MODELO_RANDOM_FOREST_PATH, 'rb') as f:
            modelo_rf = pickle.load(f)
        print("Modelo Random Forest cargado desde archivo")
    else:
        modelo_rf = entrenar_random_forest(df)

    # Asegurar compatibilidad de features
    if hasattr(modelo_rf, 'feature_names_in_'):
        columnas_modelo = list(modelo_rf.feature_names_in_)
        for col in columnas_modelo:
            if col not in X.columns:
                if col == 'es_free_game':
                    if 'es_free_game' not in df.columns:
                        df = marcar_free_games(df)
                    X['es_free_game'] = df['es_free_game'].astype(float)
                else:
                    X[col] = 0.0
        X = X[columnas_modelo]

    try:
        # Calcular probabilidad de ser anomalía (clase 1)
        prob_anomaly = modelo_rf.predict_proba(X)[:, 1]
        
        # score = 0.5 - prob. Si prob > 0.5, score < 0 (anomalía)
        df['anomalia_score'] = 0.5 - prob_anomaly
        df['es_anomalia'] = (prob_anomaly > 0.5) & (~df['es_free_game'])
    except Exception as e:
        print(f"Advertencia: Error al usar el modelo Random Forest guardado ({e}). Re-entrenando...")
        modelo_rf = entrenar_random_forest(df)
        if hasattr(modelo_rf, 'feature_names_in_'):
            X = X[list(modelo_rf.feature_names_in_)]
        prob_anomaly = modelo_rf.predict_proba(X)[:, 1]
        df['anomalia_score'] = 0.5 - prob_anomaly
        df['es_anomalia'] = (prob_anomaly > 0.5) & (~df['es_free_game'])

    ratio_mean = df['ratio_ganancia'].mean()
    ratio_std = df['ratio_ganancia'].std()
    bet_mean = df['TotalBet'].mean()
    bet_std = df['TotalBet'].std()
    umbral_ratio = ratio_mean + (3 * ratio_std)
    umbral_bet = bet_mean + (3 * bet_std)

    conteo_junto, conteo_chispeado = analizar_patron_ganancias_altas(df)
    patron_por_juego, jp_juntos, jp_chispeados = analizar_patron_jackpots(df)

    print(f"[RF] Patrón ganancias altas — Juntas (<1h): {conteo_junto} | Chispeadas (>1h): {conteo_chispeado}")
    print(f"[RF] Patrón jackpots — Juntos (<1h): {jp_juntos} | Chispeados (>1h): {jp_chispeados}")

    df['es_free_game_inusual'] = df.apply(
        lambda row: row['TotalBet'] == 0 and row['TotalWin'] > 50000 and not row.get('es_free_game', False),
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
    print(f"[RF] Anomalías detectadas: {total_anomalias} de {len(df)} registros ({total_anomalias/len(df)*100:.2f}%)")
    
    return df, modelo_rf
