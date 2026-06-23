import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

from modules.modulo3_anomalias import (
    MODELO_ANOMALIAS_PATH,
    FEATURES_ANOMALIAS,
    preparar_features_anomalias,
    marcar_free_games,
    evaluar_anomalia,
    analizar_patron_ganancias_altas,
    analizar_patron_jackpots,
    clasificar_tipo_anomalia,
    generar_razon_anomalia
)

MODELO_AUTOENCODER_PATH = 'modelo_autoencoder.pkl'

def entrenar_autoencoder(df):
    print("Entrenando Autoencoder (MLPRegressor)...")
    df = marcar_free_games(df)
    X = preparar_features_anomalias(df)
    
    # Escalado de características es fundamental para redes neuronales / autoencoders
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Arquitectura de cuello de botella: input es de dim 6, hidden_layer_sizes = (4, 2, 4)
    # Reconstruye el input (X_scaled)
    modelo_ae = MLPRegressor(
        hidden_layer_sizes=(4, 2, 4),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    modelo_ae.fit(X_scaled, X_scaled)
    
    # Calcular errores de reconstrucción para encontrar el umbral
    X_pred = modelo_ae.predict(X_scaled)
    reconstruction_errors = np.mean((X_scaled - X_pred) ** 2, axis=1)
    
    # Umbral al percentil 98 (tasa de contaminación = 2%)
    threshold = np.percentile(reconstruction_errors, 98)
    print(f"Autoencoder entrenado. Umbral de reconstrucción (percentil 98): {threshold:.6f}")
    
    bundle = {
        'scaler': scaler,
        'model': modelo_ae,
        'threshold': threshold,
        'features': FEATURES_ANOMALIAS
    }
    
    with open(MODELO_AUTOENCODER_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"Modelo de autoencoder guardado en {MODELO_AUTOENCODER_PATH}")
    return bundle

def detectar_anomalias_autoencoder(df):
    df = marcar_free_games(df)
    X = preparar_features_anomalias(df)

    if os.path.exists(MODELO_AUTOENCODER_PATH):
        with open(MODELO_AUTOENCODER_PATH, 'rb') as f:
            bundle = pickle.load(f)
        print("Modelo de autoencoder cargado desde archivo")
    else:
        bundle = entrenar_autoencoder(df)

    scaler = bundle['scaler']
    modelo_ae = bundle['model']
    threshold = bundle['threshold']
    columnas_modelo = bundle['features']

    # Asegurar compatibilidad de features
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
        X_scaled = scaler.transform(X)
        X_pred = modelo_ae.predict(X_scaled)
        reconstruction_error = np.mean((X_scaled - X_pred) ** 2, axis=1)
        
        # score = threshold - error. Así, si score < 0, es anomalía.
        # Además, al ordenar ascendente los scores, los más negativos (mayor error) salen primero.
        df['anomalia_score'] = threshold - reconstruction_error
        df['es_anomalia'] = (reconstruction_error > threshold) & (~df['es_free_game'])
    except Exception as e:
        print(f"Advertencia: Error al usar el modelo autoencoder guardado ({e}). Re-entrenando...")
        bundle = entrenar_autoencoder(df)
        scaler = bundle['scaler']
        modelo_ae = bundle['model']
        threshold = bundle['threshold']
        X_scaled = scaler.transform(X)
        X_pred = modelo_ae.predict(X_scaled)
        reconstruction_error = np.mean((X_scaled - X_pred) ** 2, axis=1)
        df['anomalia_score'] = threshold - reconstruction_error
        df['es_anomalia'] = (reconstruction_error > threshold) & (~df['es_free_game'])

    ratio_mean = df['ratio_ganancia'].mean()
    ratio_std = df['ratio_ganancia'].std()
    bet_mean = df['TotalBet'].mean()
    bet_std = df['TotalBet'].std()
    umbral_ratio = ratio_mean + (3 * ratio_std)
    umbral_bet = bet_mean + (3 * bet_std)

    conteo_junto, conteo_chispeado = analizar_patron_ganancias_altas(df)
    patron_por_juego, jp_juntos, jp_chispeados = analizar_patron_jackpots(df)

    print(f"[AE] Patrón ganancias altas — Juntas (<1h): {conteo_junto} | Chispeadas (>1h): {conteo_chispeado}")
    print(f"[AE] Patrón jackpots — Juntos (<1h): {jp_juntos} | Chispeados (>1h): {jp_chispeados}")

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
    print(f"[AE] Anomalías detectadas: {total_anomalias} de {len(df)} registros ({total_anomalias/len(df)*100:.2f}%)")
    
    # Para el autoencoder también retornamos el bundle entrenado
    return df, bundle
