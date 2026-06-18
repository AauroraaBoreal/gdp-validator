import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score


from modules.modulo3_anomalias import (
    preparar_features_anomalias,
    marcar_free_games,
    evaluar_anomalia,
    analizar_patron_ganancias_altas,
    analizar_patron_jackpots
)


def marcar_anomalia_por_reglas(row):
    """
    Esta función NO representa una etiqueta real.
    Solo representa una regla de negocio para comparar contra el modelo.
    """
    if row.get('es_free_game', False):
        return False

    if pd.notna(row.get("TotalJPWin")) and row["TotalJPWin"] > 0:
        return True

    if row.get("ratio_ganancia", 0) >= 26:
        return True

    if row.get("TotalBet", 0) == 0 and row.get("TotalWin", 0) > 50000:
        return True

    return False


def nivel_riesgo_por_score(score, q10, q25):
    """
    En IsolationForest, mientras menor es el score, más anómalo es el registro.
    """
    if score <= q10:
        return "alto"
    elif score <= q25:
        return "medio"
    else:
        return "bajo"


def evaluar_detector_no_supervisado(df, modelo, salida_csv="resultados_evaluacion_no_supervisada.csv"):
    print("\n" + "=" * 70)
    print("EVALUACIÓN NO SUPERVISADA - ISOLATION FOREST")
    print("=" * 70)

    from modules.modulo3_anomalias import marcar_free_games
    df_eval = df.copy()
    df_eval = marcar_free_games(df_eval)

    X = preparar_features_anomalias(df_eval)

    # Asegurar compatibilidad de features con el modelo cargado
    if hasattr(modelo, 'feature_names_in_'):
        columnas_modelo = list(modelo.feature_names_in_)
        for col in columnas_modelo:
            if col not in X.columns:
                if col == 'es_free_game':
                    if 'es_free_game' not in df_eval.columns:
                        df_eval = marcar_free_games(df_eval)
                    X['es_free_game'] = df_eval['es_free_game'].astype(float)
                else:
                    X[col] = 0.0
        X = X[columnas_modelo]

    try:
        df_eval["anomalia_score"] = modelo.decision_function(X)
        df_eval["prediccion_modelo"] = modelo.predict(X)
        df_eval["es_anomalia_modelo"] = (df_eval["prediccion_modelo"] == -1) & (~df_eval["es_free_game"])
    except Exception as e:
        print(f"Advertencia: Error al usar el modelo en evaluación ({e}). Re-entrenando detector...")
        from modules.modulo3_anomalias import entrenar_detector
        modelo = entrenar_detector(df_eval)
        X = preparar_features_anomalias(df_eval)
        if hasattr(modelo, 'feature_names_in_'):
            X = X[list(modelo.feature_names_in_)]
        df_eval["anomalia_score"] = modelo.decision_function(X)
        df_eval["prediccion_modelo"] = modelo.predict(X)
        df_eval["es_anomalia_modelo"] = (df_eval["prediccion_modelo"] == -1) & (~df_eval["es_free_game"])

    # Calcular umbrales y patrones para evaluar_anomalia
    ratio_mean = df_eval['ratio_ganancia'].mean()
    ratio_std = df_eval['ratio_ganancia'].std()
    bet_mean = df_eval['TotalBet'].mean()
    bet_std = df_eval['TotalBet'].std()
    umbral_ratio = ratio_mean + (3 * ratio_std)
    umbral_bet = bet_mean + (3 * bet_std)

    conteo_junto, conteo_chispeado = analizar_patron_ganancias_altas(df_eval)
    patron_por_juego, _, _ = analizar_patron_jackpots(df_eval)

    # Filtrar anomalías detectadas por el modelo usando las reglas de negocio (evita falsos positivos en ratios bajos como x2, x3)
    df_eval["es_anomalia_modelo"] = df_eval.apply(
        lambda row: row["es_anomalia_modelo"] and evaluar_anomalia(
            row, umbral_ratio, umbral_bet, conteo_junto, conteo_chispeado, patron_por_juego
        ) if row["es_anomalia_modelo"] else False,
        axis=1
    )

    df_eval["es_anomalia_regla"] = df_eval.apply(marcar_anomalia_por_reglas, axis=1)

    q10 = df_eval["anomalia_score"].quantile(0.10)
    q25 = df_eval["anomalia_score"].quantile(0.25)

    df_eval["nivel_riesgo_modelo"] = df_eval["anomalia_score"].apply(
        lambda score: nivel_riesgo_por_score(score, q10, q25)
    )

    total = len(df_eval)
    anom_modelo = int(df_eval["es_anomalia_modelo"].sum())
    anom_regla = int(df_eval["es_anomalia_regla"].sum())

    coincidencias = int(
        ((df_eval["es_anomalia_modelo"] == True) &
         (df_eval["es_anomalia_regla"] == True)).sum()
    )

    solo_modelo = int(
        ((df_eval["es_anomalia_modelo"] == True) &
         (df_eval["es_anomalia_regla"] == False)).sum()
    )

    solo_reglas = int(
        ((df_eval["es_anomalia_modelo"] == False) &
         (df_eval["es_anomalia_regla"] == True)).sum()
    )

    print(f"Total de registros evaluados: {total}")
    print(f"Anomalías detectadas por el modelo: {anom_modelo} ({anom_modelo / total * 100:.2f}%)")
    print(f"Anomalías marcadas por reglas: {anom_regla} ({anom_regla / total * 100:.2f}%)")
    print(f"Coincidencias modelo + reglas: {coincidencias}")
    print(f"Casos detectados solo por el modelo: {solo_modelo}")
    print(f"Casos detectados solo por reglas: {solo_reglas}")

    # ==========================================================
    # CUADRO DE MÉTRICAS DEL MODELO NO SUPERVISADO
    # ==========================================================
    # Importante:
    # Aquí NO estamos comparando contra una etiqueta humana real.
    # Estamos comparando la predicción del modelo contra reglas de negocio
    # usadas como referencia técnica.

    y_regla = df_eval["es_anomalia_regla"].astype(int)
    y_modelo = df_eval["es_anomalia_modelo"].astype(int)

    print("\n" + "=" * 70)
    print("CUADRO DE MÉTRICAS — MODELO NO SUPERVISADO VS REGLAS")
    print("=" * 70)

    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_regla,
            y_modelo,
            labels=[0, 1],
            target_names=["normal", "anomalia"],
            digits=4,
            zero_division=0
        )
    )

    acc = accuracy_score(y_regla, y_modelo)
    bal_acc = balanced_accuracy_score(y_regla, y_modelo)

    print(f"Accuracy general: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Balanced accuracy: {bal_acc:.4f} ({bal_acc * 100:.2f}%)")

    matriz = confusion_matrix(y_regla, y_modelo, labels=[0, 1])

    matriz_df = pd.DataFrame(
        matriz,
        index=["Regla: normal", "Regla: anomalía"],
        columns=["Modelo: normal", "Modelo: anomalía"]
    )

    print("\nMatriz de confusión:")
    print(matriz_df)

    print("\nDistribución de riesgo según modelo:")
    print(df_eval["nivel_riesgo_modelo"].value_counts())

    print("\nTop 10 registros más anómalos según el modelo:")
    columnas_mostrar = [
        "PlayerId",
        "EventTime",
        "TotalBet",
        "TotalWin",
        "TotalJPWin",
        "BalanceChange",
        "ratio_ganancia",
        "anomalia_score",
        "nivel_riesgo_modelo",

        "es_anomalia_modelo",
        "es_anomalia_regla"
    ]

    columnas_existentes = [c for c in columnas_mostrar if c in df_eval.columns]

    print(
        df_eval[columnas_existentes]
        .sort_values("anomalia_score", ascending=True)
        .head(10)
    )

    df_eval.to_csv(salida_csv, index=False)
    print(f"\nArchivo generado: {salida_csv}")
    print("=" * 70)

    return df_eval