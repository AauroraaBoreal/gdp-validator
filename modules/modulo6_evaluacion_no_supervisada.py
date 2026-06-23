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
    is_ae = isinstance(modelo, dict) and 'model' in modelo
    is_rf = not is_ae and hasattr(modelo, 'predict_proba')

    if is_ae:
        model_name = "AUTOENCODER"
    elif is_rf:
        model_name = "RANDOM FOREST"
    else:
        model_name = "ISOLATION FOREST"

    print("\n" + "=" * 70)
    print(f"EVALUACIÓN NO SUPERVISADA - {model_name}")
    print("=" * 70)

    from modules.modulo3_anomalias import marcar_free_games
    df_eval = df.copy()
    df_eval = marcar_free_games(df_eval)

    X = preparar_features_anomalias(df_eval)

    # Asegurar compatibilidad de features con el modelo cargado
    if is_ae:
        columnas_modelo = modelo.get('features', [])
    elif hasattr(modelo, 'feature_names_in_'):
        columnas_modelo = list(modelo.feature_names_in_)
    else:
        columnas_modelo = None

    if columnas_modelo:
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
        if is_ae:
            scaler = modelo['scaler']
            modelo_ae = modelo['model']
            threshold = modelo['threshold']
            X_scaled = scaler.transform(X)
            X_pred = modelo_ae.predict(X_scaled)
            reconstruction_error = np.mean((X_scaled - X_pred) ** 2, axis=1)
            df_eval["anomalia_score"] = threshold - reconstruction_error
            df_eval["es_anomalia_modelo"] = (reconstruction_error > threshold) & (~df_eval["es_free_game"])
        elif is_rf:
            prob_anomaly = modelo.predict_proba(X)[:, 1]
            df_eval["anomalia_score"] = 0.5 - prob_anomaly
            df_eval["es_anomalia_modelo"] = (prob_anomaly > 0.5) & (~df_eval["es_free_game"])
        else:
            df_eval["anomalia_score"] = modelo.decision_function(X)
            df_eval["prediccion_modelo"] = modelo.predict(X)
            df_eval["es_anomalia_modelo"] = (df_eval["prediccion_modelo"] == -1) & (~df_eval["es_free_game"])
    except Exception as e:
        print(f"Advertencia: Error al usar el modelo en evaluación ({e}). Re-entrenando detector...")
        if is_ae:
            from modules.modulo3_autoencoder import entrenar_autoencoder
            modelo = entrenar_autoencoder(df_eval)
            scaler = modelo['scaler']
            modelo_ae = modelo['model']
            threshold = modelo['threshold']
            X = preparar_features_anomalias(df_eval)
            X = X[modelo.get('features', [])]
            X_scaled = scaler.transform(X)
            X_pred = modelo_ae.predict(X_scaled)
            reconstruction_error = np.mean((X_scaled - X_pred) ** 2, axis=1)
            df_eval["anomalia_score"] = threshold - reconstruction_error
            df_eval["es_anomalia_modelo"] = (reconstruction_error > threshold) & (~df_eval["es_free_game"])
        elif is_rf:
            from modules.modulo3_random_forest import entrenar_random_forest
            modelo = entrenar_random_forest(df_eval)
            if hasattr(modelo, 'feature_names_in_'):
                X = X[list(modelo.feature_names_in_)]
            prob_anomaly = modelo.predict_proba(X)[:, 1]
            df_eval["anomalia_score"] = 0.5 - prob_anomaly
            df_eval["es_anomalia_modelo"] = (prob_anomaly > 0.5) & (~df_eval["es_free_game"])
        else:
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
    y_regla = df_eval["es_anomalia_regla"].astype(int)
    y_modelo = df_eval["es_anomalia_modelo"].astype(int)

    print("\n" + "=" * 70)
    print(f"CUADRO DE MÉTRICAS — {model_name} VS REGLAS")
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
        columns=[f"{model_name}: normal", f"{model_name}: anomalía"]
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

    if salida_csv:
        df_eval.to_csv(salida_csv, index=False)
        print(f"\nArchivo generado: {salida_csv}")
    print("=" * 70)

    return df_eval


def comparar_modelos_no_supervisados(df, modelo_if, modelo_ae, modelo_rf, salida_csv="resultados_evaluacion_no_supervisada.csv"):
    from sklearn.metrics import precision_recall_fscore_support

    # Evaluar los tres modelos sin guardar CSV individualmente
    df_if = evaluar_detector_no_supervisado(df, modelo_if, salida_csv=None)
    df_ae = evaluar_detector_no_supervisado(df, modelo_ae, salida_csv=None)
    df_rf = evaluar_detector_no_supervisado(df, modelo_rf, salida_csv=None)

    y_regla = df_if["es_anomalia_regla"].astype(int)
    y_if = df_if["es_anomalia_modelo"].astype(int)
    y_ae = df_ae["es_anomalia_modelo"].astype(int)
    y_rf = df_rf["es_anomalia_modelo"].astype(int)

    acc_if = accuracy_score(y_regla, y_if)
    acc_ae = accuracy_score(y_regla, y_ae)
    acc_rf = accuracy_score(y_regla, y_rf)

    bal_if = balanced_accuracy_score(y_regla, y_if)
    bal_ae = balanced_accuracy_score(y_regla, y_ae)
    bal_rf = balanced_accuracy_score(y_regla, y_rf)

    p_if, r_if, f_if, _ = precision_recall_fscore_support(y_regla, y_if, average='binary', pos_label=1, zero_division=0)
    p_ae, r_ae, f_ae, _ = precision_recall_fscore_support(y_regla, y_ae, average='binary', pos_label=1, zero_division=0)
    p_rf, r_rf, f_rf, _ = precision_recall_fscore_support(y_regla, y_rf, average='binary', pos_label=1, zero_division=0)

    anom_if = int(y_if.sum())
    anom_ae = int(y_ae.sum())
    anom_rf = int(y_rf.sum())
    anom_reglas = int(y_regla.sum())

    print("\n" + "=" * 90)
    print("RESUMEN COMPARATIVO DE RENDIMIENTO (CONTRA REGLAS DE NEGOCIO)")
    print("=" * 90)
    print(f"{'Métrica':<28} | {'Isolation Forest':<18} | {'Autoencoder':<15} | {'Random Forest':<15}")
    print("-" * 90)
    print(f"{'Accuracy General':<28} | {acc_if:<18.4f} | {acc_ae:<15.4f} | {acc_rf:<15.4f}")
    print(f"{'Balanced Accuracy':<28} | {bal_if:<18.4f} | {bal_ae:<15.4f} | {bal_rf:<15.4f}")
    print(f"{'Precisión (Anomalías)':<28} | {p_if:<18.4f} | {p_ae:<15.4f} | {p_rf:<15.4f}")
    print(f"{'Recall (Anomalías)':<28} | {r_if:<18.4f} | {r_ae:<15.4f} | {r_rf:<15.4f}")
    print(f"{'F1-Score (Anomalías)':<28} | {f_if:<18.4f} | {f_ae:<15.4f} | {f_rf:<15.4f}")
    print(f"{'Anomalías Detectadas':<28} | {anom_if:<18} | {anom_ae:<15} | {anom_rf:<15}")
    print(f"{'Anomalías por Reglas':<28} | {anom_reglas:<18} | {anom_reglas:<15} | {anom_reglas:<15}")
    print("=" * 90)

    # Determinar recomendación
    metricas = {
        'Isolation Forest': f_if,
        'Autoencoder': f_ae,
        'Random Forest': f_rf
    }
    mejor_modelo = max(metricas, key=metricas.get)
    empates = [m for m, f in metricas.items() if f == metricas[mejor_modelo]]
    
    if len(empates) > 1:
        bal_metricas = {
            'Isolation Forest': bal_if,
            'Autoencoder': bal_ae,
            'Random Forest': bal_rf
        }
        mejor_bal = max(empates, key=lambda x: bal_metricas[x])
        empates_bal = [m for m in empates if bal_metricas[m] == bal_metricas[mejor_bal]]
        if len(empates_bal) > 1:
            print(f"\n🏆 RECOMENDACIÓN: Múltiples modelos empatan con el mejor rendimiento: {', '.join(empates_bal)}.")
        else:
            print(f"\n🏆 RECOMENDACIÓN: El modelo de {mejor_bal} tiene el mejor rendimiento en Balanced Accuracy dentro del empate.")
    else:
        print(f"\n🏆 RECOMENDACIÓN: El modelo de {mejor_modelo} tiene el mejor F1-score.")
    print("=" * 90 + "\n")

    if salida_csv:
        df_comparativo = df_if.copy()
        df_comparativo["es_anomalia_ae"] = df_ae["es_anomalia_modelo"]
        df_comparativo["anomalia_score_ae"] = df_ae["anomalia_score"]
        df_comparativo["es_anomalia_rf"] = df_rf["es_anomalia_modelo"]
        df_comparativo["anomalia_score_rf"] = df_rf["anomalia_score"]
        df_comparativo.to_csv(salida_csv, index=False)
        print(f"Archivo comparativo generado: {salida_csv}")

    return df_if, df_ae, df_rf