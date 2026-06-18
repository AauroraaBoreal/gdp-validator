import pandas as pd
import numpy as np
from datetime import datetime

def obtener_formato_moneda(df_or_row):
    """
    Retorna el símbolo y el nombre de la moneda a partir de la columna/campo 'Currency'.
    Por defecto retorna ('$', 'MXN').
    """
    if isinstance(df_or_row, pd.DataFrame):
        if 'Currency' in df_or_row.columns and len(df_or_row) > 0:
            curr = str(df_or_row['Currency'].iloc[0]).strip().upper()
        else:
            curr = 'MXN'
    elif isinstance(df_or_row, (dict, pd.Series)):
        curr = str(df_or_row.get('Currency', 'MXN')).strip().upper()
    else:
        curr = 'MXN'

    if curr == 'PEN':
        return 'S/', 'PEN'
    else:
        return '$', 'MXN'

def buscar_retiro(df, monto_retiro):
    """
    Busca el retiro más cercano al monto ingresado usando búsqueda
    lineal vectorizada con NumPy — O(n), óptima para datos sin orden garantizado.
    
    Fórmula: Balance[i] - Balance[i+1] + BalanceChange[i+1] = monto retiro
    """
    balance = df['Balance'].values
    balance_change = df['BalanceChange'].values

    # Calcular diferencias vectorialmente para todas las filas consecutivas
    # Formula: -(Balance[i] - Balance[i+1] + BalanceChange[i+1])
    diferencias = -(balance[:-1] - balance[1:] + balance_change[1:])

    # Buscar el índice con diferencia más cercana al monto ingresado
    # Solo cuenta retiros de evento 0 o del último evento de free games (EventId != '0' pero el siguiente es '0' o cambia de juego)
    event_ids = df['EventId'].astype(str).str.strip().values
    mismo_game = (df['GameInstanceId'].values[:-1] == df['GameInstanceId'].values[1:])

    cond_inicio_cero = (event_ids[:-1] == '0')
    es_free_game_inicio = (event_ids[:-1] != '0')
    siguiente_diferente = ~mismo_game
    siguiente_cero = (event_ids[1:] == '0')
    cond_ultimo_free = es_free_game_inicio & (siguiente_diferente | siguiente_cero)

    mask_permitido = cond_inicio_cero | cond_ultimo_free
    mask_retiros = (diferencias < -0.01) & mask_permitido

    diff_abs = np.full(len(diferencias), np.inf)
    if np.any(mask_retiros):
        diff_abs[mask_retiros] = np.abs(diferencias[mask_retiros] + float(monto_retiro))

    idx_min = np.argmin(diff_abs)

    # Si la diferencia más cercana supera la tolerancia de 1000 pesos, consideramos que no se encontró
    if diff_abs[idx_min] > 1000.0:
        return None, None

    # idx_min corresponde a la fila i, el retiro ocurre en i+1
    idx_retiro = idx_min + 1
    monto_encontrado = diferencias[idx_min]

    return idx_retiro, monto_encontrado

def generar_reporte_whatsapp(df, monto_validar=None, idx_forzado=None):
    player_id = str(df['PlayerId'].iloc[0])
    simbolo, moneda = obtener_formato_moneda(df)

    if idx_forzado is not None:
        idx_evento = idx_forzado
        balance = df['Balance'].values
        balance_change = df['BalanceChange'].values
        monto_encontrado = -(balance[idx_evento - 1] - balance[idx_evento] + balance_change[idx_evento])
    elif monto_validar is not None and float(monto_validar) > 0:
        idx_evento, monto_encontrado = buscar_retiro(df, monto_validar)
        if idx_evento is None:
            raise ValueError(f"No se encontró ningún retiro cercano a {simbolo}{float(monto_validar):,.2f} {moneda} en el archivo.")
    else:
        balance = df['Balance'].values
        balance_change = df['BalanceChange'].values
        diferencias = -(balance[:-1] - balance[1:] + balance_change[1:])

        # Solo cuenta retiros de evento 0 o del último evento de free games
        event_ids = df['EventId'].astype(str).str.strip().values
        mismo_game = (df['GameInstanceId'].values[:-1] == df['GameInstanceId'].values[1:])

        cond_inicio_cero = (event_ids[:-1] == '0')
        es_free_game_inicio = (event_ids[:-1] != '0')
        siguiente_diferente = ~mismo_game
        siguiente_cero = (event_ids[1:] == '0')
        cond_ultimo_free = es_free_game_inicio & (siguiente_diferente | siguiente_cero)

        mask_permitido = cond_inicio_cero | cond_ultimo_free
        mask_retiros = (diferencias < -0.01) & mask_permitido

        if np.any(mask_retiros):
            diferencias_filtradas = np.full(len(diferencias), np.inf)
            diferencias_filtradas[mask_retiros] = diferencias[mask_retiros]
            idx_min = np.argmin(diferencias_filtradas)
            idx_evento = idx_min + 1
            monto_encontrado = diferencias[idx_min]
        else:
            idx_evento = 1
            monto_encontrado = 0.0

    evento_principal = df.loc[idx_evento]

    # Última recarga antes del retiro (recarga es cuando la diferencia > 0)
    balance = df['Balance'].values
    balance_change = df['BalanceChange'].values
    diferencias_todas = -(balance[:-1] - balance[1:] + balance_change[1:])
    indices_recargas = np.where(diferencias_todas[:idx_evento] > 0)[0]
    idx_recarga = indices_recargas[-1] + 1 if len(indices_recargas) > 0 else 0

    tramo = df.loc[idx_recarga:idx_evento]
    balance_inicial = tramo['BalanceStart'].iloc[0]
    fila_inicio = int(df.loc[idx_recarga, '_fila_csv']) if '_fila_csv' in df.columns else idx_recarga + 1
    if '_fila_csv' in df.columns:
        fila_antes = int(df.loc[idx_evento - 1, '_fila_csv']) if (idx_evento - 1) in df.index else int(df.loc[idx_evento, '_fila_csv']) - 1
        fila_despues = int(df.loc[idx_evento, '_fila_csv'])
        if fila_despues - fila_antes > 1:
            fila_retiro = fila_antes + 1
        else:
            fila_retiro = fila_despues
    else:
        fila_retiro = idx_evento + 1

    # Balance justo antes del retiro (fila anterior al retiro)
    idx_anterior = idx_evento - 1
    balance_antes_retiro = df.loc[idx_anterior, 'Balance'] if idx_anterior >= 0 else balance_inicial

    # Top 3 apuestas más altas únicas en el tramo
    top_apuestas = (
        tramo[tramo['TotalBet'] > 0]
        .drop_duplicates(subset=['TotalBet'])
        .nlargest(3, 'TotalBet')['TotalBet']
    )
    apuestas_str = ', '.join([f"{simbolo}{v:,.2f}" for v in top_apuestas]) if not top_apuestas.empty else "sin apuestas registradas"

    # Top 3 juegos más jugados en el tramo
    juegos_str = ', '.join(tramo['GameId'].value_counts().head(3).index.tolist())

    # Jackpot o ganancia alta en el tramo
    comentario_extra = ""
    jackpots_tramo = tramo[(tramo['TotalJPWin'].notna()) & (tramo['TotalJPWin'] > 0)]
    if not jackpots_tramo.empty:
        jp = jackpots_tramo.loc[jackpots_tramo['TotalJPWin'].idxmax()]
        comentario_extra = (
            f" El cliente obtuvo un jackpot de "
            f"{simbolo}{jp['TotalJPWin']:,.2f} {moneda} con una apuesta de "
            f"{simbolo}{jp['TotalBet']:,.2f} {moneda} en el juego {jp['GameId']}."
        )
    else:
        ganancias_altas = tramo[tramo['ratio_ganancia'] >= 26]
        if not ganancias_altas.empty:
            ga = ganancias_altas.loc[ganancias_altas['TotalWin'].idxmax()]
            comentario_extra = (
                f" El cliente obtuvo una ganancia de "
                f"{simbolo}{ga['TotalWin']:,.2f} {moneda} con una apuesta de "
                f"{simbolo}{ga['TotalBet']:,.2f} {moneda} en el juego {ga['GameId']}."
            )

    # Observación free games
    observacion_fg = ""
    if 'es_free_game_inusual' in df.columns:
        from modules.modulo3_anomalias import obtener_observaciones_free_games
        obs = obtener_observaciones_free_games(df)
        if obs:
            observacion_fg = f" Observación: {obs}"

    reporte = (
        f"El usuario {player_id} tenía un balance inicial de {simbolo}{balance_inicial:,.2f} {moneda} "
        f"(fila {fila_inicio}), con apuestas de {apuestas_str} {moneda} "
        f"en los juegos {juegos_str}, "
        f"alcanzó un balance de {simbolo}{balance_antes_retiro:,.2f} {moneda} antes del retiro. "
        f"El cliente realizó un retiro de {simbolo}{monto_encontrado:,.2f} {moneda} "
        f"(fila {fila_retiro}).{comentario_extra}{observacion_fg}"
    )

    return reporte

def generar_reporte_qa(df_anomalias):
    if df_anomalias.empty:
        return None

    simbolo, moneda = obtener_formato_moneda(df_anomalias)
    lineas = []
    lineas.append("⚠️ REPORTE DE ALERTA - ÁREA DE QA")
    from datetime import timezone, timedelta
    tz_peru = timezone(timedelta(hours=-5))
    lineas.append(f"Fecha de generación: {datetime.now(tz_peru).strftime('%d/%m/%Y %H:%M')}")
    lineas.append(f"Total de registros sospechosos: {len(df_anomalias)}")
    lineas.append("-" * 50)

    for _, row in df_anomalias.iterrows():
        fila_numero = row.name + 1
        lineas.append(f"\n🔴 Tipo: {row['tipo_anomalia']}")
        lineas.append(f"   Fila en CSV      : {fila_numero}")
        lineas.append(f"   GameInstanceID : {row.get('GameInstanceId', 'N/A')}")
        lineas.append(f"   Fecha y hora   : {row['EventTime']}")
        lineas.append(f"   Juego          : {row.get('GameId', 'N/A')}")
        lineas.append(f"   Apuesta        : {simbolo}{row['TotalBet']:,.2f} {moneda}")
        lineas.append(f"   Ganancia       : {simbolo}{row['TotalWin']:,.2f} {moneda}")
        if pd.notna(row.get('TotalJPWin')) and row['TotalJPWin'] > 0:
            lineas.append(f"   Jackpot        : {simbolo}{row['TotalJPWin']:,.2f} {moneda}")
        lineas.append(f"   Ratio          : x{round(row['ratio_ganancia'], 2)}")
        lineas.append(f"   Score anomalía : {round(row['anomalia_score'], 4)}")
        if pd.notna(row.get('razon_anomalia')):
            lineas.append(f"   Razón          : {row['razon_anomalia']}")

    return '\n'.join(lineas)

def mostrar_reportes(df):
    anomalias = df[df['es_anomalia'] == True]
    hay_anomalias = len(anomalias) > 0

    print("\n" + "=" * 60)

    if hay_anomalias:
        resultado = 'sospechoso'
        print("RESULTADO: ⚠️  SE DETECTARON ANOMALÍAS")
        print("=" * 60)
        print("\n📋 REPORTE WHATSAPP:")
        print("-" * 60)
        reporte_whatsapp = generar_reporte_whatsapp(df)
        print(reporte_whatsapp)
        print("\n\n📋 REPORTE PARA QA:")
        print("-" * 60)
        reporte_qa = generar_reporte_qa(anomalias)
        print(reporte_qa)
    else:
        resultado = 'normal'
        print("RESULTADO: ✅  VALIDACIÓN NORMAL")
        print("=" * 60)
        print("\n📋 REPORTE WHATSAPP:")
        print("-" * 60)
        reporte_whatsapp = generar_reporte_whatsapp(df)
        print(reporte_whatsapp)
        reporte_qa = None

    print("\n" + "=" * 60)
    return resultado, reporte_whatsapp, reporte_qa