import pandas as pd
import numpy as np
from datetime import datetime

def buscar_retiro(df, monto_retiro):
    """
    Busca el retiro más cercano al monto ingresado usando búsqueda
    lineal vectorizada con NumPy — O(n), óptima para datos sin orden garantizado.
    
    Fórmula: Balance[i] - Balance[i+1] + BalanceChange[i+1] = monto retiro
    """
    balance = df['Balance'].values
    balance_change = df['BalanceChange'].values

    # Calcular diferencias vectorialmente para todas las filas consecutivas
    diferencias = -(balance[:-1] - balance[1:] + balance_change[1:])

    # Buscar el índice con diferencia más cercana al monto ingresado
    diff_abs = np.abs(diferencias - float(monto_retiro))
    idx_min = np.argmin(diff_abs)

    # idx_min corresponde a la fila i, el retiro ocurre en i+1
    idx_retiro = idx_min + 1
    monto_encontrado = diferencias[idx_min]

    return idx_retiro, monto_encontrado

def generar_reporte_whatsapp(df, monto_validar=None, idx_forzado=None):
    player_id = str(df['PlayerId'].iloc[0])

    if idx_forzado is not None:
        idx_evento = idx_forzado
        balance = df['Balance'].values
        balance_change = df['BalanceChange'].values
        monto_encontrado = -(balance[idx_evento - 1] - balance[idx_evento] + balance_change[idx_evento])
    elif monto_validar is not None and float(monto_validar) > 0:
        idx_evento, monto_encontrado = buscar_retiro(df, monto_validar)
    else:
        balance = df['Balance'].values
        balance_change = df['BalanceChange'].values
        diferencias = -(balance[:-1] - balance[1:] + balance_change[1:])
        idx_min = np.argmax(diferencias)
        idx_evento = idx_min + 1
        monto_encontrado = diferencias[idx_min]

    evento_principal = df.loc[idx_evento]

    # Última recarga antes del retiro
    recargas = df[(df['clasificacion'] == 'recarga') & (df.index < idx_evento)]
    idx_recarga = recargas.index[-1] if not recargas.empty else 0

    tramo = df.loc[idx_recarga:idx_evento]
    balance_inicial = tramo['BalanceStart'].iloc[0]
    fila_inicio = idx_recarga + 1
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
    apuestas_str = ', '.join([f"${v:,.2f}" for v in top_apuestas]) if not top_apuestas.empty else "sin apuestas registradas"

    # Top 3 juegos más jugados en el tramo
    juegos_str = ', '.join(tramo['GameId'].value_counts().head(3).index.tolist())

    # Jackpot o ganancia alta en el tramo
    comentario_extra = ""
    jackpots_tramo = tramo[(tramo['TotalJPWin'].notna()) & (tramo['TotalJPWin'] > 0)]
    if not jackpots_tramo.empty:
        jp = jackpots_tramo.loc[jackpots_tramo['TotalJPWin'].idxmax()]
        comentario_extra = (
            f" El cliente obtuvo un jackpot de "
            f"${jp['TotalJPWin']:,.2f} MXN con una apuesta de "
            f"${jp['TotalBet']:,.2f} MXN en el juego {jp['GameId']}."
        )
    else:
        ganancias_altas = tramo[tramo['clasificacion'] == 'ganancia_alta']
        if not ganancias_altas.empty:
            ga = ganancias_altas.loc[ganancias_altas['TotalWin'].idxmax()]
            comentario_extra = (
                f" El cliente obtuvo una ganancia de "
                f"${ga['TotalWin']:,.2f} MXN con una apuesta de "
                f"${ga['TotalBet']:,.2f} MXN en el juego {ga['GameId']}."
            )

    # Observación free games
    observacion_fg = ""
    if 'es_free_game_inusual' in df.columns:
        from modules.modulo3_anomalias import obtener_observaciones_free_games
        obs = obtener_observaciones_free_games(df)
        if obs:
            observacion_fg = f" Observación: {obs}"

    reporte = (
        f"El usuario {player_id} tenía un balance inicial de ${balance_inicial:,.2f} MXN "
        f"(fila {fila_inicio}), con apuestas de {apuestas_str} MXN "
        f"en los juegos {juegos_str}, "
        f"alcanzó un balance de ${balance_antes_retiro:,.2f} MXN antes del retiro. "
        f"El cliente realizó un retiro de ${monto_encontrado:,.2f} MXN "
        f"(fila {fila_retiro}).{comentario_extra}{observacion_fg}"
    )

    return reporte

def generar_reporte_qa(df_anomalias):
    if df_anomalias.empty:
        return None

    lineas = []
    lineas.append("⚠️ REPORTE DE ALERTA - ÁREA DE QA")
    lineas.append(f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    lineas.append(f"Total de registros sospechosos: {len(df_anomalias)}")
    lineas.append("-" * 50)

    for _, row in df_anomalias.iterrows():
        fila_numero = row.name + 1
        lineas.append(f"\n🔴 Tipo: {row['tipo_anomalia']}")
        lineas.append(f"   Fila en CSV      : {fila_numero}")
        lineas.append(f"   GameInstanceID : {row.get('GameInstanceId', 'N/A')}")
        lineas.append(f"   Fecha y hora   : {row['EventTime']}")
        lineas.append(f"   Juego          : {row.get('GameId', 'N/A')}")
        lineas.append(f"   Apuesta        : ${row['TotalBet']:,.2f} MXN")
        lineas.append(f"   Ganancia       : ${row['TotalWin']:,.2f} MXN")
        if pd.notna(row.get('TotalJPWin')) and row['TotalJPWin'] > 0:
            lineas.append(f"   Jackpot        : ${row['TotalJPWin']:,.2f} MXN")
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