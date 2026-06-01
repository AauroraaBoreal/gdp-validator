import pandas as pd
from datetime import datetime

def generar_reporte_whatsapp(df):
    player_id = str(df['PlayerId'].iloc[0])

    # Encontrar la ganancia alta o jackpot principal
    jackpots = df[(df['TotalJPWin'].notna()) & (df['TotalJPWin'] > 0)]
    ganancias_altas = df[df['clasificacion'] == 'ganancia_alta']

    if not jackpots.empty:
        evento_principal = jackpots.loc[jackpots['TotalJPWin'].idxmax()]
    elif not ganancias_altas.empty:
        evento_principal = ganancias_altas.loc[ganancias_altas['TotalWin'].idxmax()]
    else:
        evento_principal = df.loc[df['TotalWin'].idxmax()]

    idx_evento = evento_principal.name

    # Encontrar la última recarga antes del evento principal
    recargas = df[(df['clasificacion'] == 'recarga') & (df.index < idx_evento)]

    if not recargas.empty:
        idx_recarga = recargas.index[-1]
    else:
        idx_recarga = 0

    # Tramo relevante: desde la última recarga hasta el evento principal
    tramo = df.loc[idx_recarga:idx_evento]

    balance_inicial = tramo['BalanceStart'].iloc[0]
    balance_final = evento_principal['Balance']
    total_apostado = tramo['TotalBet'].sum()

    # Juegos más jugados en el tramo
    juegos_conteo = tramo['GameId'].value_counts()
    juegos_principales = juegos_conteo.head(3).index.tolist()
    juegos_str = ', '.join(juegos_principales)

    # Rango de filas del tramo
    fila_inicio = idx_recarga + 1
    fila_fin = idx_evento + 1

    # Tipo de ganancia
    if not jackpots.empty:
        tipo_ganancia = f"obtuvo un jackpot de ${evento_principal['TotalJPWin']:,.2f} MXN en el juego {evento_principal['GameId']}"
    else:
        tipo_ganancia = (
            f"su ganancia más alta fue de ${evento_principal['TotalWin']:,.2f} MXN "
            f"con una apuesta de ${evento_principal['TotalBet']:,.2f} MXN "
            f"(ratio x{round(evento_principal['ratio_ganancia'], 1)}) "
            f"en el juego {evento_principal['GameId']}"
        )

    reporte = (
        f"El usuario {player_id} tenía un balance inicial de ${balance_inicial:,.2f} MXN "
        f"(fila {fila_inicio}), con apuestas de ${total_apostado:,.2f} MXN "
        f"en los juegos principales: {juegos_str}, "
        f"modificando su balance a ${balance_final:,.2f} MXN (fila {fila_fin}). "
        f"El cliente {tipo_ganancia}."
    )

    # Observación de free games inusuales
    observacion_fg = ""
    if 'es_free_game_inusual' in df.columns:
        from modules.modulo3_anomalias import obtener_observaciones_free_games
        obs = obtener_observaciones_free_games(df)
        if obs:
            observacion_fg = f" Observación: {obs}"

    reporte = (
        f"El usuario {player_id} tenía un balance inicial de ${balance_inicial:,.2f} MXN "
        f"(fila {fila_inicio}), con apuestas de ${total_apostado:,.2f} MXN "
        f"en los juegos principales: {juegos_str}, "
        f"modificando su balance a ${balance_final:,.2f} MXN (fila {fila_fin}). "
        f"El cliente {tipo_ganancia}.{observacion_fg}"
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
        fila_numero = row.name + 1  # +1 porque el índice empieza en 0
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

    return '\n'.join(lineas)

def mostrar_reportes(df):
    anomalias = df[df['es_anomalia'] == True]
    hay_anomalias = len(anomalias) > 0

    print("\n" + "=" * 60)

    if hay_anomalias:
        resultado = 'sospechoso'
        print("RESULTADO: ⚠️  SE DETECTARON ANOMALÍAS")
        print("=" * 60)

        print("\n📋 REPORTE WHATSAPP (validación general):")
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