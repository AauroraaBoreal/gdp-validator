import pandas as pd
from datetime import datetime

def generar_reporte_whatsapp(df):
    player_id = str(df['PlayerId'].iloc[0])
    balance_inicial = df['BalanceStart'].iloc[0]
    balance_final = df['Balance'].iloc[-1]
    fecha_inicio = df['EventTime'].min().strftime('%d/%m/%Y')
    fecha_fin = df['EventTime'].max().strftime('%d/%m/%Y')

    # Juegos involucrados
    juegos = df['GameId'].unique().tolist()
    juegos_str = ', '.join(juegos)

    # Total apostado
    total_apostado = df['TotalBet'].sum()

    # Detectar tipo de ganancia principal
    jackpots = df[(df['TotalJPWin'].notna()) & (df['TotalJPWin'] > 0)]
    ganancias_altas = df[df['clasificacion'] == 'ganancia_alta']

    if not jackpots.empty:
        jp = jackpots.iloc[0]
        tipo_ganancia = f"obtuvo un jackpot de ${jp['TotalJPWin']:,.2f} MXN en el juego {jp['GameId']}"
    elif not ganancias_altas.empty:
        ganancia_max = ganancias_altas.loc[ganancias_altas['TotalWin'].idxmax()]
        tipo_ganancia = f"su ganancia más alta fue de ${ganancia_max['TotalWin']:,.2f} MXN con una apuesta de ${ganancia_max['TotalBet']:,.2f} MXN (ratio x{round(ganancia_max['ratio_ganancia'], 1)})"
    else:
        ganancia_total = df['TotalWin'].sum()
        tipo_ganancia = f"acumuló un total de ganancias de ${ganancia_total:,.2f} MXN"

    reporte = (
        f"El usuario {player_id} tenía un balance inicial de ${balance_inicial:,.2f} MXN, "
        f"con apuestas de ${total_apostado:,.2f} MXN en el/los juego(s) {juegos_str}, "
        f"modificando su balance a ${balance_final:,.2f} MXN. "
        f"El cliente {tipo_ganancia}."
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
        lineas.append(f"\n🔴 Tipo: {row['tipo_anomalia']}")
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