import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_engine():
    return create_engine(DATABASE_URL)

def test_conexion():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

def crear_tablas():
    if not test_conexion():
        print("⚠️  Sin conexión a Supabase — operación omitida")
        return
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS jugadores (
                player_id VARCHAR PRIMARY KEY,
                currency VARCHAR,
                apuesta_promedio DECIMAL,
                apuesta_max DECIMAL,
                ganancia_max DECIMAL,
                ratio_promedio DECIMAL,
                total_sesiones INTEGER DEFAULT 0,
                ultima_actualizacion TIMESTAMP
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS validaciones (
                validacion_id SERIAL PRIMARY KEY,
                player_id VARCHAR REFERENCES jugadores(player_id),
                fecha_inicio TIMESTAMP,
                fecha_fin TIMESTAMP,
                total_registros INTEGER,
                resultado VARCHAR,
                reporte_whatsapp TEXT,
                fecha_procesamiento TIMESTAMP DEFAULT NOW()
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS anomalias (
                anomalia_id SERIAL PRIMARY KEY,
                validacion_id INTEGER REFERENCES validaciones(validacion_id),
                game_instance_id VARCHAR,
                event_time TIMESTAMP,
                game_id VARCHAR,
                total_bet DECIMAL,
                total_win DECIMAL,
                total_jp_win DECIMAL,
                tipo_anomalia VARCHAR,
                descripcion TEXT,
                score_anomalia DECIMAL
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS modelo_versiones (
                version_id SERIAL PRIMARY KEY,
                fecha_actualizacion TIMESTAMP DEFAULT NOW(),
                player_id VARCHAR,
                registros_nuevos INTEGER,
                accuracy_post DECIMAL
            );
        """))
        conn.commit()
    print("Tablas creadas correctamente en Supabase")

def guardar_jugador(df):
    if not test_conexion():
        print("⚠️  Sin conexión a Supabase — jugador no guardado")
        return
    engine = get_engine()
    player_id = str(df['PlayerId'].iloc[0])
    currency = str(df['Currency'].iloc[0]) if 'Currency' in df.columns else 'MXN'
    apuesta_promedio = float(round(df['TotalBet'].mean(), 2))
    apuesta_max = float(round(df['TotalBet'].max(), 2))
    ganancia_max = float(round(df['TotalWin'].max(), 2))
    ratio_promedio = float(round(df['ratio_ganancia'].mean(), 2))
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO jugadores (player_id, currency, apuesta_promedio, apuesta_max, ganancia_max, ratio_promedio, total_sesiones, ultima_actualizacion)
            VALUES (:player_id, :currency, :apuesta_promedio, :apuesta_max, :ganancia_max, :ratio_promedio, 1, NOW())
            ON CONFLICT (player_id) DO UPDATE SET
                apuesta_promedio = :apuesta_promedio,
                apuesta_max = :apuesta_max,
                ganancia_max = :ganancia_max,
                ratio_promedio = :ratio_promedio,
                total_sesiones = jugadores.total_sesiones + 1,
                ultima_actualizacion = NOW();
        """), {
            'player_id': player_id,
            'currency': currency,
            'apuesta_promedio': apuesta_promedio,
            'apuesta_max': apuesta_max,
            'ganancia_max': ganancia_max,
            'ratio_promedio': ratio_promedio
        })
        conn.commit()
    print(f"Jugador {player_id} guardado/actualizado")

def guardar_validacion(df, resultado, reporte_whatsapp):
    if not test_conexion():
        print("⚠️  Sin conexión a Supabase — validación no guardada")
        return None
    engine = get_engine()
    player_id = str(df['PlayerId'].iloc[0])
    fecha_inicio = df['EventTime'].min()
    fecha_fin = df['EventTime'].max()
    total_registros = len(df)
    with engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO validaciones (player_id, fecha_inicio, fecha_fin, total_registros, resultado, reporte_whatsapp)
            VALUES (:player_id, :fecha_inicio, :fecha_fin, :total_registros, :resultado, :reporte_whatsapp)
            RETURNING validacion_id;
        """), {
            'player_id': player_id,
            'fecha_inicio': fecha_inicio,
            'fecha_fin': fecha_fin,
            'total_registros': total_registros,
            'resultado': resultado,
            'reporte_whatsapp': reporte_whatsapp
        })
        validacion_id = result.fetchone()[0]
        conn.commit()
    print(f"Validación guardada con ID: {validacion_id}")
    return validacion_id

def guardar_anomalias(df_anomalias, validacion_id):
    if validacion_id is None:
        print("⚠️  Sin conexión a Supabase — anomalías no guardadas")
        return
    if df_anomalias.empty:
        print("No hay anomalías para guardar")
        return
    engine = get_engine()
    with engine.connect() as conn:
        for _, row in df_anomalias.iterrows():
            conn.execute(text("""
                INSERT INTO anomalias (validacion_id, game_instance_id, event_time, game_id, total_bet, total_win, total_jp_win, tipo_anomalia, descripcion, score_anomalia)
                VALUES (:validacion_id, :game_instance_id, :event_time, :game_id, :total_bet, :total_win, :total_jp_win, :tipo_anomalia, :descripcion, :score_anomalia);
            """), {
                'validacion_id': validacion_id,
                'game_instance_id': str(row.get('GameInstanceId', '')),
                'event_time': row['EventTime'],
                'game_id': str(row.get('GameId', '')),
                'total_bet': float(row['TotalBet']),
                'total_win': float(row['TotalWin']),
                'total_jp_win': float(row['TotalJPWin']) if pd.notna(row.get('TotalJPWin')) else None,
                'tipo_anomalia': str(row['tipo_anomalia']),
                'descripcion': str(row['razon_anomalia']) if pd.notna(row.get('razon_anomalia')) else f"Ratio: {round(row['ratio_ganancia'], 2)}x — Clasificación: {row['clasificacion']}",
                'score_anomalia': float(row['anomalia_score'])
            })
        conn.commit()
    print(f"Anomalías guardadas: {len(df_anomalias)} registros")