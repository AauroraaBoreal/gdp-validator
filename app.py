# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import os
import sys
import importlib

# --- HOT RELOAD FOR MODULES ---
for modulo in ['modules.modulo1_carga', 'modules.modulo3_anomalias', 'modules.modulo4_base_datos', 'modules.modulo5_reportes', 'modules.modulo6_evaluacion_no_supervisada']:
    if modulo in sys.modules:
        importlib.reload(sys.modules[modulo])

# pyrefly: ignore [missing-import]
from supabase import create_client
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def obtener_formato_moneda(df_or_row):
    """
    Retorna el símbolo y el nombre de la moneda a partir de la columna/campo 'Currency'.
    Por defecto retorna ('$', 'MXN').
    """
    try:
        from modules.modulo5_reportes import obtener_formato_moneda as _obtener
        return _obtener(df_or_row)
    except Exception:
        # Fallback local para evitar problemas de caché de módulos en Streamlit Cloud
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

# --- CONFIG ---
st.set_page_config(
    page_title="GDP Validator",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS ---
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .stApp { background-color: #0f0f0f; color: #ffffff; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #16213e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .anomalia-card {
        background: #1a0000;
        border-left: 4px solid #ff4444;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .normal-card {
        background: #001a00;
        border-left: 4px solid #44ff44;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .reporte-box {
        background: #1a1a2e;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 20px;
        font-family: monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# --- AUTENTICACIÓN ---
def login():
    st.markdown("<h1 style='text-align:center; color:#fff;'>🎰 GDP Validator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#aaa;'>Sistema inteligente de validación de jugadas</p>", unsafe_allow_html=True)
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("### Iniciar sesión")
        email = st.text_input("Correo electrónico", placeholder="analista@gdp.com")
        password = st.text_input("Contraseña", type="password")

        if st.button("Ingresar", use_container_width=True, type="primary"):
            try:
                response = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                st.session_state['user'] = response.user
                st.session_state['session'] = response.session
                st.rerun()
            except Exception as e:
                st.error("Correo o contraseña incorrectos")

def logout():
    supabase.auth.sign_out()
    st.session_state.clear()
    st.rerun()

# --- PROCESAR CSV ---
def procesar_archivo(uploaded_file, monto_validar=None):
    import tempfile
    from modules.modulo1_carga import cargar_csv
    from modules.modulo3_random_forest import detectar_anomalias_random_forest
    from modules.modulo4_base_datos import crear_tablas, guardar_jugador, guardar_validacion, guardar_anomalias
    from modules.modulo5_reportes import generar_reporte_whatsapp, generar_reporte_qa

    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with st.spinner("Cargando y preprocesando datos..."):
            df = cargar_csv(tmp_path)

        with st.spinner("Detectando anomalías..."):
            df, _ = detectar_anomalias_random_forest(df)

        with st.spinner("Guardando en base de datos..."):
            crear_tablas()
            guardar_jugador(df)

        anomalias = df[df['es_anomalia'] == True]
        resultado = 'sospechoso' if len(anomalias) > 0 else 'normal'
        reporte_whatsapp = generar_reporte_whatsapp(df, monto_validar)
        reporte_qa = generar_reporte_qa(anomalias) if len(anomalias) > 0 else None

        validacion_id = guardar_validacion(df, resultado, reporte_whatsapp)
        guardar_anomalias(anomalias, validacion_id)

        os.unlink(tmp_path)
        return df, anomalias, resultado, reporte_whatsapp, reporte_qa

    except Exception as e:
        os.unlink(tmp_path)
        raise e
        
# --- PÁGINA: VALIDAR ---
def pagina_validar():
    st.markdown("## 📂 Nueva validación")
    st.markdown("Sube el archivo CSV del jugador para procesarlo.")

    uploaded_file = st.file_uploader(
        "Selecciona el archivo CSV",
        type=['csv'],
        help="Archivo de jugadas exportado del sistema de GDP Studios"
    )

    if uploaded_file:
        st.info(f"Archivo cargado: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        # Intentar detectar la moneda del archivo cargado para mostrarla en el input
        moneda_detectada = "MXN"
        try:
            import io, csv
            content = uploaded_file.getvalue()
            # Leer el header y la primera fila del CSV
            f_in = io.StringIO(content.decode('utf-8', errors='replace'))
            reader = csv.reader(f_in)
            header = next(reader)
            if 'Currency' in header:
                idx_curr = header.index('Currency')
                first_row = next(reader)
                if len(first_row) > idx_curr:
                    val = str(first_row[idx_curr]).strip().upper()
                    if val in ['PEN', 'MXN']:
                        moneda_detectada = val
        except Exception:
            pass

        monto_validar = st.number_input(
            f"Monto del retiro a validar ({moneda_detectada})",
            min_value=0.0,
            value=0.0,
            step=0.01,
            help="Ingresa el monto del retiro a validar."
        )

        if st.button("🔍 Buscar retiro", type="primary", use_container_width=True):
            import tempfile, os
            import traceback
            from modules.modulo1_carga import cargar_csv
            from modules.modulo3_random_forest import detectar_anomalias_random_forest
            from modules.modulo5_reportes import buscar_retiro

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                with st.spinner("Cargando archivo..."):
                    df = cargar_csv(tmp_path)
                with st.spinner("Detectando anomalías..."):
                    df, _ = detectar_anomalias_random_forest(df)

                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    tmp_path = None

                # Guardar df en session state
                st.session_state['df_procesado'] = df

                if float(monto_validar) > 0:
                    # Buscar todos los retiros cercanos al monto
                    # pyrefly: ignore [missing-import]
                    import numpy as np
                    balance = df['Balance'].values
                    balance_change = df['BalanceChange'].values
                    diferencias = -(balance[:-1] - balance[1:] + balance_change[1:])

                    # Encontrar retiros con diferencia menor a $1000 del monto ingresado
                    tolerancia = 1000

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

                    # Para evitar considerar recargas (diferencias > 0) como retiros,
                    # calculamos la distancia absoluta de los retiros reales (diferencias < -0.01).
                    diff_abs = np.full(len(diferencias), np.inf)
                    if np.any(mask_retiros):
                        diff_abs[mask_retiros] = np.abs(diferencias[mask_retiros] + float(monto_validar))

                    indices_cercanos = np.where(diff_abs <= tolerancia)[0]

                    retiros_encontrados = []
                    for idx in indices_cercanos:
                        idx_retiro = idx + 1
                        monto_real = abs(diferencias[idx])
                        fecha = df.loc[idx_retiro, 'EventTime']
                        # La fila del retiro: si hay una fila saltada en el CSV, es esa. Si no, es la fila donde se refleja el salto (idx_retiro)
                        if '_fila_csv' in df.columns:
                            fila_antes = int(df.loc[idx, '_fila_csv'])
                            fila_despues = int(df.loc[idx_retiro, '_fila_csv'])
                            if fila_despues - fila_antes > 1:
                                fila_csv = fila_antes + 1 # Fila omitida en la lectura (ej. sin EventTime)
                            else:
                                fila_csv = fila_despues # Reflejado entre estas dos, apuntamos a la posterior
                        else:
                            fila_csv = idx_retiro + 1
                        retiros_encontrados.append({
                            'idx': idx_retiro,
                            'monto': monto_real,
                            'fecha': fecha,
                            'fila': fila_csv
                        })

                    st.session_state['retiros_encontrados'] = retiros_encontrados
                    st.session_state['monto_validar'] = float(monto_validar)

                else:
                    st.session_state['retiros_encontrados'] = None
                    st.session_state['monto_validar'] = 0

            except Exception as e:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                for key in ['df_procesado', 'retiros_encontrados', 'retiro_confirmado',
                            'monto_validar', 'seleccion_retiro']:
                    if key in st.session_state:
                        del st.session_state[key]
                print("--- ERROR LOGS ---")
                traceback.print_exc()
                st.warning(f"⚠️ Error al procesar el archivo: {str(e)}")

        # Mostrar opciones de retiro si hay múltiples
        if 'retiros_encontrados' in st.session_state and st.session_state['retiros_encontrados']:
            retiros = st.session_state['retiros_encontrados']
            df = st.session_state['df_procesado']
            simbolo, moneda = obtener_formato_moneda(df)

            if len(retiros) > 1:
                st.warning(f"⚠️ Se encontraron **{len(retiros)} retiros** con montos similares. Selecciona el que deseas validar:")

                opciones = {
                    f"Fila {r['fila']} — {simbolo}{r['monto']:,.2f} {moneda} — {pd.to_datetime(r['fecha']).strftime('%d/%m/%Y %H:%M')}": r
                    for r in retiros
                }

                seleccion = st.radio(
                    "Retiros encontrados:",
                    list(opciones.keys()),
                    key="seleccion_retiro"
                )

                retiro_seleccionado = opciones[seleccion]

            else:
                r = retiros[0]
                st.info(f"✅ Se encontró **1 retiro**: Fila {r['fila']} — {simbolo}{r['monto']:,.2f} {moneda} — {pd.to_datetime(r['fecha']).strftime('%d/%m/%Y %H:%M')}")
                retiro_seleccionado = r

            if st.button("✅ Validar retiro seleccionado", type="primary", use_container_width=True):
                st.session_state['retiro_confirmado'] = retiro_seleccionado
        elif 'df_procesado' in st.session_state and st.session_state.get('monto_validar', 0) > 0:
            df = st.session_state['df_procesado']
            simbolo, moneda = obtener_formato_moneda(df)
            st.error(f"❌ No se encontró ningún retiro cercano a **{simbolo}{st.session_state['monto_validar']:,.2f} {moneda}** en el archivo cargado.")

        # Procesar una vez confirmado el retiro
        if 'retiro_confirmado' in st.session_state and 'df_procesado' in st.session_state:
            df = st.session_state['df_procesado']
            retiro = st.session_state['retiro_confirmado']

            try:
                from modules.modulo5_reportes import generar_reporte_whatsapp, generar_reporte_qa
                from modules.modulo4_base_datos import crear_tablas, guardar_jugador, guardar_validacion, guardar_anomalias

                idx_evento = retiro['idx']
                monto_encontrado = retiro['monto']

                # Generar reporte con índice específico
                reporte_whatsapp = generar_reporte_whatsapp(
                    df,
                    monto_validar=st.session_state['monto_validar'],
                    idx_forzado=idx_evento
                )

                anomalias = df[df['es_anomalia'] == True]
                resultado = 'sospechoso' if len(anomalias) > 0 else 'normal'
                reporte_qa = generar_reporte_qa(anomalias) if len(anomalias) > 0 else None

                # Guardar en BD
                try:
                    crear_tablas()
                    guardar_jugador(df)
                    validacion_id = guardar_validacion(df, resultado, reporte_whatsapp)
                    guardar_anomalias(anomalias, validacion_id)
                except Exception as db_err:
                    import traceback
                    print("--- DATABASE ERROR LOGS ---")
                    traceback.print_exc()
                    st.warning(f"⚠️ Error de conexión a la base de datos: {str(db_err)}")

                # Mostrar métricas con estilo premium
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #888899; margin-bottom: 5px;">Total registros</div>
                        <div style="font-size: 1.8rem; font-weight: bold; color: #ffffff;">{len(df):,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #888899; margin-bottom: 5px;">Jugador</div>
                        <div style="font-size: 1.8rem; font-weight: bold; color: #ffffff; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{df['PlayerId'].iloc[0]}">{df['PlayerId'].iloc[0]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    color_anom = "#ff4444" if len(anomalias) > 0 else "#44ff44"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #888899; margin-bottom: 5px;">Anomalías detectadas</div>
                        <div style="font-size: 1.8rem; font-weight: bold; color: {color_anom};">{len(anomalias)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    color_res = "#ff4444" if resultado == 'sospechoso' else "#44ff44"
                    label_res = "⚠️ Sospechoso" if resultado == 'sospechoso' else "✅ Normal"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #888899; margin-bottom: 5px;">Resultado</div>
                        <div style="font-size: 1.8rem; font-weight: bold; color: {color_res};">{label_res}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()

                # --- TOP 5 JUEGOS ---
                st.markdown("### 📊 Top 5 Juegos más Jugados")
                top_juegos = df['GameId'].value_counts().head(5)
                df_top_juegos = pd.DataFrame({
                    'Juego': top_juegos.index,
                    'Jugadas': top_juegos.values,
                    'Porcentaje': (top_juegos.values / len(df) * 100).round(2)
                })

                col_chart, col_legend = st.columns([2, 1])
                with col_chart:
                    # pyrefly: ignore [missing-import]
                    import altair as alt
                    chart = alt.Chart(df_top_juegos).mark_bar(
                        cornerRadiusTopLeft=5,
                        cornerRadiusTopRight=5
                    ).encode(
                        x=alt.X('Juego:N', sort='-y', title='Juego'),
                        y=alt.Y('Jugadas:Q', title='Cantidad de Jugadas'),
                        color=alt.Color('Juego:N', scale=alt.Scale(scheme='darkmulti'), legend=None),
                        tooltip=['Juego', 'Jugadas', alt.Tooltip('Porcentaje', format='.2f', title='Porcentaje (%)')]
                    ).properties(
                        height=250
                    ).configure_axis(
                        labelColor='#cccccc',
                        titleColor='#ffffff',
                        grid=False
                    ).configure_view(
                        strokeWidth=0
                    )
                    st.altair_chart(chart, use_container_width=True)

                with col_legend:
                    st.markdown("<div style='padding-top: 10px;'><b>Resumen de Actividad:</b></div>", unsafe_allow_html=True)
                    for i, row_j in df_top_juegos.iterrows():
                        st.write(f"🎮 **{row_j['Juego']}**: {row_j['Jugadas']:,} jugadas ({row_j['Porcentaje']:.1f}%)")

                st.divider()

                if resultado == 'normal':
                    st.markdown("<div class='normal-card'>✅ <b>Validación normal</b> — No se detectaron anomalías.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='anomalia-card'>⚠️ <b>Se detectaron {len(anomalias)} registros sospechosos.</b></div>", unsafe_allow_html=True)

                st.markdown("### 📋 Reporte WhatsApp")
                st.code(reporte_whatsapp, language=None)

                if reporte_qa:
                    st.markdown("### 🔴 Registros sospechosos")
                    simbolo_qa, moneda_qa = obtener_formato_moneda(df)
                    df_qa = anomalias.copy()
                    df_qa['Fila'] = df_qa.index + 1
                    df_qa['Fecha y Hora'] = df_qa['EventTime'].astype(str)
                    df_qa['Juego'] = df_qa['GameId']
                    df_qa['Apuesta'] = df_qa['TotalBet'].apply(lambda x: f"{simbolo_qa}{x:,.2f} {moneda_qa}")
                    df_qa['Ganancia'] = df_qa['TotalWin'].apply(lambda x: f"{simbolo_qa}{x:,.2f} {moneda_qa}")
                    df_qa['Jackpot'] = df_qa['TotalJPWin'].apply(
                        lambda x: f"{simbolo_qa}{x:,.2f} {moneda_qa}" if pd.notna(x) and x > 0 else "—"
                    )
                    df_qa['Ratio'] = df_qa['ratio_ganancia'].apply(lambda x: f"x{x:,.2f}")
                    df_qa['Tipo'] = df_qa['tipo_anomalia']
                    df_qa['Razón'] = df_qa['razon_anomalia'].fillna("—")
                    df_qa['GameInstanceID'] = df_qa['GameInstanceId']

                    # Filtro interactivo por tipo de anomalía
                    tipos_disponibles = sorted(df_qa['Tipo'].dropna().unique().tolist())
                    if tipos_disponibles:
                        filtro_tipos = st.multiselect(
                            "Filtrar por tipo de anomalía:",
                            options=tipos_disponibles,
                            default=tipos_disponibles,
                            help="Selecciona los tipos de anomalías que deseas ver en la tabla."
                        )
                        df_qa_filtrado = df_qa[df_qa['Tipo'].isin(filtro_tipos)]
                    else:
                        df_qa_filtrado = df_qa

                    st.dataframe(
                        df_qa_filtrado[['Fila', 'GameInstanceID', 'Fecha y Hora', 'Juego',
                                'Apuesta', 'Ganancia', 'Jackpot', 'Ratio', 'Tipo', 'Razón']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'Razón': st.column_config.TextColumn(width='large'),
                            'GameInstanceID': st.column_config.TextColumn(width='medium'),
                        }
                    )

                    with st.expander("📋 Ver reporte de texto"):
                        st.code(reporte_qa, language=None)
            except Exception as e:
                import traceback
                print("--- VALIDATION ERROR LOGS ---")
                traceback.print_exc()
                st.warning(f"⚠️ Error al realizar la validación: {str(e)}")

            # Limpiar estado para nueva validación
            if st.button("🔄 Nueva validación"):
                for key in ['df_procesado', 'retiros_encontrados', 'retiro_confirmado',
                            'monto_validar', 'seleccion_retiro']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

# --- PÁGINA: HISTORIAL ---
def pagina_historial():
    st.markdown("## 📜 Historial de validaciones")

    try:
        response = supabase.table('validaciones').select(
            'validacion_id, player_id, fecha_inicio, fecha_fin, total_registros, resultado, fecha_procesamiento'
        ).order('fecha_procesamiento', desc=True).limit(50).execute()

        if not response.data:
            st.info("No hay validaciones registradas aún.")
            return

        df_hist = pd.DataFrame(response.data)
        df_hist['fecha_procesamiento'] = pd.to_datetime(df_hist['fecha_procesamiento'])
        if df_hist['fecha_procesamiento'].dt.tz is None:
            df_hist['fecha_procesamiento'] = df_hist['fecha_procesamiento'].dt.tz_localize('UTC')
        df_hist['fecha_procesamiento'] = df_hist['fecha_procesamiento'].dt.tz_convert('America/Lima').dt.strftime('%d/%m/%Y %H:%M')
        df_hist['fecha_inicio'] = pd.to_datetime(df_hist['fecha_inicio']).dt.strftime('%d/%m/%Y')
        df_hist['fecha_fin'] = pd.to_datetime(df_hist['fecha_fin']).dt.strftime('%d/%m/%Y')
        df_hist['resultado'] = df_hist['resultado'].apply(
            lambda x: '⚠️ Sospechoso' if x == 'sospechoso' else '✅ Normal'
        )
        df_hist.columns = ['ID', 'Jugador', 'Desde', 'Hasta', 'Registros', 'Resultado', 'Procesado']

        # Filtro por jugador
        jugadores = ['Todos'] + sorted(df_hist['Jugador'].unique().tolist())
        filtro = st.selectbox("Filtrar por jugador", jugadores)
        if filtro != 'Todos':
            df_hist = df_hist[df_hist['Jugador'] == filtro]

        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        # Ver detalle de anomalías
        st.markdown("### 🔍 Ver detalles de una validación")
        validacion_id = st.number_input("ID de validación", min_value=1, step=1)
        if st.button("Ver detalles"):
            val_resp = supabase.table('validaciones').select('top_juegos_json, player_id').eq('validacion_id', validacion_id).execute()
            anom_response = supabase.table('anomalias').select('*').eq('validacion_id', validacion_id).execute()

            if not val_resp.data:
                st.error("No se encontró ninguna validación con el ID ingresado.")
            else:
                row_val = val_resp.data[0]
                if row_val.get('top_juegos_json'):
                    import json
                    try:
                        data_juegos = json.loads(row_val['top_juegos_json'])
                        if data_juegos:
                            df_top_juegos = pd.DataFrame(data_juegos)
                            st.markdown("#### 📊 Top 5 Juegos en esta Validación")
                            col_chart, col_legend = st.columns([2, 1])
                            with col_chart:
                                # pyrefly: ignore [missing-import]
                                import altair as alt
                                chart = alt.Chart(df_top_juegos).mark_bar(
                                    cornerRadiusTopLeft=5,
                                    cornerRadiusTopRight=5
                                ).encode(
                                    x=alt.X('Juego:N', sort='-y', title='Juego'),
                                    y=alt.Y('Jugadas:Q', title='Cantidad de Jugadas'),
                                    color=alt.Color('Juego:N', scale=alt.Scale(scheme='darkmulti'), legend=None),
                                    tooltip=['Juego', 'Jugadas', alt.Tooltip('Porcentaje', format='.2f', title='Porcentaje (%)')]
                                ).properties(
                                    height=250
                                ).configure_axis(
                                    labelColor='#cccccc',
                                    titleColor='#ffffff',
                                    grid=False
                                ).configure_view(
                                    strokeWidth=0
                                )
                                st.altair_chart(chart, use_container_width=True)
                            with col_legend:
                                st.markdown("<div style='padding-top: 10px;'><b>Resumen de Actividad:</b></div>", unsafe_allow_html=True)
                                for i, row_j in df_top_juegos.iterrows():
                                    st.write(f"🎮 **{row_j['Juego']}**: {row_j['Jugadas']:,} jugadas ({row_j['Porcentaje']:.1f}%)")
                    except Exception as e:
                        st.warning(f"No se pudo cargar el gráfico de esta sesión: {e}")

                st.markdown("#### 🔴 Registros sospechosos")
                if not anom_response.data:
                    st.info("Esta validación no tiene anomalías registradas.")
                else:
                    df_anom = pd.DataFrame(anom_response.data)
                    player_resp = supabase.table('jugadores').select('currency').eq('player_id', row_val.get('player_id')).execute()
                    simbolo_an = '$'
                    moneda_an = 'MXN'
                    if player_resp.data:
                        curr_val = str(player_resp.data[0].get('currency', 'MXN')).strip().upper()
                        if curr_val == 'PEN':
                            simbolo_an = 'S/'
                            moneda_an = 'PEN'

                    df_anom['Fecha y Hora'] = pd.to_datetime(df_anom['event_time']).dt.strftime('%d/%m/%Y %H:%M')
                    df_anom['Apuesta'] = df_anom['total_bet'].apply(lambda x: f"{simbolo_an}{float(x):,.2f} {moneda_an}")
                    df_anom['Ganancia'] = df_anom['total_win'].apply(lambda x: f"{simbolo_an}{float(x):,.2f} {moneda_an}")
                    df_anom['Jackpot'] = df_anom['total_jp_win'].apply(
                        lambda x: f"{simbolo_an}{float(x):,.2f} {moneda_an}" if pd.notna(x) and float(x) > 0 else "—"
                    )

                    st.dataframe(df_anom[[
                        'Fecha y Hora', 'game_id', 'Apuesta',
                        'Ganancia', 'Jackpot', 'tipo_anomalia', 'descripcion'
                    ]].rename(columns={
                        'game_id': 'Juego',
                        'tipo_anomalia': 'Tipo',
                        'descripcion': 'Descripción'
                    }), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error al cargar historial: {str(e)}")

# --- PÁGINA: JUGADORES ---
def pagina_jugadores():
    st.markdown("## 👤 Perfil de jugadores")

    try:
        response = supabase.table('jugadores').select('*').order(
            'ultima_actualizacion', desc=True
        ).execute()

        if not response.data:
            st.info("No hay jugadores registrados aún.")
            return

        df_jug = pd.DataFrame(response.data)
        df_jug['ultima_actualizacion'] = pd.to_datetime(df_jug['ultima_actualizacion'])
        if df_jug['ultima_actualizacion'].dt.tz is None:
            df_jug['ultima_actualizacion'] = df_jug['ultima_actualizacion'].dt.tz_localize('UTC')
        df_jug['ultima_actualizacion'] = df_jug['ultima_actualizacion'].dt.tz_convert('America/Lima').dt.strftime('%d/%m/%Y %H:%M')
        def fmt_row_moneda(row, col_name):
            curr = str(row.get('currency', 'MXN')).strip().upper()
            val = row[col_name]
            simbolo = 'S/' if curr == 'PEN' else '$'
            return f"{simbolo}{val:,.2f}"

        df_jug['apuesta_promedio'] = df_jug.apply(lambda r: fmt_row_moneda(r, 'apuesta_promedio'), axis=1)
        df_jug['apuesta_max'] = df_jug.apply(lambda r: fmt_row_moneda(r, 'apuesta_max'), axis=1)
        df_jug['ganancia_max'] = df_jug.apply(lambda r: fmt_row_moneda(r, 'ganancia_max'), axis=1)

        st.dataframe(df_jug[[
            'player_id', 'currency', 'apuesta_promedio', 'apuesta_max',
            'ganancia_max', 'ratio_promedio', 'total_sesiones', 'ultima_actualizacion'
        ]].rename(columns={
            'player_id': 'Jugador',
            'currency': 'Moneda',
            'apuesta_promedio': 'Apuesta Prom.',
            'apuesta_max': 'Apuesta Máx.',
            'ganancia_max': 'Ganancia Máx.',
            'ratio_promedio': 'Ratio Prom.',
            'total_sesiones': 'Sesiones',
            'ultima_actualizacion': 'Última validación'
        }), use_container_width=True, hide_index=True)

        # Agregar el top global de juegos al final del apartado de jugadores
        st.divider()
        st.markdown("### 📊 Top 5 Juegos más Jugados en General (Global)")
        
        val_resp = supabase.table('validaciones').select('top_juegos_json').execute()
        if val_resp.data:
            import json
            global_counts = {}
            for item in val_resp.data:
                json_str = item.get('top_juegos_json')
                if json_str:
                    try:
                        records = json.loads(json_str)
                        for r in records:
                            game = r['Juego']
                            plays = int(r['Jugadas'])
                            global_counts[game] = global_counts.get(game, 0) + plays
                    except Exception:
                        pass
            
            if global_counts:
                sorted_games = sorted(global_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                total_plays = sum([g[1] for g in sorted_games])
                
                df_global = pd.DataFrame({
                    'Juego': [g[0] for g in sorted_games],
                    'Jugadas': [g[1] for g in sorted_games],
                    'Porcentaje': [(g[1] / total_plays * 100) if total_plays > 0 else 0.0 for g in sorted_games]
                })
                
                col_chart, col_legend = st.columns([2, 1])
                with col_chart:
                    # pyrefly: ignore [missing-import]
                    import altair as alt
                    chart = alt.Chart(df_global).mark_bar(
                        cornerRadiusTopLeft=5,
                        cornerRadiusTopRight=5
                    ).encode(
                        x=alt.X('Juego:N', sort='-y', title='Juego'),
                        y=alt.Y('Jugadas:Q', title='Cantidad de Jugadas'),
                        color=alt.Color('Juego:N', scale=alt.Scale(scheme='darkmulti'), legend=None),
                        tooltip=['Juego', 'Jugadas', alt.Tooltip('Porcentaje', format='.2f', title='Porcentaje (%)')]
                    ).properties(
                        height=250
                    ).configure_axis(
                        labelColor='#cccccc',
                        titleColor='#ffffff',
                        grid=False
                    ).configure_view(
                        strokeWidth=0
                    )
                    st.altair_chart(chart, use_container_width=True)
                    
                with col_legend:
                    st.markdown("<div style='padding-top: 10px;'><b>Resumen de Actividad Global:</b></div>", unsafe_allow_html=True)
                    for i, row_j in df_global.iterrows():
                        st.write(f"🎮 **{row_j['Juego']}**: {row_j['Jugadas']:,} jugadas ({row_j['Porcentaje']:.1f}%)")
            else:
                st.info("Aún no hay datos suficientes de jugadas registradas para mostrar el gráfico global.")
        else:
            st.info("Aún no hay datos de validaciones registrados para mostrar el gráfico global.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        st.error(f"Error al cargar jugadores: {str(e)}")

# --- NAVEGACIÓN PRINCIPAL ---
def main():
    if 'user' not in st.session_state:
        login()
        return

    user = st.session_state['user']

    with st.sidebar:
        st.markdown("### 🎰 GDP Validator")
        st.markdown(f"👤 **{user.email}**")
        st.divider()

        pagina = st.radio(
            "Navegación",
            ["📂 Nueva validación", "📜 Historial", "👤 Jugadores"],
            label_visibility="collapsed"
        )



        st.divider()
        if st.button("Cerrar sesión", use_container_width=True):
            logout()

    if pagina == "📂 Nueva validación":
        pagina_validar()
    elif pagina == "📜 Historial":
        pagina_historial()
    elif pagina == "👤 Jugadores":
        pagina_jugadores()

if __name__ == "__main__":
    main()