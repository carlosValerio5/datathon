"""
Aplicación Streamlit ENRIQUECIDA para visualización de resultados de XGBoost
con procesamiento por lotes y insights de ingeniería de variables.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
import time
import gc  # Garbage collector para liberar memoria

# Configuración para controlar el uso de memoria
  # Máximo número de filas a mostrar en tablas

# Configuración de la página
st.set_page_config(
    page_title="Optimización de Cobranza - Insights Enriquecidos",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("Optimización de Cobranza Domiciliada - Insights Enriquecidos")
st.markdown("""
Esta aplicación utiliza modelos XGBoost y **features enriquecidas** para optimizar la cobranza,
identificando variables clave y simulando escenarios. Los datos se procesan por lotes.
""")

# --- Funciones de Carga y Conexión (sin cambios) ---
@st.cache_resource
def load_models():
    """Carga los modelos y resultados guardados."""
    try:
        with open("resultados/pipeline_min.pkl", "rb") as f:
            pipeline_min = pickle.load(f)
        with open("resultados/pipeline_max.pkl", "rb") as f:
            pipeline_max = pickle.load(f)
        importance_min = pd.read_pickle("resultados/importance_min.pkl")
        importance_max = pd.read_pickle("resultados/importance_max.pkl")
        with open("resultados/ecuaciones.txt", "r") as f:
            ecuaciones = f.read()
        return pipeline_min, pipeline_max, importance_min, importance_max, ecuaciones
    except Exception as e:
        st.error(f"Error al cargar modelos: {e}")
        return None, None, None, None, None

@st.cache_resource(ttl=300)
def connect_to_db():
    """Establece conexión con la base de datos DuckDB."""
    try:
        # Asegúrate de que la ruta a tu base de datos sea correcta
        return duckdb.connect("../credifiel/db/credifiel.duckdb")
    except Exception as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return None


# Función para verificar y reconectar si es necesario
def get_connection():
    """Obtiene una conexión activa a la base de datos, reconectando si es necesario."""
    conn = connect_to_db()

    # Verificar si la conexión está activa
    try:
        # Ejecutar una consulta simple para verificar la conexión
        conn.execute("SELECT 1").fetchone()
        return conn
    except:
        # Si hay error, intentar reconectar
        st.warning("Reconectando a la base de datos...")
        # Forzar la recreación del recurso en caché
        connect_to_db.clear()
        return connect_to_db()

# --- Funciones de Carga de Datos y Estadísticas (adaptadas para features enriquecidas) ---
def load_data(conn, query):
    if not conn:
        return pd.DataFrame()
    try:
        # limited_query = f"{query} LIMIT {max_rows}"
        with st.spinner(f"Cargando datos completos..."):
            start_time = time.time()
            df = conn.execute(query).fetchdf()
            elapsed_time = time.time() - start_time
        st.success(f"Datos cargados en {elapsed_time:.2f} segundos. Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return pd.DataFrame()

def get_stats(conn, table_name):
    """Obtiene estadísticas básicas de una tabla."""
    if not conn:
        return {}
    try:
        count_query = f"SELECT COUNT(*) as total FROM {table_name}"
        total = conn.execute(count_query).fetchone()[0]
        stats_query = f"""
        SELECT 
            AVG(tasa_exito) as avg_tasa_exito,
            MIN(tasa_exito) as min_tasa_exito,
            MAX(tasa_exito) as max_tasa_exito
        FROM {table_name}
        """
        stats = conn.execute(stats_query).fetchone()
        return {
            "total_registros": total,
            "tasa_exito_promedio": stats[0] if stats else 0,
            "tasa_exito_min": stats[1] if stats else 0,
            "tasa_exito_max": stats[2] if stats else 1,
        }
    except Exception as e:
        st.error(f"Error al obtener estadísticas: {e}")
        return {}

# --- Función para crear ecuación (sin cambios) ---
def crear_ecuacion_optimizacion(importance_df, top_n=10):
    """Crea una ecuación de optimización."""
    top_features = importance_df.head(top_n)
    equation = "Score = "
    for i, row in top_features.iterrows():
        feature = row["Feature"]
        importance = row["Importance"]
        if i > 0:
            equation += " + "
        equation += f"{importance:.4f} * {feature}"
    return equation

# --- Carga Inicial ---
pipeline_min, pipeline_max, importance_min, importance_max, ecuaciones = load_models()
conn = get_connection()

# --- Sidebar ---
st.sidebar.header("Configuración")
model_type = st.sidebar.selectbox(
    "Seleccionar Modelo", ["Minimización de Costos", "Maximización de Cobranza"]
)

# Determinar tabla y orden según modelo
if model_type == "Minimización de Costos":
    table_name = "analytics.mart_optimizacion_costos" # Asume que este mart tiene los features
    order_by = "indice_eficiencia DESC"
    importance_df = importance_min
    pipeline = pipeline_min
    color_scale = "Blues"
else:
    table_name = "analytics.mart_maximizacion_cobranza" # Asume que este mart tiene los features
    order_by = "indice_potencial_recuperacion DESC"
    importance_df = importance_max
    pipeline = pipeline_max
    color_scale = "Reds"

equation_min_fixed = (
    "Score_minimizacion = "
    "0.6074 * tasa_recuperacion_monto + "
    "0.3605 * tasa_recuperacion + "
    "0.0251 * nombre_emisora_SANTANDER_CUENTA + "
    "0.0027 * idBanco + "
    "0.0019 * idCredito + "
    "0.0016 * costo_por_peso_recuperado + "
    "0.0006 * idEmisora + "
    "0.0002 * nombre_emisora_BANAMEX_TARJETA + "
    "0.0000 * monto_recuperado + "
    "0.0000 * total_exitosos"
)

# Obtener estadísticas
stats = get_stats(get_connection(), table_name)
st.sidebar.subheader("Estadísticas Generales")
if stats:
    st.sidebar.metric("Total de registros", f"{stats.get("total_registros", 0):,}")
    st.sidebar.metric("Tasa de éxito promedio", f"{stats.get("tasa_exito_promedio", 0):.2%}")
    # ... (otras métricas si se desean)

# --- Filtros (Adaptados para features enriquecidas si están en los marts) ---
st.sidebar.subheader("Filtros")
filter_options = st.sidebar.expander("Opciones de filtrado", expanded=False)
where_clauses = []

with filter_options:
    # Filtro por tasa de éxito
    min_tasa = float(stats.get("tasa_exito_min", 0))
    max_tasa = float(stats.get("tasa_exito_max", 1))
    tasa_exito_range = st.slider(
        "Rango de tasa de éxito",
        min_value=min_tasa,
        max_value=max_tasa,
        value=(min_tasa, max_tasa),
        format="%.2f",
    )
    where_clauses.append(f"tasa_exito BETWEEN {tasa_exito_range[0]} AND {tasa_exito_range[1]}")

    # Filtro por Rango de Monto (si existe la columna en el mart)
    try:
        rangos_monto = conn.execute(f"SELECT DISTINCT rango_monto FROM {table_name} WHERE rango_monto IS NOT NULL").fetchdf()["rango_monto"].tolist()
        selected_rangos = st.multiselect("Rango de Monto Exigible", rangos_monto, default=rangos_monto)
        if selected_rangos and len(selected_rangos) < len(rangos_monto):
            where_clauses.append(f"rango_monto IN ({', '.join([f'{r}' for r in selected_rangos])})")
    except Exception:
        st.info("Columna 'rango_monto' no encontrada para filtrar.")

    # Filtro por Año (si existe la columna en el mart)
    try:
        years = conn.execute(f"SELECT DISTINCT year FROM {table_name} WHERE year IS NOT NULL ORDER BY year").fetchdf()["year"].tolist()
        selected_years = st.multiselect("Año", years, default=years)
        if selected_years and len(selected_years) < len(years):
            where_clauses.append(f"year IN ({', '.join(map(str, selected_years))})")
    except Exception:
        st.info("Columna 'year' no encontrada para filtrar.")

# Construir cláusula WHERE final
where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
base_query = f"SELECT * FROM {table_name} WHERE {where_clause}"

# --- Pestañas Principales ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Análisis de Variables",
    "💡 Insights Enriquecidos",
    "🔍 Estrategias Óptimas",
    "🧪 Simulador",
])

# Pestaña 1: Análisis de Variables (Importancia y SHAP)
with tab1:
    if importance_df is not None:
        # Visualizar top 15 características

        st.subheader("Ecuación de optimización")
        if model_type == "Minimización de Costos":
            # mostramos SIEMPRE la ecuación fija
            st.code(equation_min_fixed)
        else:
            # Maximizacion: intenta usar el archivo ecuaciones.txt;
            # si no existe, cae en la función que la genera al vuelo
            if ecuaciones:
                # 🛈 tu archivo ecuaciones.txt sigue funcionando igual
                lines = ecuaciones.split("\n\n")
                st.code(lines[1].split("\n", 1)[1] if len(lines) > 1 else "Ecuación no encontrada")
            else:
                st.code(crear_ecuacion_optimizacion(importance_df))

        # Mostrar gráficos SHAP
        st.subheader("Análisis SHAP")
        col1, col2 = st.columns(2)
        shap_suffix = "minimizacion" if model_type == "Minimización de Costos" else "maximizacion"
        with col1:
            shap_bar_path = f"resultados/shap_bar_{shap_suffix}.png"
            if os.path.exists(shap_bar_path):
                st.image(shap_bar_path, caption="SHAP - Importancia")
            else: st.info("Gráfico SHAP (barras) no disponible.")
        with col2:
            shap_summary_path = f"resultados/shap_summary_{shap_suffix}.png"
            if os.path.exists(shap_summary_path):
                st.image(shap_summary_path, caption="SHAP - Impacto")
            else: st.info("Gráfico SHAP (resumen) no disponible.")
    else:
        st.warning("No se pudieron cargar datos de importancia.")
# Pestaña 2: Insights Enriquecidos (ENHANCED)
with tab2:
    st.header("💡 Insights Enriquecidos de Cobranza")

    # Get available columns from the main data table
    try:
        columnas = conn.execute("DESCRIBE analytics.stg_tabla_maestra_completa").fetchdf()["column_name"].tolist()
        st.expander("🔍 Columnas Disponibles", expanded=False).write(columnas)
    except Exception as e:
        st.error(f"Error al obtener columnas: {e}")
        columnas = []

    # Build filters specific to the insights table
    st.subheader("🎯 Filtros para Insights")
    col1, col2, col3 = st.columns(3)

    insight_filters = []

    with col1:
        # Date filter if available
        if any(col for col in columnas if 'fecha' in col.lower() or 'date' in col.lower()):
            date_columns = [col for col in columnas if 'fecha' in col.lower() or 'date' in col.lower()]
            st.write(f"Columnas de fecha disponibles: {date_columns}")

            # Try to get date range
            try:
                date_col = date_columns[0] if date_columns else None
                if date_col:
                    date_range_query = f"""
                    SELECT 
                        MIN({date_col}) as min_date,
                        MAX({date_col}) as max_date
                    FROM analytics.stg_tabla_maestra_completa
                    WHERE {date_col} IS NOT NULL
                    """
                    date_range = conn.execute(date_range_query).fetchone()
                    if date_range and date_range[0]:
                        start_date = st.date_input("Fecha inicio", value=pd.to_datetime(date_range[0]).date())
                        end_date = st.date_input("Fecha fin", value=pd.to_datetime(date_range[1]).date())
                        insight_filters.append(f"{date_col} BETWEEN '{start_date}' AND '{end_date}'")
            except Exception:
                st.info("No se pudo configurar filtro de fechas")

    with col2:
        # Status filter
        if any(col for col in columnas if 'status' in col.lower() or 'estado' in col.lower()):
            status_columns = [col for col in columnas if 'status' in col.lower() or 'estado' in col.lower()]
            try:
                status_col = status_columns[0]
                status_values = conn.execute(
                    f"SELECT DISTINCT {status_col} FROM analytics.stg_tabla_maestra_completa WHERE {status_col} IS NOT NULL").fetchdf()[
                    status_col].tolist()
                selected_status = st.multiselect(f"Filtrar por {status_col}", status_values, default=status_values)
                if selected_status and len(selected_status) < len(status_values):
                    status_list = "', '".join(selected_status)
                    insight_filters.append(f"{status_col} IN ('{status_list}')")
            except Exception:
                st.info("No se pudo configurar filtro de estado")

    with col3:
        # Amount range filter
        if any(col for col in columnas if 'monto' in col.lower() or 'amount' in col.lower()):
            amount_columns = [col for col in columnas if 'monto' in col.lower() or 'amount' in col.lower()]
            try:
                amount_col = amount_columns[0]
                amount_range = conn.execute(
                    f"SELECT MIN({amount_col}), MAX({amount_col}) FROM analytics.stg_tabla_maestra_completa WHERE {amount_col} IS NOT NULL").fetchone()
                if amount_range and amount_range[0] is not None:
                    min_amount = st.number_input("Monto mínimo", value=float(amount_range[0]), min_value=0.0)
                    max_amount = st.number_input("Monto máximo", value=float(amount_range[1]), min_value=0.0)
                    insight_filters.append(f"{amount_col} BETWEEN {min_amount} AND {max_amount}")
            except Exception:
                st.info("No se pudo configurar filtro de monto")

    # Build final WHERE clause for insights
    insight_where = " AND ".join(insight_filters) if insight_filters else "1=1"

    # Create tabs for different insights
    insight_tab4 = st.tabs([
        "🏦 Análisis por Entidad"
    ])

    # Tab 4: Entity Analysis
    with insight_tab4[0]:
        st.subheader("Análisis por Entidad")

        # Look for bank/entity columns
        entity_columns = [col for col in columnas if any(
            keyword in col.lower() for keyword in ['banco', 'bank', 'emisora', 'entity', 'institution'])]

        if entity_columns:
            entity_col = st.selectbox("Seleccionar entidad", entity_columns, key="entity_analysis")

            try:
                entity_query = f"""
                SELECT 
                    {entity_col},
                    COUNT(*) as total_casos,
                    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as porcentaje
                FROM analytics.stg_tabla_maestra_completa
                WHERE {insight_where} AND {entity_col} IS NOT NULL
                GROUP BY {entity_col}
                ORDER BY total_casos DESC
                LIMIT 20
                """
                df_entity = conn.execute(entity_query).fetchdf()

                if not df_entity.empty:
                    fig_entity = px.bar(df_entity, x=entity_col, y='total_casos',
                                        title=f"Casos por {entity_col}")
                    fig_entity.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_entity, use_container_width=True)

                    st.dataframe(df_entity)

            except Exception as e:
                st.error(f"Error en análisis por entidad: {e}")
        else:
            st.info("No se encontraron columnas de entidad")

    # Summary insights
    st.subheader("📈 Resumen de Insights")
    try:
        summary_query = f"""
        SELECT 
            COUNT(*) as total_registros,
            COUNT(DISTINCT CASE WHEN EXISTS(
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'stg_tabla_maestra_completa' 
                AND column_name LIKE '%banco%'
            ) THEN 1 END) as entidades_unicas
        FROM analytics.stg_tabla_maestra_completa
        WHERE {insight_where}
        """
        summary = conn.execute(summary_query).fetchone()

        col1, col2 = st.columns(2)
        col1.metric("Total de Registros Filtrados", f"{summary[0]:,}")

        # Additional summary based on available columns
        if any('monto' in col.lower() for col in columnas):
            monto_col = next(col for col in columnas if 'monto' in col.lower())
            total_amount_query = f"""
            SELECT SUM({monto_col}) as total_monto
            FROM analytics.stg_tabla_maestra_completa
            WHERE {insight_where} AND {monto_col} IS NOT NULL
            """
            total_amount = conn.execute(total_amount_query).fetchone()[0]
            col2.metric("Monto Total", f"${total_amount:,.2f}" if total_amount else "N/A")

    except Exception as e:
        st.warning(f"Error en resumen: {e}")

with tab3:
    st.header(f"Estrategias óptimas para {model_type.lower()}")
    query = f"{base_query} ORDER BY {order_by}"
    df = load_data(get_connection(), query)
    if not df.empty:
        if model_type == "Minimización de Costos":
            cols_to_show = ["idBanco", "nombre_banco", "idEmisora", "nombre_emisora",
                            "tipo_servicio", "tipo_cobro", "tasa_exito",
                            "costo_por_peso_recuperado", "indice_eficiencia"]
            st.dataframe(df[[col for col in cols_to_show if col in df.columns]])
            st.subheader("Relación costo-efectividad")
            fig = px.scatter(df, x="costo_por_peso_recuperado", y="tasa_exito",
                             size="monto_recuperado", color="indice_eficiencia",
                             hover_name="nombre_emisora",
                             hover_data=[col for col in ["nombre_banco", "tipo_servicio", "tipo_cobro"] if col in df.columns],
                             title="Relación entre costo y efectividad",
                             color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            cols_to_show = ["idBanco", "nombre_banco", "idEmisora", "nombre_emisora",
                            "tipo_servicio", "tipo_cobro", "tasa_exito",
                            "tasa_recuperacion_monto", "indice_potencial_recuperacion"]
            st.dataframe(df[[col for col in cols_to_show if col in df.columns]])
            st.subheader("Potencial de recuperación")
            fig = px.scatter(df, x="tasa_exito", y="tasa_recuperacion_monto",
                             size="monto_recuperado", color="indice_potencial_recuperacion",
                             hover_name="nombre_emisora",
                             hover_data=[col for col in ["nombre_banco", "tipo_servicio", "tipo_cobro"] if col in df.columns],
                             title="Relación entre tasa de éxito y recuperación",
                             color_continuous_scale="Plasma")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se pudieron cargar datos de estrategias óptimas con los filtros actuales.")

# Pestaña 4: Simulador (Adaptado para usar features si están disponibles)
with tab4:
    st.header("Simulador de escenarios")
    if conn and pipeline:
        # Obtener opciones para selectores
        try:
            bancos = conn.execute(f"SELECT DISTINCT idBanco, nombre_banco FROM {table_name}").fetchdf()
            tipos_servicio = conn.execute(f"SELECT DISTINCT tipo_servicio FROM {table_name}").fetchdf()
            tipos_cobro = conn.execute(f"SELECT DISTINCT tipo_cobro FROM {table_name}").fetchdf()
        except Exception as e:
            st.error(f"Error al obtener opciones para simulador: {e}")
            bancos = pd.DataFrame({"idBanco": [], "nombre_banco": []})
            tipos_servicio = pd.DataFrame({"tipo_servicio": []})
            tipos_cobro = pd.DataFrame({"tipo_cobro": []})

        col1, col2, col3 = st.columns(3)
        with col1:
            banco_id = st.selectbox("Banco", bancos["nombre_banco"].tolist() if not bancos.empty else [], key="sim_banco")
            if banco_id:
                try:
                    emisoras = conn.execute(f"SELECT DISTINCT idEmisora, nombre_emisora FROM {table_name} WHERE idBanco = '{banco_id}'").fetchdf()
                    emisora_id = st.selectbox("Emisora", emisoras["idEmisora"].tolist() if not emisoras.empty else [], key="sim_emisora")
                except Exception: emisora_id = None
            else: emisora_id = None
        with col2:
            tipo_servicio = st.selectbox("Tipo de servicio", tipos_servicio["tipo_servicio"].tolist() if not tipos_servicio.empty else [], key="sim_servicio")
            tipo_cobro = st.selectbox("Tipo de cobro", tipos_cobro["tipo_cobro"].tolist() if not tipos_cobro.empty else [], key="sim_cobro")
        with col3:
            monto_cobrar = st.number_input("Monto a cobrar", min_value=100.0, max_value=10000.0, value=1000.0, key="sim_monto")
            intentos_previos = st.number_input("Intentos previos", min_value=0, max_value=10, value=0, key="sim_intentos")
            hora_dia = st.slider("Hora del día (simulada)", 0, 23, 10, key="sim_hora")
            dia_semana = st.selectbox("Día de la semana (simulado)", list(range(7)), format_func=lambda x: ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"][x], key="sim_dia")

        if st.button("Simular escenario"):
            if banco_id and emisora_id and tipo_servicio and tipo_cobro:
                try:
                    # Obtener un registro base para completar features faltantes
                    base_record_query = f"SELECT * FROM {table_name} WHERE idBanco = '{banco_id}' LIMIT 1"
                    base_df = conn.execute(base_record_query).fetchdf()

                    if not base_df.empty:
                        escenario = base_df.iloc[[0]].copy() # Copiar estructura y valores por defecto

                        # Sobrescribir con valores del simulador
                        escenario["idBanco"] = banco_id
                        escenario["idEmisora"] = emisora_id
                        escenario["tipo_servicio"] = tipo_servicio
                        escenario["tipo_cobro"] = tipo_cobro
                        escenario["montoCobrar"] = monto_cobrar
                        escenario["montoExigible"] = monto_cobrar # Asumir por simplicidad

                        # Añadir/Actualizar features enriquecidas
                        if "numeroIntento" in escenario.columns: escenario["numeroIntento"] = intentos_previos + 1
                        if "hora_del_dia" in escenario.columns: escenario["hora_del_dia"] = hora_dia
                        if "dia_semana" in escenario.columns: escenario["dia_semana"] = dia_semana
                        # ... (añadir otras features si es necesario, como rango_monto)

                        # Asegurar que todas las columnas necesarias para el pipeline estén presentes
                        # (Puede requerir cargar X_train.columns desde el script de entrenamiento)
                        # Por simplicidad, asumimos que las columnas base son suficientes

                        with st.spinner("Calculando probabilidades..."):
                            prob = pipeline.predict_proba(escenario)[0][1]
                            costo_estimado = 2.5 # Valor simulado

                        st.subheader("Resultados de la simulación")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Probabilidad de éxito", f"{prob:.2%}")
                            st.metric("Costo estimado", f"${costo_estimado:.2f}")
                        with col2:
                            st.metric("Monto esperado", f"${prob*monto_cobrar:.2f}")
                            st.metric("ROI estimado", f"{(prob*monto_cobrar - costo_estimado)/costo_estimado:.2%}" if costo_estimado > 0 else "N/A")

                        # Recomendación (simplificada)
                        st.subheader("Recomendación")
                        if prob > 0.7: st.success("✅ Alta probabilidad de éxito. Recomendado.")
                        elif prob > 0.4: st.warning("⚠️ Probabilidad moderada. Evaluar costo/beneficio.")
                        else: st.error("❌ Baja probabilidad de éxito. No recomendado.")

                    else:
                        st.error("No se encontró un registro base para simular.")
                except Exception as e:
                    st.error(f"Error durante la simulación: {e}")
            else:
                st.warning("Complete todos los campos para simular.")
    else:
        st.warning("No se pudo cargar el modelo o conectar a la base de datos.")

# --- Información Adicional ---
st.sidebar.markdown("---")
st.sidebar.info("Aplicación con insights enriquecidos y procesamiento por lotes.")

# --- Limpieza Final ---

gc.collect()
