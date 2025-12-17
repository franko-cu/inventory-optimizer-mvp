import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Inventory Optimizer AI", layout="wide")

# Función auxiliar para descargar CSV
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- SIDEBAR: CONFIGURACIÓN AVANZADA (Feature 3: Dinamismo) ---
st.sidebar.header("⚙️ Configuración del Modelo")
st.sidebar.write("Ajusta la sensibilidad de la IA:")

interval_width = st.sidebar.slider(
    "Intervalo de Confianza", 
    min_value=0.5, 
    max_value=0.99, 
    value=0.80, 
    help="Nivel de seguridad. Más alto = Mayor stock de seguridad (menos riesgo, más costo)."
)

lead_time = st.sidebar.number_input(
    "⏱️ Lead Time (Días proveedor)", 
    min_value=1, 
    value=3,
    help="Días que tarda en llegar el pedido."
)

st.title("📦 Motor de Sugerencias de Inventario AI")
st.markdown("Sube tus datos de ventas y deja que la IA calcule cuánto pedir para evitar quiebres de stock.")
st.markdown("---")

# --- 2. PLANTILLA DE EJEMPLO (Feature 2: Guía de Formato) ---
col_up, col_info = st.columns([2, 1])

with col_info:
    st.info("💡 ¿No tienes datos a mano?")
    # Crear dataframe de ejemplo
    example_data = pd.DataFrame({
        'Fecha': ['01/01/2024', '02/01/2024', '03/01/2024', '04/01/2024', '05/01/2024'],
        'ID_Producto': ['A100', 'A100', 'A100', 'A100', 'A100'],
        'Cantidad': [10, 15, 8, 12, 20]
    })
    csv_example = convert_df(example_data)
    
    st.download_button(
        label="📥 Descargar CSV de Ejemplo",
        data=csv_example,
        file_name='plantilla_ventas.csv',
        mime='text/csv',
        help="Usa este formato como guía."
    )

# --- 3. CARGA Y MAPEO DE DATOS (Feature 1: Flexibilidad) ---
with col_up:
    uploaded_file = st.file_uploader("Sube tu archivo histórico (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        # Cargar datos
        df = pd.read_csv(uploaded_file)
        
        with st.expander("🔎 Previsualizar datos cargados", expanded=True):
            st.dataframe(df.head(3))
            
        st.write("---")
        st.subheader("🔧 Configuración de Variables")
        
        # Selectores inteligentes de columnas
        c1, c2, c3 = st.columns(3)
        with c1:
            col_fecha = st.selectbox("1. ¿Cuál es la columna de FECHA?", df.columns, index=0)
        with c2:
            col_prod = st.selectbox("2. ¿Cuál es el ID del PRODUCTO?", df.columns, index=1 if len(df.columns) > 1 else 0)
        with c3:
            col_cant = st.selectbox("3. ¿Cuál es la CANTIDAD vendida?", df.columns, index=2 if len(df.columns) > 2 else 0)

        # Filtrado de producto
        productos = df[col_prod].unique()
        selected_prod = st.selectbox("📦 Selecciona el producto a analizar:", productos)
        
        if st.button("🚀 Analizar Inventario"):
            
            # --- 4. PREPARACIÓN DE DATOS (ETL) ---
            # Filtrar y copiar para no afectar el original
            df_filtered = df[df[col_prod] == selected_prod].copy()
            
            # Renombrar dinámicamente según la selección del usuario
            df_prophet = df_filtered.groupby(col_fecha)[col_cant].sum().reset_index()
            df_prophet.columns = ['ds', 'y'] # Prophet exige estos nombres
            
            # Manejo de fechas
            try:
                df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], dayfirst=True)
            except Exception as e:
                st.error(f"Error en formato de fechas: {e}")
                st.stop()

            # --- 5. MODELADO (PROPHET) ---
            with st.spinner(f'Entrenando IA con {interval_width*100}% de confianza...'):
                # Usamos el intervalo configurado en el Sidebar
                m = Prophet(interval_width=interval_width, daily_seasonality=True)
                m.fit(df_prophet)
                
                # Predicción
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)
                
                # --- 6. VISUALIZACIÓN Y RESULTADOS ---
                st.markdown("---")
                st.subheader(f"📈 Proyección de Demanda: {selected_prod}")
                
                # Gráfico interactivo
                fig = plot_plotly(m, forecast)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # LÓGICA DE NEGOCIO
                # Calcular demanda futura (Lead Time)
                futuro_lead_time = forecast.tail(30).head(lead_time)
                # Usamos yhat_upper (escenario pesimista de demanda) para seguridad
                punto_reorden = futuro_lead_time['yhat_upper'].sum()
                
                # Panel de Resultados
                st.success("✅ Análisis Completado")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("Punto de Reorden (Sugerido)", f"{punto_reorden:.0f} u.", help=f"Cobertura para {lead_time} días")
                with col_res2:
                    st.metric("Confianza del Modelo", f"{interval_width*100:.0f}%")
                with col_res3:
                    st.metric("Lead Time Configurado", f"{lead_time} días")

                # Mensaje de Acción
                st.info(f"💡 Interpretación: Si tu stock actual es menor a **{punto_reorden:.0f} unidades**, deberías hacer un pedido hoy para cubrir la demanda de los próximos {lead_time} días.")

                # Feature Extra: Exportar Resultados
                st.write("---")
                results_csv = convert_df(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))
                st.download_button(
                    label="📥 Descargar Predicción Detallada",
                    data=results_csv,
                    file_name=f'prediccion_{selected_prod}.csv',
                    mime='text/csv'
                )

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

else:
    st.info("👆 Sube un archivo CSV para comenzar o descarga la plantilla de ejemplo.")