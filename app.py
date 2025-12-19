import streamlit as st
import pandas as pd
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

st.set_page_config(page_title="Optimizador de inventarios con Machine Learning", layout="wide")

def convertir_df(df):
    return df.to_csv(index=False).encode('utf-8')

if 'manual_rows' not in st.session_state:
    st.session_state.manual_rows = []

st.sidebar.header("Configuración del Modelo")
st.sidebar.write("Ajusta la sensibilidad de los cálculos:")

interval_width = st.sidebar.slider(
    "Intervalo de Confianza",
    min_value=0.5,
    max_value=0.99,
    value=0.80,
    help="Nivel de seguridad. Más alto = Mayor stock de seguridad (menos riesgo, más costo)."
)

tiempo_de_entrega = st.sidebar.number_input(
    "Tiempo de entrega",
    min_value=1,
    value=3,
    help="Días que tarda en llegar el pedido."
)

st.title("Motor de Sugerencias de Inventario AI")
st.markdown("Sube tus datos de ventas para realizar los cálculos.")
st.markdown("---")

col_up, col_info = st.columns([2, 1])

with col_info:
    st.info("No tienes datos a mano?")
    datos_ejemplo = pd.DataFrame({
        'Fecha': ['01/01/2024', '02/01/2024', '03/01/2024', '04/01/2024', '05/01/2024'],
        'ID_Producto': ['A100', 'A100', 'A100', 'A100', 'A100'],
        'Cantidad': [10, 15, 8, 12, 20]
    })
    csv_example = convertir_df(datos_ejemplo)
    st.download_button(
        label="Descargar CSV de Ejemplo",
        data=csv_example,
        file_name='plantilla_ventas.csv',
        mime='text/csv',
        help="Usa este formato como guía."
    )

with col_up:
    archivo_cargado = st.file_uploader("Sube tu archivo histórico (CSV)", type=['csv'])

st.markdown("### Agregar productos manualmente")
with st.expander("Cargar filas a mano", expanded=False):
    with st.form("manual_entry"):
        fecha_input = st.date_input("Fecha", value=date.today())
        prod_input = st.text_input("ID del producto", value="A100")
        cant_input = st.number_input("Cantidad vendida", min_value=0, step=1, value=1)
        submitted = st.form_submit_button("Agregar fila")
        if submitted:
            st.session_state.manual_rows.append({
                'Fecha': fecha_input.strftime("%d/%m/%Y"),
                'ID_Producto': prod_input,
                'Cantidad': cant_input
            })
            st.success("Fila agregada")

df = None

if archivo_cargado is not None:
    try:
        df = pd.read_csv(archivo_cargado)
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

if st.session_state.manual_rows:
    manual_df = pd.DataFrame(st.session_state.manual_rows)
    st.write("Productos ingresados manualmente:")
    st.dataframe(manual_df.tail(5))
    df = manual_df if df is None else pd.concat([df, manual_df], ignore_index=True)

if df is not None:
    with st.expander("Previsualizar datos cargados", expanded=True):
        st.dataframe(df.head(5))

    st.write("---")
    st.subheader("Configuración de Variables")

    c1, c2, c3 = st.columns(3)
    with c1:
        col_fecha = st.selectbox("Columna de fecha", df.columns, index=0)
    with c2:
        col_prod = st.selectbox("ID del producto", df.columns, index=1 if len(df.columns) > 1 else 0)
    with c3:
        col_cant = st.selectbox("Cantidad vendida", df.columns, index=2 if len(df.columns) > 2 else 0)

    productos = df[col_prod].unique()
    selected_prod = st.selectbox("Selecciona el producto a analizar", productos)

    if st.button("Analizar Inventario"):
        df_filtered = df[df[col_prod] == selected_prod].copy()

        df_prophet = df_filtered.groupby(col_fecha)[col_cant].sum().reset_index()
        df_prophet.columns = ['ds', 'y']

        try:
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], dayfirst=True)
        except Exception as e:
            st.error(f"Error en formato de fechas: {e}")
            st.stop()

        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet = df_prophet.dropna(subset=['ds', 'y'])
        df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

        if len(df_prophet) < 2:
            st.error("Se necesitan al menos 2 fechas con datos válidos para entrenar el modelo.")
            st.stop()

        with st.spinner(f'Entrenando IA con {interval_width*100}% de confianza...'):
            m = Prophet(interval_width=interval_width, daily_seasonality=True)
            m.fit(df_prophet)

            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)

            st.markdown("---")
            st.subheader(f"Proyección de Demanda: {selected_prod}")

            fig = plot_plotly(m, forecast)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            futuro_tiempo_de_entrega = forecast.tail(30).head(tiempo_de_entrega)
            punto_reorden = futuro_tiempo_de_entrega['yhat_upper'].sum()

            st.success("Análisis Completado")

            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Punto de Reorden Sugerido", f"{punto_reorden:.0f} u.")
            with col_res2:
                st.metric("Confianza del Modelo", f"{interval_width*100:.0f}%")
            with col_res3:
                st.metric("Lead Time", f"{tiempo_de_entrega} días")

            st.info(
                f"Si tu stock actual es menor a {punto_reorden:.0f} unidades, deberías hacer un pedido hoy para cubrir la demanda de los próximos {tiempo_de_entrega} días."
            )

            st.write("---")
            results_csv = convertir_df(
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
            )
            st.download_button(
                label="Descargar Predicción Detallada",
                data=results_csv,
                file_name=f'prediccion_{selected_prod}.csv',
                mime='text/csv'
            )

else:
    st.info("Sube un archivo CSV, ingresa filas manualmente o descarga la plantilla de ejemplo.")
