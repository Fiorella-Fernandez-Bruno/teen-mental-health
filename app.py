import io

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Teen Mental Health - EDA", page_icon="📊", layout="wide")

AUTOR = "Fiorella Fernández Bruno"
CURSO = "Especialización en Python for Analytics - DMC Institute"
ANIO = 2026

COLUMNAS_ESPERADAS = [
    "age", "gender", "daily_social_media_hours", "platform_usage", "sleep_hours",
    "screen_time_before_sleep", "academic_performance", "physical_activity",
    "social_interaction_level", "stress_level", "anxiety_level",
    "addiction_level", "depression_label",
]


def clasificar_variables(df):
    numericas = df.select_dtypes(include=np.number).columns.tolist()
    categoricas = df.select_dtypes(exclude=np.number).columns.tolist()
    if "depression_label" in numericas:
        numericas.remove("depression_label")
        categoricas.append("depression_label")
    return numericas, categoricas


class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def cargar_csv(archivo):
        try:
            df = pd.read_csv(archivo)
        except Exception as e:
            return None, f"No se pudo leer el archivo: {e}"
        if df.empty:
            return None, "El archivo está vacío."
        faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
        if faltantes:
            return None, f"Faltan columnas: {', '.join(faltantes)}"
        return df, None

    def dimensiones(self):
        return self.df.shape

    def info_texto(self):
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    def resumen_tipos(self):
        return pd.DataFrame({
            "Tipo de dato": self.df.dtypes.astype(str),
            "Valores nulos": self.df.isna().sum(),
            "Valores únicos": self.df.nunique(),
        })

    def duplicados(self):
        return int(self.df.duplicated().sum())

    def clasificar_variables(self):
        return clasificar_variables(self.df)

    def estadisticas(self, columnas):
        desc = self.df[columnas].describe().T
        desc["mediana"] = self.df[columnas].median()
        desc["moda"] = self.df[columnas].mode().iloc[0]
        desc["IQR"] = desc["75%"] - desc["25%"]
        desc["CV %"] = desc["std"] / desc["mean"] * 100
        return desc.round(2)

    def valores_extremos(self, columna):
        q1, q3 = self.df[columna].quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((self.df[columna] < lim_inf) | (self.df[columna] > lim_sup)).sum())
        return n, round(lim_inf, 2), round(lim_sup, 2)

    def faltantes(self):
        conteo = self.df.isna().sum()
        return pd.DataFrame({
            "Nulos": conteo,
            "% Nulos": (conteo / len(self.df) * 100).round(2),
        })

if "df" not in st.session_state:
    st.session_state.df = None

st.sidebar.title("📊 Teen Mental Health")
st.sidebar.caption("Análisis Exploratorio de Datos")
modulo = st.sidebar.radio(
    "Navegación",
    ["🏠 Home", "📂 Carga del dataset", "🔍 Análisis Exploratorio (EDA)", "✅ Conclusiones"],
)
st.sidebar.divider()
if st.session_state.df is not None:
    st.sidebar.success(f"Dataset cargado: {st.session_state.df.shape[0]:,} filas")
else:
    st.sidebar.warning("Dataset no cargado")
st.sidebar.caption(f"{AUTOR} · {ANIO}")


if modulo == "🏠 Home":
    st.title("Hábitos digitales y bienestar en adolescentes")
    st.subheader("Análisis Exploratorio de Datos con Python y Streamlit")
    st.markdown(
        "**Objetivo del análisis:** explorar cómo se relacionan los hábitos digitales "
        "(uso de redes sociales, pantalla antes de dormir), el descanso, la actividad física "
        "y la interacción social con las variables de bienestar registradas en el dataset. "
        "El enfoque es **exploratorio y educativo**: no se construyen modelos predictivos."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 👤 Autor")
        st.markdown(f"- **Nombre:** {AUTOR}\n- **Curso:** {CURSO}\n- **Año:** {ANIO}")
    with col2:
        st.markdown("#### 🛠️ Tecnologías utilizadas")
        st.markdown("- Python\n- Pandas y NumPy\n- Matplotlib y Seaborn\n- Streamlit")

    st.markdown("#### 📁 Sobre el dataset")
    st.markdown(
        "`Teen_Mental_Health_Dataset.csv` contiene **1,200 registros y 13 variables** sobre "
        "adolescentes de 13 a 19 años: uso diario de redes sociales, plataforma utilizada, "
        "horas de sueño, pantalla antes de dormir, rendimiento académico, actividad física, "
        "interacción social, escalas de estrés, ansiedad y dependencia (1 a 10) y la "
        "etiqueta binaria `depression_label`."
    )
    st.info("⚠️ Los resultados son exploratorios. No constituyen un diagnóstico clínico "
            "ni sustituyen la valoración de profesionales de la salud.")


elif modulo == "📂 Carga del dataset":
    st.title("📂 Carga del dataset")
    st.write("Sube el archivo `Teen_Mental_Health_Dataset.csv` para habilitar el análisis.")

    archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])
    if archivo is not None:
        df, error = DataAnalyzer.cargar_csv(archivo)
        if error:
            st.error(error)
        else:
            st.session_state.df = df
            st.success("✅ Archivo cargado y validado correctamente.")

    if st.session_state.df is not None:
        analizador = DataAnalyzer(st.session_state.df)
        filas, columnas = analizador.dimensiones()

        c1, c2, c3 = st.columns(3)
        c1.metric("Filas", f"{filas:,}")
        c2.metric("Columnas", columnas)
        c3.metric("Duplicados", analizador.duplicados())

        st.markdown("#### Vista previa")
        n = st.slider("Número de filas a mostrar", 5, 50, 5)
        st.dataframe(st.session_state.df.head(n), width="stretch")
    else:
        st.info("Aún no se ha cargado ningún archivo.")


elif modulo == "🔍 Análisis Exploratorio (EDA)":
    st.title("🔍 Análisis Exploratorio de Datos")

    if st.session_state.df is None:
        st.warning("Primero carga el dataset en el módulo **📂 Carga del dataset**.")
        st.stop()

    analizador = DataAnalyzer(st.session_state.df)
    df = analizador.df

    tabs = st.tabs([
        "1. Info general", "2. Variables", "3. Estadísticas", "4. Faltantes",
        "5. Numéricas", "6. Categóricas", "7. Num vs Cat", "8. Cat vs Cat",
        "9. Filtros", "10. Hallazgos",
    ])

    with tabs[0]:
        st.header("Ítem 1: Información general del dataset")
        st.write("Revisamos la estructura del archivo: qué columnas tiene, de qué tipo "
                 "es cada una y si hay datos vacíos o filas repetidas.")

        filas, columnas = analizador.dimensiones()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", f"{filas:,}")
        c2.metric("Columnas", columnas)
        c3.metric("Valores nulos", int(df.isna().sum().sum()))
        c4.metric("Duplicados", analizador.duplicados())

        st.subheader("Tipos de datos y valores nulos por columna")
        st.dataframe(analizador.resumen_tipos(), width="stretch")

        if st.checkbox("Mostrar salida completa de df.info()"):
            st.code(analizador.info_texto())

        st.success(f"El dataset tiene {filas:,} registros y {columnas} variables, "
                   f"sin valores nulos ni duplicados.")

    with tabs[1]:
        st.header("Ítem 2: Clasificación de variables")
        st.write("Separamos las columnas en **numéricas** (cantidades que se pueden "
                 "promediar) y **categóricas** (grupos o etiquetas) con la función "
                 "personalizada `clasificar_variables()`.")

        numericas, categoricas = analizador.clasificar_variables()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Variables numéricas", len(numericas))
            st.dataframe(pd.DataFrame({"Variable": numericas,
                                       "Tipo": [str(df[c].dtype) for c in numericas]}),
                         hide_index=True, width="stretch")
        with c2:
            st.metric("Variables categóricas", len(categoricas))
            st.dataframe(pd.DataFrame({"Variable": categoricas,
                                       "Categorías": [df[c].nunique() for c in categoricas]}),
                         hide_index=True, width="stretch")

        st.info("`depression_label` se guarda como número (0 y 1), pero es una etiqueta "
                "de sí/no. Por eso se clasifica como categórica.")

    with tabs[2]:
        st.header("Ítem 3: Estadísticas descriptivas")
        st.write("Resumimos cada variable numérica con medidas de centro (media, mediana, moda), "
                 "posición (cuartiles) y dispersión (desviación estándar, IQR y coeficiente "
                 "de variación).")

        numericas, _ = analizador.clasificar_variables()
        st.dataframe(analizador.estadisticas(numericas), width="stretch")

        st.subheader("Detalle por variable")
        variable = st.selectbox("Elige una variable", numericas)
        serie = df[variable]
        n_ext, lim_inf, lim_sup = analizador.valores_extremos(variable)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Media", f"{serie.mean():.2f}")
        c2.metric("Mediana", f"{serie.median():.2f}")
        c3.metric("Desv. estándar", f"{serie.std():.2f}")
        c4.metric("Valores extremos", n_ext)

        asimetria = serie.skew()
        if abs(asimetria) < 0.5:
            forma = f"la asimetría es {asimetria:.2f}, cercana a 0: la distribución es simétrica"
        elif asimetria > 0:
            forma = f"la asimetría es {asimetria:.2f}: hay una cola hacia valores altos"
        else:
            forma = f"la asimetría es {asimetria:.2f}: hay una cola hacia valores bajos"

        st.info(f"En **{variable}**, {forma}. El 50% central de los datos está entre "
                f"{serie.quantile(0.25):.2f} y {serie.quantile(0.75):.2f}. "
                f"Se consideran extremos los valores fuera de [{lim_inf}, {lim_sup}] "
                f"(regla de 1.5 × IQR): se encontraron {n_ext}.")

        st.success("Ninguna variable numérica presenta valores extremos según la regla del IQR. "
                   "En todas, la asimetría es cercana a 0 y la media es parecida a la mediana: "
                   "las distribuciones son simétricas. La actividad física y las escalas de estrés, "
                   "ansiedad y dependencia tienen la mayor dispersión relativa (CV mayor a 50%): "
                   "hay adolescentes en todo el rango de valores.")

    with tabs[3]:
        st.header("Ítem 4: Análisis de valores faltantes")
        st.write("Contamos cuántos datos vacíos tiene cada variable y qué porcentaje "
                 "representan sobre el total de registros.")

        tabla_nulos = analizador.faltantes()
        total_nulos = int(tabla_nulos["Nulos"].sum())

        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(tabla_nulos, width="stretch")
        with c2:
            st.metric("Total de valores nulos", total_nulos)
            st.metric("Variables con nulos", int((tabla_nulos["Nulos"] > 0).sum()))

        if total_nulos > 0:
            st.bar_chart(tabla_nulos["% Nulos"])
        else:
            st.info("No se muestra gráfico porque ninguna variable tiene valores faltantes.")

        st.success("El dataset no tiene valores faltantes, así que no es necesario eliminar "
                   "registros ni completar datos. Se conservan los 1,200 registros, y todos "
                   "los análisis usan la misma base. Si existieran nulos, las opciones serían "
                   "eliminar las filas (si fueran pocas) o completarlas con la mediana "
                   "(variables numéricas) o la moda (variables categóricas).")

    for i in range(4, 10):
        with tabs[i]:
            st.info("Pendiente.")


elif modulo == "✅ Conclusiones":
    st.title("✅ Conclusiones")

    if st.session_state.df is None:
        st.warning("Primero carga el dataset en el módulo **📂 Carga del dataset**.")
        st.stop()

    st.info("Pendiente.")
