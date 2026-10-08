"""
Caso de Estudio N°4 - Teen Mental Health
EDA interactivo con Streamlit
Especialización en Python for Analytics - DMC Institute
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# =====================================================================
# CONFIGURACIÓN GENERAL
# =====================================================================
st.set_page_config(page_title="Teen Mental Health - EDA", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid")

AUTOR = "Fiorella Fernández Bruno"
CURSO = "Especialización en Python for Analytics - DMC Institute"
ANIO = 2026

COLUMNAS_ESPERADAS = [
    "age", "gender", "daily_social_media_hours", "platform_usage", "sleep_hours",
    "screen_time_before_sleep", "academic_performance", "physical_activity",
    "social_interaction_level", "stress_level", "anxiety_level",
    "addiction_level", "depression_label",
]
VARIABLES_BIENESTAR = ["stress_level", "anxiety_level", "addiction_level", "sleep_hours"]
VARIABLES_HABITOS = ["daily_social_media_hours", "screen_time_before_sleep",
                     "physical_activity", "academic_performance"]


# =====================================================================
# CLASE PRINCIPAL (POO)
# =====================================================================
class DataAnalyzer:
    """Encapsula carga, validación, clasificación, estadísticas,
    visualizaciones y filtros del dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ---------- Carga y validación ----------
    @staticmethod
    def cargar_csv(archivo):
        """Lee el CSV subido. Devuelve (DataFrame, mensaje_error)."""
        try:
            df = pd.read_csv(archivo)
        except Exception as e:
            return None, f"No se pudo leer el archivo: {e}"
        if df.empty:
            return None, "El archivo está vacío."
        faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
        if faltantes:
            return None, f"Faltan columnas esperadas: {', '.join(faltantes)}"
        return df, None

    def dimensiones(self):
        return self.df.shape

    # ---------- Información general ----------
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

    # ---------- Clasificación de variables ----------
    def clasificar_variables(self):
        """Separa numéricas y categóricas. depression_label es 0/1:
        se trata como categórica (etiqueta), aunque esté guardada como entero."""
        numericas = self.df.select_dtypes(include=np.number).columns.tolist()
        categoricas = self.df.select_dtypes(exclude=np.number).columns.tolist()
        if "depression_label" in numericas:
            numericas.remove("depression_label")
            categoricas.append("depression_label")
        return numericas, categoricas

    # ---------- Estadísticas ----------
    def estadisticas(self, columnas=None):
        columnas = columnas or self.clasificar_variables()[0]
        desc = self.df[columnas].describe().T
        desc["mediana"] = self.df[columnas].median()
        desc["moda"] = self.df[columnas].mode().iloc[0]
        desc["rango_IQR"] = desc["75%"] - desc["25%"]
        return desc.round(2)

    def valores_extremos(self, columna):
        """Cuenta valores fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
        q1, q3 = self.df[columna].quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((self.df[columna] < lim_inf) | (self.df[columna] > lim_sup)).sum())
        return n, lim_inf, lim_sup

    def faltantes(self):
        conteo = self.df.isna().sum()
        return pd.DataFrame({"Nulos": conteo,
                             "% Nulos": (conteo / len(self.df) * 100).round(2)})

    def frecuencias(self, columna):
        conteo = self.df[columna].value_counts()
        return pd.DataFrame({"Frecuencia": conteo,
                             "Proporción (%)": (conteo / len(self.df) * 100).round(1)})

    # ---------- Filtros y comparaciones ----------
    def filtrar(self, edad=None, generos=None, plataformas=None, interaccion=None):
        d = self.df
        if edad:
            d = d[d["age"].between(edad[0], edad[1])]
        if generos:
            d = d[d["gender"].isin(generos)]
        if plataformas:
            d = d[d["platform_usage"].isin(plataformas)]
        if interaccion:
            d = d[d["social_interaction_level"].isin(interaccion)]
        return d

    def comparar_grupos(self, columna_num, columna_grupo):
        return (self.df.groupby(columna_grupo)[columna_num]
                .agg(["count", "mean", "median", "std"]).round(2))

    def tabla_cruzada(self, fila, columna, normalizar=True):
        return (pd.crosstab(self.df[fila], self.df[columna],
                            normalize="index" if normalizar else False) * (100 if normalizar else 1)).round(1)

    # ---------- Visualizaciones ----------
    def histograma(self, columna, bins=20):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(self.df[columna], bins=bins, kde=True, ax=ax, color="#3B6FB6")
        ax.axvline(self.df[columna].mean(), color="#D9534F", ls="--", label="Media")
        ax.axvline(self.df[columna].median(), color="#2E8B57", ls=":", label="Mediana")
        ax.set_title(f"Distribución de {columna}")
        ax.legend()
        fig.tight_layout()
        return fig

    def barras(self, columna):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        orden = self.df[columna].value_counts().index
        sns.countplot(data=self.df, x=columna, order=orden, ax=ax, color="#3B6FB6")
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)
        ax.set_title(f"Frecuencia de {columna}")
        fig.tight_layout()
        return fig

    def boxplot_por_grupo(self, columna_num, columna_grupo):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.boxplot(data=self.df, x=columna_grupo, y=columna_num, ax=ax, color="#9DB9E0")
        ax.set_title(f"{columna_num} según {columna_grupo}")
        fig.tight_layout()
        return fig

    def barras_apiladas(self, fila, columna):
        tabla = self.tabla_cruzada(fila, columna)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        tabla.plot(kind="bar", stacked=True, ax=ax, colormap="Blues", edgecolor="white")
        ax.set_ylabel("% dentro de cada grupo")
        ax.set_title(f"{columna} por {fila}")
        ax.legend(title=columna, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=0)
        fig.tight_layout()
        return fig


# =====================================================================
# ESTADO DE LA SESIÓN (para no perder el archivo al navegar)
# =====================================================================
if "df" not in st.session_state:
    st.session_state.df = None

# =====================================================================
# SIDEBAR - MENÚ PRINCIPAL
# =====================================================================
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

# =====================================================================
# MÓDULO 1: HOME
# =====================================================================
if modulo == "🏠 Home":
    st.title("Hábitos digitales y bienestar en adolescentes")
    st.subheader("Análisis Exploratorio de Datos con Python y Streamlit")

    st.markdown(
        """
        **Objetivo del análisis:** explorar cómo se relacionan los hábitos digitales
        (uso de redes sociales, pantalla antes de dormir), el descanso, la actividad física
        y la interacción social con las variables de bienestar registradas en el dataset.
        El enfoque es **exploratorio y educativo**: no se construyen modelos predictivos.
        """
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
        """
        `Teen_Mental_Health_Dataset.csv` contiene **1,200 registros y 13 variables** sobre
        adolescentes de 13 a 19 años: uso diario de redes sociales, plataforma utilizada,
        horas de sueño, pantalla antes de dormir, rendimiento académico, actividad física,
        interacción social, escalas de estrés, ansiedad y dependencia (1 a 10) y la
        etiqueta binaria `depression_label`.
        """
    )
    st.info("⚠️ Los resultados son exploratorios. No constituyen un diagnóstico clínico "
            "ni sustituyen la valoración de profesionales de la salud.")

# =====================================================================
# MÓDULO 2: CARGA DEL DATASET
# =====================================================================
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

# =====================================================================
# MÓDULO 3: EDA
# =====================================================================
elif modulo == "🔍 Análisis Exploratorio (EDA)":
    st.title("🔍 Análisis Exploratorio de Datos")

    if st.session_state.df is None:
        st.warning("Primero carga el dataset en el módulo **📂 Carga del dataset**.")
        st.stop()

    analizador = DataAnalyzer(st.session_state.df)
    st.info("Los 10 ítems de análisis se construyen en el siguiente paso.")

# =====================================================================
# MÓDULO 4: CONCLUSIONES
# =====================================================================
elif modulo == "✅ Conclusiones":
    st.title("✅ Conclusiones")

    if st.session_state.df is None:
        st.warning("Primero carga el dataset en el módulo **📂 Carga del dataset**.")
        st.stop()

    st.info("Las 5 conclusiones se redactan al terminar el EDA.")
