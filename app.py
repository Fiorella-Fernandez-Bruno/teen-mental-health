import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

    def histograma(self, columna, bins=15):
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(self.df[columna], bins=bins, kde=True, color="#3B6FB6", ax=ax)
        ax.axvline(self.df[columna].mean(), color="#D9534F", linestyle="--", label="Media")
        ax.axvline(self.df[columna].median(), color="#2E8B57", linestyle=":", label="Mediana")
        ax.set_title(f"Distribución de {columna}")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        fig.tight_layout()
        return fig

    def boxplot_escalas(self, columnas):
        datos = self.df[columnas].melt(var_name="Escala", value_name="Valor")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=datos, x="Escala", y="Valor", color="#9DB9E0", ax=ax)
        ax.set_title("Comparación de escalas (1 a 10)")
        fig.tight_layout()
        return fig

    def frecuencias(self, columna):
        conteo = self.df[columna].value_counts()
        return pd.DataFrame({
            "Frecuencia": conteo,
            "Proporción (%)": (conteo / len(self.df) * 100).round(1),
        })

    def barras(self, columna):
        tabla = self.frecuencias(columna)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=tabla.index.astype(str), y=tabla["Frecuencia"], color="#3B6FB6", ax=ax)
        for i, (n, p) in enumerate(zip(tabla["Frecuencia"], tabla["Proporción (%)"])):
            ax.text(i, n, f"{n} ({p}%)", ha="center", va="bottom")
        ax.set_ylim(0, tabla["Frecuencia"].max() * 1.15)
        ax.set_title(f"Frecuencia de {columna}")
        ax.set_xlabel(columna)
        ax.set_ylabel("Frecuencia")
        fig.tight_layout()
        return fig

    def comparar_grupos(self, columna_num, columna_grupo):
        return (self.df.groupby(columna_grupo)[columna_num]
                .agg(["count", "mean", "median", "std", "min", "max"]).round(2))

    def boxplot_por_grupo(self, columna_num, columna_grupo):
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=self.df, x=columna_grupo, y=columna_num, color="#9DB9E0", ax=ax)
        ax.set_title(f"{columna_num} según {columna_grupo}")
        fig.tight_layout()
        return fig

    def tabla_cruzada(self, fila, columna, porcentaje=True):
        if porcentaje:
            return (pd.crosstab(self.df[fila], self.df[columna], normalize="index") * 100).round(1)
        return pd.crosstab(self.df[fila], self.df[columna], margins=True, margins_name="Total")

    def grafico_cruzado(self, fila, columna):
        tabla = self.tabla_cruzada(fila, columna)
        fig, ax = plt.subplots(figsize=(7, 4))
        if columna == "depression_label":
            sns.barplot(x=tabla.index, y=tabla[1], color="#D9534F", ax=ax)
            for i, valor in enumerate(tabla[1]):
                ax.text(i, valor, f"{valor}%", ha="center", va="bottom")
            ax.set_ylim(0, tabla[1].max() * 1.3)
            ax.set_ylabel("% con depression_label = 1")
        else:
            tabla.plot(kind="bar", ax=ax, colormap="Blues", edgecolor="black")
            ax.set_ylabel("% dentro de cada grupo")
            ax.legend(title=columna)
            plt.xticks(rotation=0)
        ax.set_title(f"{columna} según {fila}")
        ax.set_xlabel(fila)
        fig.tight_layout()
        return fig

    def filtrar(self, edades, generos, plataformas, interacciones):
        d = self.df
        d = d[d["age"].between(edades[0], edades[1])]
        d = d[d["gender"].isin(generos)]
        d = d[d["platform_usage"].isin(plataformas)]
        d = d[d["social_interaction_level"].isin(interacciones)]
        return d

    def promedio_por_tramos(self, habito, bienestar):
        tramos = pd.qcut(self.df[habito], 4, duplicates="drop")
        return self.df.groupby(tramos, observed=True)[bienestar].agg(["count", "mean"]).round(2)

    def grafico_tramos(self, habito, bienestar):
        tabla = self.promedio_por_tramos(habito, bienestar)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=tabla.index.astype(str), y=tabla["mean"], color="#3B6FB6", ax=ax)
        ax.set_title(f"Promedio de {bienestar} por tramos de {habito}")
        ax.set_xlabel(f"Tramos de {habito}")
        ax.set_ylabel(f"Promedio de {bienestar}")
        fig.tight_layout()
        return fig

    def contar_condiciones(self):
        return ((self.df["daily_social_media_hours"] > 5).astype(int)
                + (self.df["sleep_hours"] < 6).astype(int)
                + (self.df["stress_level"] >= 7).astype(int)
                + (self.df["anxiety_level"] >= 7).astype(int))

    def resumen_condiciones(self):
        condiciones = self.contar_condiciones()
        tabla = self.df.groupby(condiciones)["depression_label"].agg(["count", "sum"])
        tabla.columns = ["Adolescentes", "Con etiqueta 1"]
        tabla["% con etiqueta 1"] = (tabla["Con etiqueta 1"] / tabla["Adolescentes"] * 100).round(1)
        tabla.index.name = "Condiciones cumplidas"
        return tabla

    def grafico_condiciones(self):
        tabla = self.resumen_condiciones()
        colores = ["#D9534F" if p > 0 else "#9DB9E0" for p in tabla["% con etiqueta 1"]]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(tabla.index.astype(str), tabla["Adolescentes"], color=colores, edgecolor="black")
        for i, (n, p) in enumerate(zip(tabla["Adolescentes"], tabla["% con etiqueta 1"])):
            ax.text(i, n, f"{n}\n({p}% con etiqueta 1)", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, tabla["Adolescentes"].max() * 1.25)
        ax.set_title("Adolescentes según cuántas condiciones cumplen")
        ax.set_xlabel("Número de condiciones cumplidas (de 4)")
        ax.set_ylabel("Adolescentes")
        fig.tight_layout()
        return fig

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

    with tabs[4]:
        st.header("Ítem 5: Distribución de variables numéricas")
        st.write("Con histogramas vemos cómo se reparten los valores de cada variable: "
                 "dónde se concentran, si son simétricos y si hay valores alejados.")

        numericas, _ = analizador.clasificar_variables()
        c1, c2 = st.columns([1, 2])
        with c1:
            variable_hist = st.selectbox("Variable", numericas, key="hist_var")
            bins = st.slider("Número de barras", 5, 40, 15)
            serie = df[variable_hist]
            st.metric("Media", f"{serie.mean():.2f}")
            st.metric("Mediana", f"{serie.median():.2f}")
            st.metric("Asimetría", f"{serie.skew():.2f}")
        with c2:
            st.pyplot(analizador.histograma(variable_hist, bins))

        st.info("En todas las variables las barras tienen alturas parecidas: los datos están "
                "repartidos de forma **pareja (distribución casi uniforme)**, sin un pico central. "
                "La media y la mediana coinciden (asimetría cercana a 0) y no hay valores "
                "aislados en los extremos.")

        st.subheader("Comparación de escalas de estrés, ansiedad y dependencia")
        escalas = ["stress_level", "anxiety_level", "addiction_level"]
        c1, c2 = st.columns([2, 1])
        with c1:
            st.pyplot(analizador.boxplot_escalas(escalas))
        with c2:
            st.dataframe(df[escalas].agg(["mean", "median", "std"]).T.round(2), width="stretch")

        st.success("Las tres escalas usan el mismo rango (1 a 10) y se comportan de forma muy "
                   "parecida: medianas entre 5 y 6, y la mitad central de los datos entre 3 y 8. "
                   "Cada nivel tiene aproximadamente la misma cantidad de adolescentes. Esta "
                   "comparación es descriptiva y no constituye un diagnóstico clínico.")

    with tabs[5]:
        st.header("Ítem 6: Análisis de variables categóricas")
        st.write("Contamos cuántos adolescentes hay en cada categoría y qué proporción "
                 "representan del total.")

        _, categoricas = analizador.clasificar_variables()
        variable_cat = st.selectbox("Variable categórica", categoricas, key="cat_var")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.pyplot(analizador.barras(variable_cat))
        with c2:
            st.dataframe(analizador.frecuencias(variable_cat), width="stretch")

        st.info("**gender**, **platform_usage** y **social_interaction_level** están "
                "equilibradas: cada categoría reúne entre el 30% y el 50% de los registros. "
                "Esto permite comparar los grupos entre sí en igualdad de condiciones.")

        st.subheader("Categoría clave: depression_label")
        positivos = int(df["depression_label"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Etiqueta 0 (ausencia)", f"{len(df) - positivos:,}")
        c2.metric("Etiqueta 1 (presencia)", positivos)
        c3.metric("% con etiqueta 1", f"{positivos / len(df) * 100:.1f}%")

        st.success(f"Solo {positivos} de {len(df):,} registros tienen depression_label = 1. "
                   "Es un grupo pequeño, por lo que en las comparaciones siguientes se usarán "
                   "porcentajes y se interpretarán los resultados con cautela.")

    with tabs[6]:
        st.header("Ítem 7: Análisis bivariado (numérico vs categórico)")
        st.write("Comparamos una variable numérica entre los grupos de una variable "
                 "categórica, para ver si sus valores cambian de un grupo a otro.")

        st.subheader("Promedios según depression_label")
        variables_clave = ["daily_social_media_hours", "sleep_hours",
                           "academic_performance", "physical_activity"]
        resumen = df.groupby("depression_label")[variables_clave].mean().T.round(2)
        resumen.columns = ["Etiqueta 0", "Etiqueta 1"]
        resumen["Diferencia"] = (resumen["Etiqueta 1"] - resumen["Etiqueta 0"]).round(2)
        st.dataframe(resumen, width="stretch")

        st.subheader("Comparación interactiva")
        numericas, categoricas = analizador.clasificar_variables()
        c1, c2 = st.columns(2)
        with c1:
            var_num = st.selectbox("Variable numérica", numericas,
                                   index=numericas.index("daily_social_media_hours"), key="biv_num")
        with c2:
            var_grupo = st.selectbox("Agrupar por", categoricas,
                                     index=categoricas.index("depression_label"), key="biv_cat")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.pyplot(analizador.boxplot_por_grupo(var_num, var_grupo))
        with c2:
            st.dataframe(analizador.comparar_grupos(var_num, var_grupo), width="stretch")

        st.success("El grupo con depression_label = 1 usa más redes sociales (mediana de 7.0 "
                   "horas frente a 4.4) y duerme menos (mediana de 4.6 horas frente a 6.5). "
                   "En cambio, el rendimiento académico y la actividad física son prácticamente "
                   "iguales en ambos grupos. Son asociaciones observadas en los datos, no "
                   "relaciones de causa y efecto.")

    with tabs[7]:
        st.header("Ítem 8: Análisis bivariado (categórico vs categórico)")
        st.write("Cruzamos dos variables categóricas para ver si la distribución de una "
                 "cambia según los grupos de la otra. Usamos porcentajes por fila para "
                 "comparar grupos de distinto tamaño.")

        comparaciones = {
            "Plataforma vs depression_label": ("platform_usage", "depression_label"),
            "Interacción social vs depression_label": ("social_interaction_level", "depression_label"),
            "Género vs plataforma": ("gender", "platform_usage"),
        }
        eleccion = st.selectbox("Comparación", list(comparaciones.keys()), key="cruce")
        fila, columna = comparaciones[eleccion]

        c1, c2 = st.columns([2, 1])
        with c1:
            st.pyplot(analizador.grafico_cruzado(fila, columna))
        with c2:
            st.write("**Porcentaje por fila**")
            st.dataframe(analizador.tabla_cruzada(fila, columna), width="stretch")
            if st.checkbox("Mostrar conteos absolutos", key="conteos"):
                st.dataframe(analizador.tabla_cruzada(fila, columna, porcentaje=False),
                             width="stretch")

        st.success("El porcentaje con depression_label = 1 es muy parecido en todas las "
                   "plataformas (entre 2.3% y 3.0%) y en todos los niveles de interacción "
                   "social (entre 2.2% y 2.9%). Hombres y mujeres usan Instagram, TikTok y "
                   "ambas en proporciones casi iguales (alrededor de un tercio cada una). "
                   "En este dataset, la plataforma y el nivel de interacción social no "
                   "marcan diferencias; las diferencias aparecen en las horas de uso y de "
                   "sueño (ítem 7).")

    with tabs[8]:
        st.header("Ítem 9: Análisis basado en parámetros seleccionados")
        st.write("Filtra a los adolescentes según su perfil y elige una variable de bienestar "
                 "y una de hábitos para ver cómo se relacionan en el grupo seleccionado.")

        c1, c2 = st.columns(2)
        with c1:
            edades = st.slider("Rango de edad", 13, 19, (13, 19))
            generos = st.multiselect("Género", ["female", "male"], default=["female", "male"])
        with c2:
            plataformas = st.multiselect("Plataforma", ["Instagram", "TikTok", "Both"],
                                         default=["Instagram", "TikTok", "Both"])
            interacciones = st.multiselect("Interacción social", ["low", "medium", "high"],
                                           default=["low", "medium", "high"])

        c1, c2 = st.columns(2)
        with c1:
            bienestar = st.selectbox("Variable de bienestar",
                                     ["stress_level", "anxiety_level", "addiction_level",
                                      "academic_performance"], key="bienestar")
        with c2:
            habito = st.selectbox("Variable de hábitos",
                                  ["daily_social_media_hours", "screen_time_before_sleep",
                                   "sleep_hours", "physical_activity"], key="habito")

        filtrado = DataAnalyzer(analizador.filtrar(edades, generos, plataformas, interacciones))
        datos = filtrado.df

        if len(datos) < 20:
            st.warning("Hay muy pocos registros con estos filtros. Amplía la selección.")
        else:
            correlacion = datos[habito].corr(datos[bienestar])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Registros", f"{len(datos):,}")
            c2.metric(f"Promedio {habito}", f"{datos[habito].mean():.2f}")
            c3.metric(f"Promedio {bienestar}", f"{datos[bienestar].mean():.2f}")
            c4.metric("Correlación", f"{correlacion:.2f}")

            c1, c2 = st.columns([2, 1])
            with c1:
                st.pyplot(filtrado.grafico_tramos(habito, bienestar))
            with c2:
                st.write("**Promedio por tramo**")
                st.dataframe(filtrado.promedio_por_tramos(habito, bienestar), width="stretch")

            if abs(correlacion) < 0.1:
                fuerza = "prácticamente no hay relación"
            elif abs(correlacion) < 0.3:
                fuerza = "hay una relación débil"
            else:
                fuerza = "hay una relación moderada o fuerte"

            st.info(f"En los {len(datos):,} registros seleccionados, la correlación entre "
                    f"**{habito}** y **{bienestar}** es {correlacion:.2f}: {fuerza}. "
                    f"Si las barras tienen alturas parecidas, el promedio de {bienestar} "
                    f"no cambia aunque cambie {habito}.")

        st.success("Con todos los registros, las correlaciones entre los hábitos y las escalas "
                   "de estrés, ansiedad, dependencia y rendimiento académico son cercanas a 0. "
                   "Es decir, por sí solos los hábitos no se reflejan en el promedio de estas "
                   "escalas; la asociación más clara está con depression_label (ítem 7).")

    with tabs[9]:
        st.header("Ítem 10: Hallazgos clave")
        st.write("Reunimos lo encontrado en los ítems anteriores. Contamos cuántas de estas "
                 "4 condiciones cumple cada adolescente: más de 5 horas diarias en redes, "
                 "menos de 6 horas de sueño, estrés de 7 o más y ansiedad de 7 o más.")

        resumen = analizador.resumen_condiciones()
        c1, c2, c3 = st.columns(3)
        c1.metric("Cumplen las 4 condiciones", int(resumen.loc[4, "Adolescentes"]))
        c2.metric("De ellos, con etiqueta 1", f"{resumen.loc[4, '% con etiqueta 1']}%")
        c3.metric("Con etiqueta 1 entre quienes cumplen 3 o menos",
                  f"{resumen.loc[:3, '% con etiqueta 1'].max()}%")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.pyplot(analizador.grafico_condiciones())
        with c2:
            st.dataframe(resumen, width="stretch")

        st.subheader("Insights principales")
        st.markdown(
            "1. **La etiqueta aparece solo cuando coinciden las 4 condiciones.** Los 31 casos "
            "con depression_label = 1 cumplen las 4, y nadie con 3 o menos la tiene.\n"
            "2. **Ningún factor aislado basta.** 198 adolescentes cumplen 3 condiciones y "
            "ninguno tiene la etiqueta.\n"
            "3. **Lo que importa es el tiempo, no la plataforma.** Instagram, TikTok o ambas "
            "muestran porcentajes casi iguales (ítem 8).\n"
            "4. **Las horas de redes y de sueño marcan la diferencia** entre grupos (ítem 7), "
            "aunque por sí solas no cambian el promedio de estrés o ansiedad (ítem 9).\n"
            "5. **Rendimiento académico y actividad física no se diferencian** entre grupos."
        )

        st.subheader("Recomendaciones de interpretación")
        st.markdown(
            "- Mirar la **combinación** de hábitos y bienestar, no cada variable por separado.\n"
            "- Priorizar acciones sobre **horas de uso de redes y horas de sueño**, más que "
            "sobre una plataforma específica.\n"
            "- Interpretar con cautela: el grupo con etiqueta 1 es pequeño (2.6%) y el patrón "
            "es propio de este dataset.\n"
            "- Usar estos resultados para **orientar preguntas y seguimiento**, no para "
            "predecir ni diagnosticar."
        )

        st.info("⚠️ Estos hallazgos son exploratorios y educativos. No constituyen un "
                "diagnóstico clínico ni sustituyen la valoración de profesionales de la salud.")


elif modulo == "✅ Conclusiones":
    st.title("✅ Conclusiones")

    if st.session_state.df is None:
        st.warning("Primero carga el dataset en el módulo **📂 Carga del dataset**.")
        st.stop()

    analizador = DataAnalyzer(st.session_state.df)
    st.write("Cinco conclusiones basadas en el análisis exploratorio. Cada una se vincula con "
             "una evidencia visual o estadística; despliega cada una para verla.")

    with st.expander("1. La base de datos es confiable y completa", expanded=True):
        st.write("Los 1,200 registros están completos: no hay valores nulos, duplicados ni "
                 "valores extremos, y todas las variables tienen distribuciones simétricas. "
                 "**Decisión:** los resultados pueden usarse sin limpieza previa, sabiendo "
                 "que representan a todo el grupo analizado.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", f"{len(analizador.df):,}")
        c2.metric("Valores nulos", int(analizador.df.isna().sum().sum()))
        c3.metric("Duplicados", analizador.duplicados())
        st.caption("Evidencia: ítems 1, 3 y 4.")

    with st.expander("2. Más horas en redes sociales se asocian con la etiqueta"):
        st.write("El grupo con depression_label = 1 usa redes 7.0 horas al día (mediana), "
                 "frente a 4.4 horas del resto. Ningún adolescente con 5 horas o menos tiene "
                 "la etiqueta. **Decisión:** las acciones de prevención deberían enfocarse en "
                 "el tiempo diario de uso de redes.")
        st.pyplot(analizador.boxplot_por_grupo("daily_social_media_hours", "depression_label"))
        st.caption("Evidencia: ítem 7.")

    with st.expander("3. Dormir menos también se asocia con la etiqueta"):
        st.write("El grupo con depression_label = 1 duerme 4.6 horas (mediana), frente a 6.5 "
                 "horas del resto. Nadie que duerma 6 horas o más tiene la etiqueta. "
                 "**Decisión:** promover hábitos de sueño es tan relevante como reducir el "
                 "tiempo en redes.")
        st.pyplot(analizador.boxplot_por_grupo("sleep_hours", "depression_label"))
        st.caption("Evidencia: ítem 7.")

    with st.expander("4. La plataforma usada no marca diferencias"):
        st.write("El porcentaje con depression_label = 1 va de 2.3% a 3.0% entre Instagram, "
                 "TikTok y ambas, una diferencia mínima. **Decisión:** las acciones deben "
                 "centrarse en cuánto tiempo se usan las redes, no en cuál se usa.")
        st.pyplot(analizador.grafico_cruzado("platform_usage", "depression_label"))
        st.caption("Evidencia: ítem 8.")

    with st.expander("5. Lo que importa es la combinación de factores"):
        st.write("La etiqueta aparece solo en los 31 adolescentes que cumplen a la vez 4 "
                 "condiciones: más de 5 horas en redes, menos de 6 horas de sueño, estrés de "
                 "7 o más y ansiedad de 7 o más. Ninguno de los 198 que cumplen 3 condiciones "
                 "la tiene. **Decisión:** el seguimiento debe mirar perfiles completos y no "
                 "un solo indicador aislado.")
        st.pyplot(analizador.grafico_condiciones())
        st.caption("Evidencia: ítem 10.")

    st.info("⚠️ Conclusiones exploratorias y educativas. No constituyen un diagnóstico clínico "
            "ni sustituyen la valoración de profesionales de la salud.")
