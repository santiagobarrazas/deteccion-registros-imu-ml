"""Interfaz del prototipo: control de calidad automático de registros IMU con datos simulados."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from features import DESCRIPCION, NOMBRES, extraer, series  # noqa: E402
from modelo import crear_modelo, particionar  # noqa: E402
from simulador import CLASES  # noqa: E402

CORRUPTO = "ARCHIVO_CORRUPTO"
ETIQUETAS = {**CLASES, CORRUPTO: "Archivo ilegible o sin datos IMU"}
RECOMENDACION = {
    "OK": "Registro válido. Puede continuar con el siguiente paciente.",
    "PERDIDA_PAQUETES": "Acerque la tablet a los sensores y verifique la batería. Repita la prueba ahora.",
    "SENSOR_DESCONECTADO": "Un sensor dejó de transmitir. Revise batería y conexión Bluetooth y repita la prueba.",
    "SENSOR_SIN_MOVIMIENTO": "Un sensor casi no registra movimiento. Verifique que esté colocado y ajustado en el tobillo.",
    "SATURACION": "La señal excede el rango del sensor. Revise la fijación (golpes directos al sensor) y repita.",
    "DUPLICADO": "El archivo tiene muestras repetidas por una subida concurrente. Deduplicar antes del análisis.",
    CORRUPTO: "El archivo no se puede leer o no tiene muestras. Recupere la copia desde la tablet.",
}
COLOR = {"OK": "#1a7f37", CORRUPTO: "#8250df"}

st.set_page_config(page_title="Prototipo · Calidad IMU", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Entrenando el modelo con los datos simulados…")
def cargar_modelo():
    tabla = pd.read_csv(RAIZ / "data" / "indicadores_simulados.csv")
    X_tr, _, y_tr, _ = particionar(tabla[NOMBRES], tabla["clase"])
    modelo = crear_modelo().fit(X_tr, y_tr)
    metricas = json.loads((RAIZ / "models" / "metricas.json").read_text())
    return modelo, NOMBRES, metricas


def leer(contenido):
    """Devuelve el registro o None si el archivo está corrupto o no trae datos IMU."""
    try:
        registro = json.loads(contenido)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(registro, dict) or not registro.get("imuData"):
        return None
    return registro


def diagnosticar(registro, modelo, nombres):
    if registro is None:
        return CORRUPTO, {CORRUPTO: 1.0}, None
    feats = extraer(registro)
    probas = modelo.predict_proba(pd.DataFrame([feats], columns=nombres))[0]
    probabilidades = dict(zip(modelo.classes_, probas))
    return max(probabilidades, key=probabilidades.get), probabilidades, feats


def tarjeta(clase, confianza):
    color = COLOR.get(clase, "#cf222e")
    estado = "APTO para análisis" if clase == "OK" else "NO APTO: repetir o revisar"
    st.markdown(
        f"""<div style="border-left:8px solid {color};padding:14px 18px;border-radius:6px;
        background:rgba(127,127,127,.08)"><div style="font-size:.9rem;opacity:.75">{estado}</div>
        <div style="font-size:1.6rem;font-weight:700;color:{color}">{ETIQUETAS[clase]}</div>
        <div style="opacity:.8">Confianza del modelo: {confianza:.0%}</div></div>""",
        unsafe_allow_html=True,
    )
    st.write("")
    (st.success if clase == "OK" else st.warning)(RECOMENDACION[clase])


def grafica_senal(registro):
    fig = go.Figure()
    for dev, s in series(registro).items():
        fig.add_trace(go.Scattergl(x=s["t"], y=s["gyro"], mode="lines", name=dev, line={"width": 1}))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Tiempo (s)",
                      yaxis_title="|Giroscopio| (°/s)", legend=dict(orientation="h", y=1.12))
    return fig


def tabla_caracteristicas(feats, rango_ok):
    filas = []
    for nombre, valor in feats.items():
        bajo, alto = rango_ok[nombre]
        fuera = valor < bajo - 1e-9 or valor > alto + 1e-9
        filas.append({"Indicador": DESCRIPCION[nombre], "Valor": round(valor, 3),
                      "Rango normal": f"{bajo:g} – {alto:g}", "Estado": "⚠️ fuera" if fuera else "✓"})
    return pd.DataFrame(filas).sort_values("Estado", ascending=False)


def pestana_registro(modelo, nombres, metricas):
    ejemplos = sorted((RAIZ / "data" / "ejemplos").glob("*.json"))
    origen = st.radio("Origen del registro", ["Ejemplo simulado", "Subir archivo JSON de la tablet"], horizontal=True)
    if origen == "Ejemplo simulado":
        elegido = st.selectbox("Ejemplo", ejemplos, index=next(i for i, p in enumerate(ejemplos) if p.stem == "ejemplo_ok"),
                               format_func=lambda p: p.stem.replace("ejemplo_", "").upper())
        contenido = elegido.read_bytes()
    else:
        subido = st.file_uploader("Registro IMU (.json)", type="json")
        if not subido:
            st.info("Suba un registro con el formato de los ejemplos (bloque imuData). "
                    "El archivo se procesa solo en su navegador: no se envía a ningún servidor.")
            return
        contenido = subido.getvalue()

    registro = leer(contenido)
    clase, probabilidades, feats = diagnosticar(registro, modelo, nombres)
    izq, der = st.columns([1, 1.4])
    with izq:
        tarjeta(clase, probabilidades[clase])
        if feats:
            dist = pd.DataFrame({"Clase": [ETIQUETAS[c] for c in probabilidades], "Probabilidad": list(probabilidades.values())})
            fig = px.bar(dist.sort_values("Probabilidad"), x="Probabilidad", y="Clase", orientation="h", range_x=[0, 1])
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
    with der:
        if registro:
            st.plotly_chart(grafica_senal(registro), width="stretch")
    if feats:
        st.subheader("Por qué: indicadores del registro vs. rango de registros válidos")
        st.dataframe(tabla_caracteristicas(feats, metricas["rango_ok"]), hide_index=True, width="stretch")


def pestana_lote(modelo, nombres):
    st.write("Revise todos los registros de una jornada de captura de una vez.")
    archivos = st.file_uploader("Registros (.json)", type="json", accept_multiple_files=True)
    if not archivos:
        archivos = [p.open("rb") for p in sorted((RAIZ / "data" / "ejemplos").glob("*.json"))]
        st.caption("Mostrando los ejemplos simulados. Suba sus archivos para analizarlos.")
    filas = []
    for archivo in archivos:
        clase, probabilidades, _ = diagnosticar(leer(archivo.read()), modelo, nombres)
        filas.append({"Archivo": Path(archivo.name).name, "Diagnóstico": ETIQUETAS[clase],
                      "Apto": "✅" if clase == "OK" else "❌", "Confianza": f"{probabilidades[clase]:.0%}",
                      "Acción": RECOMENDACION[clase]})
    df = pd.DataFrame(filas)
    aptos = (df["Apto"] == "✅").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", len(df))
    c2.metric("Aptos", aptos)
    c3.metric("A repetir / revisar", len(df) - aptos)
    st.dataframe(df, hide_index=True, width="stretch")


def pestana_modelo(metricas):
    rep = metricas["reporte"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exactitud (prueba)", f"{rep['accuracy']:.1%}")
    c2.metric("F1 macro (CV 5-fold)", f"{metricas['f1_macro_cv'][0]:.3f}", f"± {metricas['f1_macro_cv'][1]:.3f}", delta_color="off", delta_arrow="off")
    c3.metric("Recall de defectos", f"{metricas['recall_defectos']:.1%}")
    c4.metric("Falsas alarmas", f"{metricas['falsas_alarmas']:.1%}")
    st.caption(f"Random Forest (200 árboles) con {len(metricas['importancia'])} indicadores de calidad. "
               f"Entrenamiento: {metricas['n_entrenamiento']} registros simulados; prueba: {metricas['n_prueba']}.")

    izq, der = st.columns(2)
    with izq:
        st.subheader("Matriz de confusión")
        etiquetas = [c.replace("_", " ").title() for c in metricas["clases"]]
        fig = px.imshow(np.array(metricas["matriz_confusion"]), x=etiquetas, y=etiquetas, text_auto=True,
                        color_continuous_scale="Blues", labels=dict(x="Predicho", y="Real", color="Registros"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with der:
        st.subheader("Importancia de los indicadores")
        imp = pd.DataFrame({"Indicador": [DESCRIPCION[k] for k in metricas["importancia"]],
                            "Importancia": list(metricas["importancia"].values())})
        fig = px.bar(imp.sort_values("Importancia"), x="Importancia", y="Indicador", orientation="h")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Desempeño por clase")
    st.dataframe(pd.DataFrame({ETIQUETAS[c]: rep[c] for c in metricas["clases"]}).T.round(3), width="stretch")


def pestana_impacto(metricas):
    st.write("Calculadora de impacto. Los valores por defecto son **supuestos editables**: reemplácelos por los de su operación.")
    izq, der = st.columns([1, 1.3])
    with izq:
        registros = st.number_input("Registros capturados por mes", 10, 100_000, 400, 10)
        tasa = st.slider("Registros defectuosos (%)", 0.0, 50.0, 15.0, 0.5) / 100
        min_revision = st.number_input("Minutos de revisión manual por registro", 0.0, 60.0, 4.0, 0.5)
        costo_hora = st.number_input("Costo hora del analista (COP)", 0, 1_000_000, 35_000, 1_000)
        costo_recaptura = st.number_input("Costo de recapturar un paciente después (COP)", 0, 5_000_000, 120_000, 5_000,
                                          help="Transporte, nueva cita y tiempo del fisioterapeuta")
        frac_recuperable = st.slider("Pacientes que sí se logran recapturar después (%)", 0, 100, 40) / 100
        min_repetir = st.number_input("Minutos para repetir la prueba en el momento", 0.0, 30.0, 3.0, 0.5)
    recall, falsas = metricas["recall_defectos"], metricas["falsas_alarmas"]

    defectuosos = registros * tasa
    detectados = defectuosos * recall
    falsas_alarmas = (registros - defectuosos) * falsas
    horas_ahorradas = registros * min_revision / 60
    horas_repetir = (detectados + falsas_alarmas) * min_repetir / 60
    ahorro = (horas_ahorradas - horas_repetir) * costo_hora + detectados * frac_recuperable * costo_recaptura
    util_antes = 1 - tasa
    util_despues = 1 - tasa * (1 - recall)

    with der:
        c1, c2 = st.columns(2)
        c1.metric("Registros útiles para investigación", f"{util_despues:.1%}", f"{(util_despues - util_antes) * 100:+.1f} pp")
        c2.metric("Defectos detectados en sitio / mes", f"{detectados:.0f}", f"de {defectuosos:.0f}", delta_color="off", delta_arrow="off")
        c3, c4 = st.columns(2)
        c3.metric("Horas de revisión liberadas / mes", f"{horas_ahorradas - horas_repetir:.1f} h")
        c4.metric("Ahorro estimado / mes", f"${ahorro:,.0f} COP")
        c5, c6 = st.columns(2)
        c5.metric("Tiempo hasta detectar un defecto", "< 1 s", "antes: semanas", delta_color="off", delta_arrow="off")
        c6.metric("Repeticiones innecesarias / mes", f"{falsas_alarmas:.1f}")
        st.caption(f"Usa el recall ({recall:.1%}) y la tasa de falsas alarmas ({falsas:.1%}) medidos en la prueba del modelo.")


def main():
    modelo, nombres, metricas = cargar_modelo()
    st.title("Control de calidad automático de registros IMU")
    st.caption("Prototipo académico · Escenario: laboratorio de análisis de movimiento · Prueba TAPPING_TOE · Datos 100 % simulados")
    with st.sidebar:
        st.markdown("### Cómo funciona")
        st.markdown("1. **Simulador:** genera registros IMU con y sin fallas.")
        st.markdown("2. **Indicadores:** 13 medidas de calidad de la señal.")
        st.markdown("3. **Random Forest:** clasifica el registro en 6 clases.")
        st.caption("Todo corre en su navegador. No hay datos personales: los registros son sintéticos.")
    tabs = st.tabs(["🔎 Analizar registro", "📂 Jornada (lote)", "📊 Desempeño del modelo", "💰 Impacto"])
    with tabs[0]:
        pestana_registro(modelo, nombres, metricas)
    with tabs[1]:
        pestana_lote(modelo, nombres)
    with tabs[2]:
        pestana_modelo(metricas)
    with tabs[3]:
        pestana_impacto(metricas)


main()
