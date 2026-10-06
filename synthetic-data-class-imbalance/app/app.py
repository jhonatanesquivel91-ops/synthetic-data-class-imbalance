"""Aplicación de demostración: evaluación de datos sintéticos para clasificación binaria desbalanceada.

Ejecutar desde la raíz del repositorio:   streamlit run app/app.py
Para usarla sin conexión, ejecutar antes (con Internet):   python app/prepare_offline.py
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAVY, YEL, TEAL, RED, GREY, STEEL = "#003264", "#F3C133", "#2A9D8F", "#C0392B", "#8A97A8", "#8FA6C4"
COLORS = {"REAL": "#555555", "REAL-CW": "#7B5EA7", "SMOTE-NC": NAVY, "CTGAN": YEL, "TVAE": STEEL, "HYBRID": TEAL}
DS = {"DS1": ("dataset_01_cdc_diabetes", "Diabetes Health Indicators", "Salud", 6.18),
      "DS2": ("dataset_02_thyroid_cancer", "Thyroid Cancer Recurrence", "Salud", 2.55),
      "DS3": ("dataset_03_aids_trial_175", "AIDS Clinical Trial (175)", "Salud", 3.10),
      "DS4": ("dataset_04_credit_default", "Credit Card Default", "Finanzas", 3.52),
      "DS5": ("dataset_05_taiwanese_bankruptcy", "Taiwanese Bankruptcy", "Finanzas", 29.99)}
NAME = {"REAL": "REAL", "REAL_CW": "REAL-CW", "SMOTE": "SMOTE-NC", "CTGAN": "CTGAN", "TVAE": "TVAE", "HYBRID": "HYBRID"}
EXAMPLES = {"Thyroid Cancer Recurrence (UCI 915, 383 registros)": (915, "Recurred"),
            "AIDS Clinical Trial 175 (UCI 890, 2 139 registros)": (890, "cid")}

st.set_page_config(page_title="Evaluador de datos sintéticos", layout="wide")
st.markdown(f"""<style>
h1, h2, h3 {{ color: {NAVY}; }}
div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
.block-container {{ padding-top: 2rem; }}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ datos de la tesis
@st.cache_data
def thesis_data():
    U, F = [], []
    for k, (folder, *_ ) in DS.items():
        p = os.path.join(ROOT, folder, "outputs", "metrics_v2")
        U.append(pd.read_csv(os.path.join(p, "utility_runs.csv")).assign(DS=k))
        F.append(pd.read_csv(os.path.join(p, "fidelity_runs.csv")).assign(DS=k))
    U = pd.concat(U); F = pd.concat(F)
    U["Método"] = U.train_set.map(NAME); F["Método"] = F.method.map(NAME)
    ST = pd.read_csv(os.path.join(ROOT, "results", "stats_full.csv"))
    names = {"cdc_diabetes": "DS1", "thyroid_cancer_recurrence": "DS2", "aids_clinical_trial_175": "DS3",
             "credit_card_default": "DS4", "taiwanese_bankruptcy_prediction": "DS5"}
    ST["DS"] = ST.dataset.map(names); ST["Método"] = ST.method.map(NAME)
    return U, F, ST


def page_results():
    st.title("Resultados de la tesis")
    st.caption("10 corridas por dataset que regeneran los datos sintéticos · pruebas pareadas con corrección de Holm · "
               "Esquivel et al., IJACSA 17(9), 2026 · DOI 10.14569/IJACSA.2026.0170971")
    U, F, ST = thesis_data()
    c1, c2, c3 = st.columns([1.4, 1, 1])
    ds = c1.selectbox("Dataset", list(DS), format_func=lambda k: f"{k} · {DS[k][1]} ({DS[k][3]}:1)")
    clf = c2.selectbox("Clasificador", ["CatBoost", "XGBoost"])
    met = c3.selectbox("Métrica", E.METRICS, format_func=lambda m: E.LABELS[m])

    sub = U[(U.DS == ds) & (U.model == clf)]
    agg = sub.groupby("Método")[E.METRICS].agg(["mean", "std"])
    order = [m for m in ["REAL", "REAL-CW", "SMOTE-NC", "CTGAN", "TVAE", "HYBRID"] if m in agg.index]
    stt = ST[(ST.DS == ds) & (ST.model == clf) & (ST.metric == met)].set_index("Método")

    k1, k2, k3, k4 = st.columns(4)
    real = agg.loc["REAL", (met, "mean")]
    syn = [m for m in order if m not in ("REAL", "REAL-CW")]
    best = max(syn, key=lambda m: agg.loc[m, (met, "mean")])
    k1.metric("Ratio de desbalance", f"{DS[ds][3]}:1")
    k2.metric(f"{E.LABELS[met]} con REAL", f"{real:.3f}")
    k3.metric("Mejor método sintético", best, f"{agg.loc[best, (met, 'mean')] - real:+.3f}")
    sy = stt[stt.index.isin(syn)]
    k4.metric("Métodos sintéticos que mejoran", f"{int((sy.res == 'UP').sum())} de {len(sy)}",
              f"{int((sy.res == 'DOWN').sum())} empeoran", delta_color="off")

    marks = []
    for m in order:
        r = stt.res.get(m) if m != "REAL" else None
        marks.append({"UP": "▲", "DOWN": "▼"}.get(r, ""))
    means = [agg.loc[m, (met, "mean")] for m in order]; sds = [agg.loc[m, (met, "std")] for m in order]
    fig = go.Figure(go.Bar(x=order, y=means, error_y=dict(type="data", array=sds, color="#444"),
                           marker_color=[COLORS[m] for m in order], hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    for m, mu, sd, mk in zip(order, means, sds, marks):
        fig.add_annotation(x=m, y=mu + sd, text=f"<b>{mu:.3f}</b> {mk}", showarrow=False, yshift=14,
                           font=dict(size=14, color={"▲": TEAL, "▼": RED}.get(mk, "#333")))
    fig.update_layout(height=420, margin=dict(t=30, b=10), yaxis_title=E.LABELS[met], plot_bgcolor="white",
                      yaxis=dict(gridcolor="#E3E8EF", range=[0, min(1.08, agg[(met, "mean")].max() * 1.25 + 0.05)]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("▲ / ▼: diferencia significativa frente a REAL (p < 0.05; Holm entre métodos sintéticos, REAL-CW sin corrección). "
               "Las barras de error son la desviación estándar entre corridas.")

    t1, t2 = st.tabs(["Todas las métricas", "Fidelidad estadística"])
    with t1:
        tab = pd.DataFrame({E.LABELS[m]: [f"{agg.loc[o, (m, 'mean')]:.4f} ± {agg.loc[o, (m, 'std')]:.4f}" for o in order]
                            for m in E.METRICS}, index=order)
        st.dataframe(tab, use_container_width=True)
    with t2:
        f = F[F.DS == ds].groupby("Método")[["wasserstein_std", "corr_diff_mean", "cat_diff_mean", "diversity"]].mean()
        f.columns = ["Wasserstein (↓)", "Δ Correlación (↓)", "Δ Categórica (↓)", "Diversidad"]
        st.dataframe(f.round(4), use_container_width=True)
        st.caption(f"Diversidad de los registros minoritarios reales: {F[F.DS == ds].real_diversity_ref.iloc[0]:.2f}. "
                   "Valores muy superiores indican registros dispersos fuera de la clase real.")


# ------------------------------------------------------------------ evaluación de un dataset
@st.cache_data(show_spinner="Descargando el dataset de la UCI…")
def load_uci(uid):
    cache = os.path.join(os.path.dirname(__file__), "data", f"uci_{uid}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=uid)
    df = pd.concat([d.data.features, d.data.targets], axis=1)
    os.makedirs(os.path.dirname(cache), exist_ok=True); df.to_csv(cache, index=False)
    return df


def op_chart(res):
    agg = res.groupby("condition")[["Recall", "Precision"]].mean()
    fig = go.Figure()
    r = np.linspace(0.01, 1, 200)
    for f1 in [0.2, 0.4, 0.6, 0.8]:
        p = f1 * r / (2 * r - f1); ok = (p > 0) & (p <= 1)
        fig.add_trace(go.Scatter(x=r[ok], y=p[ok], mode="lines", line=dict(color="#C9D1DC", dash="dot", width=1),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=1.0, y=f1 / (2 - f1), text=f"F1 = {f1}", showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=11, color=GREY))
    for c in agg.index:
        fig.add_trace(go.Scatter(x=[agg.loc[c, "Recall"]], y=[agg.loc[c, "Precision"]], mode="markers+text", name=c,
                                 text=[c], textposition="top center", marker=dict(size=14, color=COLORS.get(c, NAVY),
                                 line=dict(color="white", width=1))))
    fig.update_layout(height=420, xaxis=dict(title="Recall", range=[0, 1.18], gridcolor="#E3E8EF"),
                      yaxis=dict(title="Precisión", range=[0, 1.05], gridcolor="#E3E8EF"),
                      plot_bgcolor="white", margin=dict(t=20, b=10), showlegend=False)
    return fig


def page_evaluate():
    st.title("Evaluar un dataset")
    st.write("Compara REAL, pesos de clase (REAL-CW) y generadores sintéticos sobre tus datos, con el mismo protocolo de la "
             "tesis en versión rápida: partición fija 80/20, varias corridas que regeneran los datos y pruebas pareadas.")
    src = st.radio("Origen de los datos", ["Dataset de ejemplo (UCI)", "Subir un CSV"], horizontal=True)
    df = None
    if src.startswith("Dataset"):
        ex = st.selectbox("Ejemplo", list(EXAMPLES))
        try:
            df = load_uci(EXAMPLES[ex][0]); default_target = EXAMPLES[ex][1]
        except Exception:
            st.error("No se encontró el dataset guardado ni hay conexión con la UCI. Ejecuta antes, con Internet, `python app/prepare_offline.py`, o sube un CSV.")
            return
    else:
        up = st.file_uploader("Archivo CSV con una columna objetivo binaria", type="csv")
        if up is None:
            return
        df = pd.read_csv(up); default_target = df.columns[-1]
    st.dataframe(df.head(), use_container_width=True)

    c1, c2 = st.columns([1, 2])
    target = c1.selectbox("Variable objetivo", list(df.columns), index=list(df.columns).index(default_target))
    cats = c2.multiselect("Variables categóricas", [c for c in df.columns if c != target],
                          default=E.guess_categorical(df, target))
    try:
        P = E.Prepared(df, target, cats)
    except ValueError as e:
        st.error(str(e)); return
    info = P.info()
    k = st.columns(4)
    k[0].metric("Registros", f"{info['registros']:,}")
    k[1].metric("Ratio de desbalance", f"{info['ratio']}:1")
    k[2].metric("Minoría para entrenar", info["minoría en entrenamiento"])
    k[3].metric("Positivos en prueba", info["positivos en prueba"])

    c1, c2, c3 = st.columns([2, 1, 1])
    methods = c1.multiselect("Métodos sintéticos", E.SYNTHETIC, default=["SMOTE-NC"],
                             help="CTGAN y TVAE entrenan redes neuronales: tardan más. Para la demo en vivo usa SMOTE-NC.")
    n_runs = c2.slider("Corridas", 3, 10, 5)
    epochs = c3.slider("Épocas (CTGAN/TVAE)", 20, 300, 100, step=20, disabled=not any(m in methods for m in ["CTGAN", "TVAE"]))

    if st.button("Ejecutar evaluación", type="primary"):
        bar = st.progress(0.0, text="Iniciando…")
        res = E.run(P, methods, n_runs=n_runs, epochs=epochs, progress=lambda f, t: bar.progress(f, text=t))
        bar.empty()
        st.session_state["res"] = res
    if "res" not in st.session_state:
        return
    res = st.session_state["res"]; cmp = E.compare(res)

    st.subheader("Lectura de resultados")
    for line in E.recommendation(cmp):
        st.markdown(f"- {line}")

    if True:
        st.markdown("**Métricas (media ± DE) y comparación con REAL**")
        conds = ["REAL"] + [c for c in res.condition.unique() if c != "REAL"]
        agg = res.groupby("condition")[E.METRICS].agg(["mean", "std"])
        rows = []
        for c in conds:
            row = {"Condición": c}
            for m in ["AUC_PR", "Recall", "Precision", "F1", "GMean"]:
                v = f"{agg.loc[c, (m, 'mean')]:.3f} ± {agg.loc[c, (m, 'std')]:.3f}"
                if c != "REAL":
                    r = cmp[(cmp.metric == m) & (cmp.condition == c)].iloc[0]
                    v += {"sube": " ▲", "baja": " ▼"}.get(r.result, "")
                row[E.LABELS[m]] = v
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Condición"), use_container_width=True)
        st.caption("▲ / ▼: diferencia significativa frente a REAL (p < 0.05).")
    c1, _ = st.columns([2, 1])
    with c1:
        st.markdown("**Puntos de operación (umbral 0.5) y curvas iso-F1**")
        st.plotly_chart(op_chart(res), use_container_width=True)

    with st.expander("Detalle de las pruebas estadísticas"):
        d = cmp.copy(); d["metric"] = d.metric.map(E.LABELS)
        st.dataframe(d.round(4), use_container_width=True)
    st.download_button("Descargar resultados por corrida (CSV)", res.to_csv(index=False).encode(), "resultados_corridas.csv")


st.sidebar.markdown(f"<h2 style='color:{NAVY};margin-bottom:0'>Datos sintéticos y desbalance de clases</h2>",
                    unsafe_allow_html=True)
st.sidebar.caption("Universidad Peruana Unión · Facultad de Ingeniería y Arquitectura")
page = st.sidebar.radio("Sección", ["Resultados de la tesis", "Evaluar un dataset"])
st.sidebar.markdown("---")
st.sidebar.caption("Esquivel, Humpiri, Vega y Saboya (2026). Comparative evaluation of synthetic data generation methods "
                   "for binary classification problems with class imbalance. IJACSA, 17(9).")
page_results() if page.startswith("Resultados") else page_evaluate()
