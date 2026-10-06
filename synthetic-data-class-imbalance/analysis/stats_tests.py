"""Validación estadística de los resultados (Sección III.F y Tabla IV del artículo).

Lee dataset_0*/outputs/metrics_v2/utility_runs.csv (10 corridas por dataset) y compara,
para cada dataset, clasificador y métrica, cada condición de entrenamiento contra REAL:
  - prueba t pareada y, si Shapiro-Wilk rechaza la normalidad (p<0.05), Wilcoxon de rangos con signo;
  - corrección de Holm dentro de cada familia de 4 métodos sintéticos;
  - REAL_CW se contrasta por separado, sin corrección;
  - tamaño del efecto d_z e intervalo de confianza al 95 % de la diferencia media.
Uso (desde la raíz del repositorio):  python analysis/stats_tests.py
Salidas: results/stats_full.csv y results/summary_full.csv
"""
import glob, os
import numpy as np
import pandas as pd
from scipy import stats

SYN = ["SMOTE", "CTGAN", "TVAE", "HYBRID"]
METRICS = ["AUC_PR", "Recall", "Precision", "F1", "BalancedAcc", "Specificity", "GMean"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (n - k) * p[i])); adj[i] = run
    return adj


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "dataset_0*", "outputs", "metrics_v2", "utility_runs.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out_dir = os.path.join(ROOT, "results"); os.makedirs(out_dir, exist_ok=True)

    summary = df.groupby(["dataset", "model", "train_set"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(os.path.join(out_dir, "summary_full.csv"), index=False)

    rows = []
    for (ds, clf), g in df.groupby(["dataset", "model"]):
        base = g[g.train_set == "REAL"].set_index("seed").sort_index()
        for met in METRICS:
            fam = []
            for meth in SYN + ["REAL_CW"]:
                s = g[g.train_set == meth].set_index("seed").sort_index()
                d = (s[met] - base.loc[s.index, met]).values
                n = len(d); sd = d.std(ddof=1)
                if sd == 0:
                    t_p = dz = sw_p = w_p = np.nan; ci = (d.mean(), d.mean())
                else:
                    t_p = stats.ttest_rel(s[met], base.loc[s.index, met]).pvalue
                    dz = d.mean() / sd
                    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
                    ci = (d.mean() - h, d.mean() + h)
                    sw_p = stats.shapiro(d).pvalue if n >= 3 else np.nan
                    w_p = stats.wilcoxon(d).pvalue if n >= 6 else np.nan
                fam.append(dict(dataset=ds, model=clf, metric=met, method=meth, n=n,
                                real=base[met].mean(), synthetic=s[met].mean(), delta=d.mean(),
                                ci_low=ci[0], ci_high=ci[1], p_t=t_p, d_z=dz, shapiro_p=sw_p, p_wilcoxon=w_p))
            syn = [r for r in fam if r["method"] in SYN]
            for key in ["p_t", "p_wilcoxon"]:
                ps = [r[key] for r in syn]
                ok = [i for i, v in enumerate(ps) if not np.isnan(v)]
                adj = holm([ps[i] for i in ok]) if ok else []
                for j, i in enumerate(ok):
                    syn[i][key + "_holm"] = adj[j]
            rows += fam

    out = pd.DataFrame(rows)
    nonnormal = out.shapiro_p < 0.05
    corrected = np.where(nonnormal, out.get("p_wilcoxon_holm"), out.get("p_t_holm"))
    raw = np.where(nonnormal, out.p_wilcoxon, out.p_t)
    out["p_final"] = np.where(out.method == "REAL_CW", raw, corrected)
    out["res"] = np.where(out.p_final.isna(), "n/a",
                 np.where(out.p_final < 0.05, np.where(out.delta > 0, "UP", "DOWN"), "ns"))
    out.to_csv(os.path.join(out_dir, "stats_full.csv"), index=False)

    syn_only = out[out.method.isin(SYN)]
    print("Resumen de 40 comparaciones por métrica (métodos sintéticos vs REAL):")
    print(syn_only.groupby("metric").res.value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
