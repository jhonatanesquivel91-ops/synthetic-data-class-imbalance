"""Motor de evaluación rápida: misma lógica que 10_multirun_fair.py, simplificada para la demo.

Partición 80/20 estratificada fija (semilla 42); en cada corrida se regeneran los datos sintéticos
y se reentrena el clasificador con la semilla de la corrida. Las diferencias frente a REAL se contrastan
con pruebas pareadas (t o Wilcoxon según Shapiro-Wilk) y corrección de Holm entre métodos sintéticos.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import (average_precision_score, precision_score, f1_score,
                             balanced_accuracy_score, confusion_matrix)
from imblearn.over_sampling import SMOTE, SMOTENC
from xgboost import XGBClassifier

SYNTHETIC = ["SMOTE-NC", "CTGAN", "TVAE"]
METRICS = ["AUC_PR", "Recall", "Precision", "F1", "Specificity", "GMean", "BalancedAcc"]
LABELS = {"AUC_PR": "AUC-PR", "Recall": "Recall", "Precision": "Precisión", "F1": "F1-Score",
          "Specificity": "Especificidad", "GMean": "G-mean", "BalancedAcc": "Balanced Accuracy"}


def guess_categorical(df, target, max_levels=10):
    cats = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if s.dtype == object or str(s.dtype) == "category" or s.dtype == bool or s.nunique() <= max_levels:
            cats.append(c)
    return cats


class Prepared:
    """Partición fija y transformaciones ajustadas solo con entrenamiento (sin fuga)."""

    def __init__(self, df, target, cat_cols, test_size=0.2, seed=42):
        df = df.dropna(subset=[target]).copy()
        y_raw = df[target].astype(str)
        counts = y_raw.value_counts()
        if len(counts) != 2:
            raise ValueError(f"La variable objetivo debe tener 2 clases; tiene {len(counts)}.")
        self.pos_label = counts.idxmin()            # la clase minoritaria es la positiva
        y = (y_raw == self.pos_label).astype(int).values
        X = df.drop(columns=[target])
        self.cat_cols = [c for c in cat_cols if c in X.columns]
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]
        X[self.cat_cols] = X[self.cat_cols].astype(str)
        for c in self.num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        Xtr, Xte, self.y_tr, self.y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
        # Imputación simple ajustada en entrenamiento
        self.med = Xtr[self.num_cols].median()
        Xtr = Xtr.fillna(self.med); Xte = Xte.fillna(self.med)
        self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.sc = StandardScaler()
        self.X_tr_raw = Xtr.copy()
        if self.cat_cols:
            self.X_tr_raw[self.cat_cols] = self.enc.fit_transform(Xtr[self.cat_cols]).astype(int)
        self.X_tr = self.to_scaled(self.X_tr_raw)
        Xte_e = Xte.copy()
        if self.cat_cols:
            Xte_e[self.cat_cols] = self.enc.transform(Xte[self.cat_cols]).astype(int)
        self.X_te = self.to_scaled(Xte_e, fit=False)
        self.m = int(self.y_tr.sum()); self.M = int(len(self.y_tr) - self.m)
        self.n_gen = self.M - self.m

    def to_scaled(self, raw, fit=None):
        out = raw.copy()
        if self.num_cols:
            if fit is None and not hasattr(self.sc, "mean_"):
                out[self.num_cols] = self.sc.fit_transform(raw[self.num_cols].astype(float))
            else:
                out[self.num_cols] = self.sc.transform(raw[self.num_cols].astype(float))
        return out

    def to_raw(self, scaled):
        out = scaled.copy()
        if self.num_cols:
            out[self.num_cols] = self.sc.inverse_transform(scaled[self.num_cols].astype(float))
        return out

    def info(self):
        return {"registros": len(self.y_tr) + len(self.y_te), "clase positiva (minoría)": self.pos_label,
                "ratio": round(self.M / max(self.m, 1), 2), "minoría en entrenamiento": self.m,
                "positivos en prueba": int(self.y_te.sum()), "numéricas": len(self.num_cols),
                "categóricas": len(self.cat_cols)}


def generate(P, method, seed, epochs=100):
    """Devuelve SOLO los registros sintéticos nuevos, en el espacio escalado."""
    if method == "SMOTE-NC":
        k = max(1, min(5, P.m - 1))
        idx = [P.X_tr.columns.get_loc(c) for c in P.cat_cols]
        sm = (SMOTENC(categorical_features=idx, k_neighbors=k, random_state=seed) if idx
              else SMOTE(k_neighbors=k, random_state=seed))
        Xr, _ = sm.fit_resample(P.X_tr, P.y_tr)
        return pd.DataFrame(Xr, columns=P.X_tr.columns).iloc[len(P.X_tr):].reset_index(drop=True)
    import torch
    from ctgan import CTGAN, TVAE
    torch.manual_seed(seed); np.random.seed(seed)
    minority = P.X_tr_raw[P.y_tr == 1].reset_index(drop=True)
    bs = max(10, min(500, (len(minority) // 10) * 10))
    model = (CTGAN(epochs=epochs, batch_size=bs, pac=10, verbose=False) if method == "CTGAN"
             else TVAE(epochs=epochs, batch_size=bs))
    model.fit(minority, P.cat_cols)
    syn = model.sample(P.n_gen)[P.X_tr.columns]
    return P.to_scaled(syn, fit=False)


def evaluate(P, Xt, yt, seed, weighted=False):
    w = (P.M / max(P.m, 1)) if weighted else 1.0
    clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=-1, scale_pos_weight=w)
    clf.fit(Xt, yt)
    score = clf.predict_proba(P.X_te)[:, 1]
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(P.y_te, pred, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"AUC_PR": average_precision_score(P.y_te, score), "Recall": rec,
            "Precision": precision_score(P.y_te, pred, zero_division=0), "F1": f1_score(P.y_te, pred, zero_division=0),
            "Specificity": spec, "GMean": float(np.sqrt(rec * spec)),
            "BalancedAcc": balanced_accuracy_score(P.y_te, pred), "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def run(P, methods, n_runs=5, epochs=100, progress=None):
    rows = []
    total = n_runs * (2 + len(methods)); done = 0
    for r in range(n_runs):
        seed = 42 + r
        conds = [("REAL", P.X_tr, P.y_tr, False), ("REAL-CW", P.X_tr, P.y_tr, True)]
        for meth in methods:
            syn = generate(P, meth, seed, epochs)
            conds.append((meth, pd.concat([P.X_tr, syn], ignore_index=True),
                          np.concatenate([P.y_tr, np.ones(len(syn), dtype=int)]), False))
        for name, Xt, yt, wtd in conds:
            rows.append({"run": r + 1, "seed": seed, "condition": name, **evaluate(P, Xt, yt, seed, wtd)})
            done += 1
            if progress:
                progress(done / total, f"Corrida {r + 1}/{n_runs} · {name}")
    return pd.DataFrame(rows)


def _holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (n - k) * p[i])); adj[i] = run
    return adj


def compare(res):
    """Pruebas pareadas de cada condición contra REAL, por métrica."""
    base = res[res.condition == "REAL"].set_index("run").sort_index()
    out = []
    for met in METRICS:
        fam = []
        for cond in [c for c in res.condition.unique() if c != "REAL"]:
            s = res[res.condition == cond].set_index("run").sort_index()
            d = (s[met] - base.loc[s.index, met]).values
            p = np.nan; test = "—"
            if len(d) >= 3 and d.std(ddof=1) > 0:
                normal = stats.shapiro(d).pvalue >= 0.05
                if not normal and len(d) >= 6:
                    p = stats.wilcoxon(d).pvalue; test = "Wilcoxon"
                else:
                    p = stats.ttest_rel(s[met], base.loc[s.index, met]).pvalue; test = "t pareada"
            fam.append({"metric": met, "condition": cond, "mean": s[met].mean(), "sd": s[met].std(ddof=1),
                        "real": base[met].mean(), "delta": d.mean(), "p": p, "test": test})
        syn = [f for f in fam if f["condition"] in SYNTHETIC and not np.isnan(f["p"])]
        if syn:
            for f, a in zip(syn, _holm([f["p"] for f in syn])):
                f["p"] = a
        out += fam
    df = pd.DataFrame(out)
    df["result"] = np.where(df.p.isna(), "sin variación",
                   np.where(df.p < 0.05, np.where(df.delta > 0, "sube", "baja"), "sin diferencia"))
    return df


def recommendation(cmp):
    """Lectura automática en lenguaje simple, en la línea de las conclusiones de la tesis."""
    lines = []
    rec = cmp[cmp.metric == "Recall"]; auc = cmp[cmp.metric == "AUC_PR"]; pre = cmp[cmp.metric == "Precision"]
    up = rec[(rec.result == "sube") & rec.condition.isin(SYNTHETIC)]
    if len(up):
        best = up.sort_values("mean", ascending=False).iloc[0]
        cost = pre[pre.condition == best.condition].iloc[0]
        lines.append(f"La síntesis aumenta el Recall de forma significativa: {best.condition} lo lleva de "
                     f"{best.real:.3f} a {best['mean']:.3f}, con Precisión de {cost.real:.3f} a {cost['mean']:.3f}.")
    else:
        lines.append("Ningún método sintético aumenta el Recall de forma significativa en este dataset.")
    cw = rec[rec.condition == "REAL-CW"]
    if len(cw):
        cw = cw.iloc[0]
        best_syn = rec[rec.condition.isin(SYNTHETIC)]["mean"].max() if rec.condition.isin(SYNTHETIC).any() else np.nan
        if not np.isnan(best_syn) and cw["mean"] >= best_syn - 0.02:
            lines.append(f"Los pesos de clase (REAL-CW) logran un Recall similar o mayor ({cw['mean']:.3f}) sin generar "
                         "registros: conviene probarlos primero.")
        else:
            lines.append(f"Los pesos de clase (REAL-CW) alcanzan un Recall de {cw['mean']:.3f}.")
    a_up = auc[(auc.result == "sube") & auc.condition.isin(SYNTHETIC)]
    a_dn = auc[(auc.result == "baja") & auc.condition.isin(SYNTHETIC)]
    if len(a_up):
        lines.append("El AUC-PR mejora significativamente con: " + ", ".join(a_up.condition) + ".")
    elif len(a_dn):
        lines.append("El AUC-PR no mejora y baja con: " + ", ".join(a_dn.condition) +
                     ". La síntesis desplaza la frontera de decisión más de lo que mejora la separación de clases.")
    else:
        lines.append("El AUC-PR no cambia de forma significativa: la síntesis no mejora el ordenamiento global.")
    return lines
