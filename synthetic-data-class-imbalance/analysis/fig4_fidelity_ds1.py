#!/usr/bin/env python3
# =========================================================
# Figura 4 · Fidelidad de distribución de SMOTE-NC en DS1
# Compara la minoría real del entrenamiento interno con los registros sintéticos
# de SMOTE-NC (semilla 42) en las cuatro variables más importantes.
# Se ejecuta dentro de la carpeta del dataset, después de los notebooks 01, 03 y 04:
#     cd dataset_01_cdc_diabetes
#     python ../analysis/fig4_fidelity_ds1.py
# Salidas: outputs/plots/distribution_analysis_ds1_v2.png y outputs/plots/fig4_values.csv
# =========================================================
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter
from scipy.stats import wasserstein_distance
from imblearn.over_sampling import SMOTE, SMOTENC
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT)
from utils.config_loader import load_config

config = load_config("dataset_config.yaml")
DS_ID = config.get("dataset", "X")
RATIO = config.get("ratio", "0.00")
SEED = config.get("random_state", 42)
num_cols = config.get("numerical_features", [])
cat_cols = config.get("categorical_features", [])
OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Datos: mismo entrenamiento interno que 10_multirun_fair.py ----------------
X_tr_proc = pd.read_csv("outputs/splits/X_train_in.csv")
X_tr_raw = pd.read_csv("outputs/splits/X_train_raw_in.csv")[X_tr_proc.columns]
y_tr = pd.read_csv("outputs/splits/y_train_in.csv")["y"].values
scaler = joblib.load("outputs/data/scaler.pkl")
assert len(X_tr_proc) == len(X_tr_raw) == len(y_tr), "Train interno desalineado"

cnt = Counter(y_tr)
minority, majority = min(cnt, key=cnt.get), max(cnt, key=cnt.get)
M = cnt[majority]
real_min = X_tr_raw[y_tr == minority].reset_index(drop=True)

# ---------------- SMOTE-NC: misma k y semilla 42 que 10_multirun_fair.py ----------------
k = json.load(open("outputs/smote/best_config.json"))["params"]["k_neighbors"]
cat_idx = [X_tr_proc.columns.get_loc(c) for c in cat_cols]
sm = (SMOTENC(categorical_features=cat_idx, k_neighbors=k, sampling_strategy={minority: M}, random_state=SEED)
      if cat_idx else SMOTE(k_neighbors=k, sampling_strategy={minority: M}, random_state=SEED))
X_res, _ = sm.fit_resample(X_tr_proc, y_tr)
syn = pd.DataFrame(X_res, columns=X_tr_proc.columns).iloc[len(X_tr_proc):].reset_index(drop=True)
if num_cols:
    syn[num_cols] = scaler.inverse_transform(syn[num_cols].astype(float))
print(f"Minoría real: {len(real_min)} | sintéticos SMOTE-NC: {len(syn)} | k={k}")

# ---------------- Importancia de variables: solo con el entrenamiento interno real ----------------
model = XGBClassifier(n_estimators=100, tree_method="hist", random_state=SEED)
model.fit(X_tr_proc, y_tr)
feature_df = pd.DataFrame({"feature": X_tr_proc.columns, "importance": model.feature_importances_})
selected = feature_df.sort_values("importance", ascending=False)["feature"].head(4).tolist()

# ---------------- Estilo de las figuras del artículo ----------------
sns.set_theme(style="whitegrid", font="serif")
fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=300)
axes = axes.flatten()
rows = []
for ax, col in zip(axes, selected):
    r = real_min[col].dropna().to_numpy().astype(float)
    s = syn[col].dropna().to_numpy().astype(float)
    wd = wasserstein_distance(r, s)
    imp = feature_df.loc[feature_df.feature == col, "importance"].values[0]
    if len(np.unique(r)) <= 15:
        bins = np.arange(min(r.min(), s.min()) - 0.5, max(r.max(), s.max()) + 1.5, 1)
        ax.hist(r, bins=bins, density=True, alpha=0.45, label="Real (minority)", edgecolor="black", linewidth=0.6)
        ax.hist(s, bins=bins, density=True, alpha=0.45, label="SMOTE-NC (synthetic)", edgecolor="black", linewidth=0.6)
        for v in np.unique(np.concatenate([r, s])):
            rows.append({"feature": col, "value": v, "real_pct": round(100 * np.mean(r == v), 2),
                         "synthetic_pct": round(100 * np.mean(s == v), 2), "WD": round(wd, 4), "importance": round(imp, 3)})
    else:
        sns.kdeplot(r, ax=ax, lw=2.2, fill=True, alpha=0.15, label="Real (minority)")
        sns.kdeplot(s, ax=ax, lw=2.4, linestyle="--", label="SMOTE-NC (synthetic)")
        rows.append({"feature": col, "value": "mean", "real_pct": round(r.mean(), 3),
                     "synthetic_pct": round(s.mean(), 3), "WD": round(wd, 4), "importance": round(imp, 3)})
    ax.set_title(f"{col}\n(Imp: {imp:.3f} | WD: {wd:.4f})", fontsize=12, weight="bold", pad=15)
    ax.set_xlabel("Value"); ax.set_ylabel("Density")
    ax.legend(fontsize=9, frameon=True, facecolor="white", framealpha=0.9, edgecolor="gray")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.suptitle("Distribution Fidelity on Most Predictive Features\n"
             "Real minority records vs. SMOTE-NC synthetic records (seed 42)\n"
             f"Dataset {DS_ID} | Imbalance Ratio {RATIO}:1",
             fontsize=17, weight="bold", y=0.97)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{OUTPUT_DIR}/distribution_analysis_ds{DS_ID}_v2.png"
plt.savefig(out, dpi=500, bbox_inches="tight")
pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/fig4_values.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n✔ Guardado: {out}  y  {OUTPUT_DIR}/fig4_values.csv")
