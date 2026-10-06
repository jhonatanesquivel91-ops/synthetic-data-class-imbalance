#!/usr/bin/env python3
# ==========================================================
# 10 · Protocolo final de evaluación (10 corridas)
# ----------------------------------------------------------
# Se ejecuta dentro de la carpeta de cada dataset (run_all.sh lo hace para los cinco).
# En cada corrida (semillas 42 a 51):
#   1. Regenera los datos sintéticos con SMOTE-NC, CTGAN, TVAE y el híbrido SMOTE-NC -> CTGAN,
#      usando los hiperparámetros seleccionados en los notebooks 04 a 07.
#   2. Mide su fidelidad frente a la minoría real (Wasserstein, correlación, categórica, diversidad).
#   3. Entrena XGBoost y CatBoost con configuración fija sobre: REAL, REAL_CW (pesos de clase)
#      y REAL + sintéticos de cada método, y los evalúa en la partición de prueba.
# Todas las condiciones parten del mismo entrenamiento interno real. Los resultados se guardan
# al terminar cada semilla en outputs/metrics_v2/, por lo que la ejecución se puede reanudar.
# ==========================================================

import sys, os, json, math, time, random, warnings
import numpy as np
import pandas as pd
import torch
import joblib
from collections import Counter

from sklearn.metrics import (precision_score, recall_score, f1_score,
                             balanced_accuracy_score, average_precision_score,
                             confusion_matrix)
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, SMOTENC
from ctgan import CTGAN
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT)
from utils.config_loader import load_config

config = load_config("dataset_config.yaml")
DATASET = config["dataset_name"]
RANDOM_STATE = config.get("random_state", 42)

# ---------------------------------------------------------
# Número de corridas (por defecto 10; se puede cambiar con la variable de entorno N_RUNS)
# ---------------------------------------------------------
N_RUNS = int(os.environ.get("N_RUNS", "10"))
SEEDS = [RANDOM_STATE + i for i in range(N_RUNS)]

USE_GPU = torch.cuda.is_available()
num_cols = config.get("numerical_features", [])
cat_cols = config.get("categorical_features", [])

OUT = "outputs/metrics_v2"
os.makedirs(OUT, exist_ok=True)
RUNS_CSV = f"{OUT}/utility_runs.csv"
FID_CSV = f"{OUT}/fidelity_runs.csv"
print(f"Dataset: {DATASET} | GPU: {USE_GPU} | semillas: {SEEDS}")
# ---------------------------------------------------------
# DATOS: entrenamiento interno (notebook 03) y prueba (notebook 01)
# ---------------------------------------------------------
SPLIT_DIR, DATA_DIR = "outputs/splits", "outputs/data"

X_tr_proc = pd.read_csv(f"{SPLIT_DIR}/X_train_in.csv")      # entrenamiento interno, escalado
X_tr_raw  = pd.read_csv(f"{SPLIT_DIR}/X_train_raw_in.csv")  # entrenamiento interno, escala original
y_tr      = pd.read_csv(f"{SPLIT_DIR}/y_train_in.csv")["y"].values
X_test    = pd.read_csv(f"{DATA_DIR}/X_test.csv")           # prueba, escalada (solo evaluación final)
y_test    = pd.read_csv(f"{DATA_DIR}/y_test.csv")["y"].values
scaler    = joblib.load(f"{DATA_DIR}/scaler.pkl")

assert len(X_tr_proc) == len(X_tr_raw) == len(y_tr), "Train interno RAW y escalado desalineados"
X_tr_raw = X_tr_raw[X_tr_proc.columns]  # mismo orden de columnas

cnt = Counter(y_tr)
minority, majority = min(cnt, key=cnt.get), max(cnt, key=cnt.get)
M, m = cnt[majority], cnt[minority]
n_to_generate = M - m
real_min_raw = X_tr_raw[y_tr == minority].reset_index(drop=True)
cat_idx = [X_tr_proc.columns.get_loc(c) for c in cat_cols]

print(f"Train interno: {dict(cnt)} | a generar: {n_to_generate}")
print(f"Test: {dict(Counter(y_test))}")

# ---------------------------------------------------------
# HIPERPARÁMETROS SELECCIONADOS (notebooks 04 a 07). No se vuelven a ajustar.
# ---------------------------------------------------------
def load_json(p):
    with open(p) as f:
        return json.load(f)

cfg_smote  = load_json("outputs/smote/best_config.json")
cfg_ctgan  = load_json("outputs/ctgan/best_config.json")
cfg_tvae   = load_json("outputs/tvae/best_config.json")
cfg_hybrid = load_json("outputs/hybrid/best_config.json")

K_SMOTE = cfg_smote["params"]["k_neighbors"]
P_CTGAN = cfg_ctgan["best_params"]
P_TVAE  = cfg_tvae["best_params"]
P_HYB   = cfg_hybrid["best_params"]
ALPHA   = cfg_hybrid.get("alpha_smote", 0.5)

# Resumen de la configuración usada (Tabla A1 del artículo)
with open(f"{OUT}/used_configs.json", "w") as f:
    json.dump({"dataset": DATASET, "smote_k": K_SMOTE, "ctgan": P_CTGAN, "tvae": P_TVAE,
               "hybrid": P_HYB, "alpha_smote": ALPHA,
               "train_in": {str(k): int(v) for k, v in cnt.items()},
               "test": {str(k): int(v) for k, v in Counter(y_test).items()},
               "n_features": X_tr_proc.shape[1], "n_num": len(num_cols), "n_cat": len(cat_cols),
               "proxy_scores": {"smote": cfg_smote.get("best_auc_pr"), "ctgan": cfg_ctgan.get("best_auc_pr"),
                                "tvae": cfg_tvae.get("best_auc_pr"), "hybrid": cfg_hybrid.get("best_auc_pr")}},
              f, indent=4)
print("K SMOTE:", K_SMOTE, "| CTGAN:", P_CTGAN, "| TVAE:", P_TVAE, "| HYBRID:", P_HYB, "alpha:", ALPHA)
# ---------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if USE_GPU:
        torch.cuda.manual_seed_all(seed)

def free_gpu():
    if USE_GPU:
        torch.cuda.empty_cache()

def to_scaled(df_raw):
    """Escala original -> espacio escalado de los clasificadores (scaler del notebook 01)."""
    out = df_raw[X_tr_proc.columns].copy()
    if num_cols:
        out[num_cols] = scaler.transform(out[num_cols].astype(float))
    return out

def to_raw(df_proc):
    out = df_proc.copy()
    if num_cols:
        out[num_cols] = scaler.inverse_transform(out[num_cols].astype(float))
    return out

def make_smote(k, strategy, seed):
    if cat_idx:
        return SMOTENC(categorical_features=cat_idx, k_neighbors=k,
                       sampling_strategy=strategy, random_state=seed)
    return SMOTE(k_neighbors=k, sampling_strategy=strategy, random_state=seed)

# ---------------------------------------------------------
# GENERADORES: cada uno devuelve SOLO las muestras sintéticas nuevas, en RAW
# (misma lógica que los notebooks 04 a 07, con la semilla de la corrida)
# ---------------------------------------------------------
def gen_smote(seed):
    sm = make_smote(K_SMOTE, {minority: M}, seed)
    X_res, _ = sm.fit_resample(X_tr_proc, y_tr)
    X_new = pd.DataFrame(X_res, columns=X_tr_proc.columns).iloc[len(X_tr_proc):]  # imblearn añade las nuevas al final
    return to_raw(X_new).reset_index(drop=True)

def gen_ctgan(seed, train_df, params):
    set_seed(seed)
    model = CTGAN(epochs=params["epochs"], batch_size=params["batch_size"], pac=params["pac"],
                  verbose=False, enable_gpu=USE_GPU)
    model.fit(train_df, cat_cols)
    synth = model.sample(n_to_generate)
    del model; free_gpu()
    return synth[X_tr_proc.columns].reset_index(drop=True)

def gen_tvae(seed):
    md = SingleTableMetadata()
    md.detect_from_dataframe(real_min_raw)
    for c in cat_cols:
        md.update_column(c, sdtype="categorical")
    set_seed(seed)
    model = TVAESynthesizer(metadata=md, epochs=P_TVAE["epochs"], batch_size=P_TVAE["batch_size"],
                            embedding_dim=P_TVAE["embedding_dim"], cuda=USE_GPU)
    model.fit(real_min_raw)
    synth = model.sample(num_rows=n_to_generate)
    del model; free_gpu()
    return synth[X_tr_proc.columns].reset_index(drop=True)

def gen_hybrid(seed):
    # Paso 1: SMOTE-NC solo hasta el piso alpha*M (Algoritmo 1)
    floor = int(math.ceil(ALPHA * M))
    sm = make_smote(K_SMOTE, {minority: floor}, seed)
    X_sm, y_sm = sm.fit_resample(X_tr_proc, y_tr)
    X_sm = pd.DataFrame(X_sm, columns=X_tr_proc.columns)
    # Paso 2: volver a escala original y quedarse con la minoría ampliada (real + SMOTE)
    min_expanded_raw = to_raw(X_sm[np.asarray(y_sm) == minority]).reset_index(drop=True)
    # Paso 3: CTGAN entrenado SOLO con esa minoría ampliada; genera M - m muestras
    # Paso 4 (en el bucle): set final = real + CTGAN (las muestras SMOTE se descartan)
    return gen_ctgan(seed, min_expanded_raw, P_HYB)

# ---------------------------------------------------------
# FIDELIDAD frente a la minoría real + diversidad de referencia de los registros reales
# ---------------------------------------------------------
metric_scaler = StandardScaler().fit(real_min_raw[num_cols]) if num_cols else None
real_min_sc = pd.DataFrame(metric_scaler.transform(real_min_raw[num_cols]), columns=num_cols) if num_cols else None

def fidelity(syn_raw):
    r = {}
    if num_cols:
        syn_sc = pd.DataFrame(metric_scaler.transform(syn_raw[num_cols].astype(float)), columns=num_cols)
        r["wasserstein_std"] = float(np.mean([wasserstein_distance(real_min_sc[c], syn_sc[c]) for c in num_cols]))
        smp = syn_sc.sample(min(len(syn_sc), 2000), random_state=42)
        r["diversity"] = float(np.mean(pdist(smp, metric="euclidean")))
    if len(num_cols) > 1:
        rc = real_min_raw[num_cols].corr().fillna(0).values
        sc = syn_raw[num_cols].astype(float).corr().fillna(0).values
        mask = ~np.eye(len(num_cols), dtype=bool)
        r["corr_diff_mean"] = float(np.mean(np.abs(rc - sc)[mask]))
    if cat_cols:
        diffs = []
        for c in cat_cols:
            a = real_min_raw[c].value_counts(normalize=True)
            b = syn_raw[c].astype(float).round().value_counts(normalize=True)
            diffs.append(sum(abs(a.get(v, 0) - b.get(v, 0)) for v in set(a.index) | set(b.index)))
        r["cat_diff_mean"] = float(np.mean(diffs))
    return r

REAL_DIVERSITY = (float(np.mean(pdist(real_min_sc.sample(min(len(real_min_sc), 2000), random_state=42))))
                  if num_cols else np.nan)
print(f"Diversidad de referencia de la minoría REAL: {REAL_DIVERSITY:.4f}")

# ---------------------------------------------------------
# CLASIFICADORES con configuración fija (300 árboles, tasa 0.05, profundidad 6)
# ---------------------------------------------------------
SPW = M / m  # peso de clase para la línea base sensible al costo

def build_models(seed, weighted=False):
    xgb_kw = dict(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
                  device="cuda" if USE_GPU else "cpu", random_state=seed)
    cat_kw = dict(iterations=300, learning_rate=0.05, depth=6, loss_function="Logloss",
                  verbose=False, random_seed=seed, task_type="GPU" if USE_GPU else "CPU")
    if weighted:
        xgb_kw["scale_pos_weight"] = SPW
        cat_kw["class_weights"] = {0: 1, 1: SPW}
    return {"XGBoost": XGBClassifier(**xgb_kw), "CatBoost": CatBoostClassifier(**cat_kw)}

def evaluate(model, X_train, y_train):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {"AUC_PR": average_precision_score(y_test, y_score),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "BalancedAcc": balanced_accuracy_score(y_test, y_pred),
            "Specificity": spec, "GMean": math.sqrt(rec * spec),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
# ---------------------------------------------------------
# BUCLE DE CORRIDAS (con reanudación)
# ---------------------------------------------------------
done = set()
if os.path.exists(RUNS_CSV):
    done = set(pd.read_csv(RUNS_CSV)["seed"].unique())
    print("Semillas ya completadas (se saltan):", sorted(done))

GENERATORS = {"SMOTE": gen_smote,
              "CTGAN": lambda s: gen_ctgan(s, real_min_raw, P_CTGAN),
              "TVAE": gen_tvae,
              "HYBRID": gen_hybrid}

for run_idx, seed in enumerate(SEEDS, start=1):
    if seed in done:
        continue
    print(f"\n=== {DATASET} | RUN {run_idx}/{N_RUNS} | seed={seed} ===")
    set_seed(seed)

    # 1) Conjuntos de entrenamiento: todos parten del mismo entrenamiento interno real
    train_sets = {"REAL": (X_tr_proc, y_tr, False),
                  "REAL_CW": (X_tr_proc, y_tr, True)}
    fid_rows = []
    for name, gen in GENERATORS.items():
        t0 = time.time()
        syn_raw = gen(seed)
        gen_time = time.time() - t0
        assert len(syn_raw) == n_to_generate, f"{name}: generó {len(syn_raw)} de {n_to_generate}"
        fid_rows.append({"dataset": DATASET, "run": run_idx, "seed": seed, "method": name,
                         "gen_time_s": round(gen_time, 1), "real_diversity_ref": REAL_DIVERSITY,
                         **fidelity(syn_raw)})
        X_aug = pd.concat([X_tr_proc, to_scaled(syn_raw)], ignore_index=True)
        y_aug = np.concatenate([y_tr, np.full(len(syn_raw), minority)])
        train_sets[name] = (X_aug, y_aug, False)
        print(f"  {name}: generado en {gen_time/60:.1f} min")

    # 2) Clasificadores con la semilla de la corrida
    util_rows = []
    for ts_name, (Xt, yt, weighted) in train_sets.items():
        for clf_name, model in build_models(seed, weighted).items():
            util_rows.append({"dataset": DATASET, "run": run_idx, "seed": seed,
                              "train_set": ts_name, "model": clf_name, **evaluate(model, Xt, yt)})

    # 3) Guardado inmediato (si se corta, se reanuda desde aquí)
    pd.DataFrame(util_rows).to_csv(RUNS_CSV, mode="a", header=not os.path.exists(RUNS_CSV), index=False)
    pd.DataFrame(fid_rows).to_csv(FID_CSV, mode="a", header=not os.path.exists(FID_CSV), index=False)
    print(f"  Semilla {seed} guardada")

# ---------------------------------------------------------
# RESUMEN (las pruebas estadísticas están en analysis/stats_tests.py)
# ---------------------------------------------------------
df = pd.read_csv(RUNS_CSV)
print("\nCorridas completas:", df["seed"].nunique())
print(df.groupby(["train_set", "model"])[["AUC_PR", "Recall", "Precision", "F1", "GMean"]]
        .agg(["mean", "std"]).round(4))
print(pd.read_csv(FID_CSV).groupby("method")[["wasserstein_std", "diversity", "gen_time_s"]].mean().round(4))
