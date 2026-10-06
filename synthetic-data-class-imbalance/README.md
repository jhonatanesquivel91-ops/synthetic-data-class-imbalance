# Datos sintéticos para clasificación binaria con desbalance de clases

Código, resultados y demo del artículo:

> J. Esquivel, C. Humpiri, J. Vega y N. Saboya, «Comparative Evaluation of Synthetic Data Generation Methods for Binary Classification Problems with Class Imbalance», *International Journal of Advanced Computer Science and Applications (IJACSA)*, vol. 17, n.º 9, 2026. DOI: [10.14569/IJACSA.2026.0170971](https://doi.org/10.14569/IJACSA.2026.0170971)

Tesis para obtener el título profesional de Ingeniero de Sistemas · Facultad de Ingeniería y Arquitectura · Universidad Peruana Unión.
Autores de la tesis: Jhonatan Raul Esquivel Godoy, Christian Quispe Humpiri y José David Vega Vega · Asesor: Dr. Nemias Saboya.

**English summary.** This repository compares four synthetic data generation paradigms for imbalanced tabular binary classification (SMOTE-NC, CTGAN, TVAE and a hybrid SMOTE-NC → CTGAN) on five UCI datasets with imbalance ratios from 2.55:1 to 29.99:1. It combines statistical fidelity and predictive utility (XGBoost and CatBoost) over 10 runs that regenerate the synthetic data, with paired tests and Holm correction, and includes a cost-sensitive baseline (class weights).

## Resultado principal

La síntesis desplaza la frontera de decisión hacia la clase minoritaria: el **Recall sube significativamente en 27 de 40 comparaciones** y nunca baja, mientras el **AUC-PR baja en 19 y solo sube en 2** (DS5 con XGBoost). Los pesos de clase (REAL-CW) logran un desplazamiento similar sin generar datos. SMOTE-NC es el método más fiel en los cinco datasets.

## Estructura

```
├── 10_multirun_fair.py        # Protocolo final: 10 corridas, generación + fidelidad + utilidad
├── run_all.sh                 # Ejecuta 10_multirun_fair.py en los 5 datasets
├── utils/                     # Carga de configuración y utilidades
├── dataset_0X_<nombre>/
│   ├── dataset_config.yaml    # Variables numéricas, categóricas y objetivo (id UCI)
│   ├── 01_data_prep.ipynb     # Descarga desde UCI, partición 80/20 y escalado (sin fuga)
│   ├── 03_internal_split.ipynb # Partición interna entrenamiento/validación
│   ├── 04–07_*.ipynb          # Búsqueda de hiperparámetros: SMOTE-NC, CTGAN, TVAE, híbrido
│   └── outputs/
│       ├── */best_config.json # Hiperparámetros seleccionados (Tabla A1)
│       └── metrics_v2/        # Resultados de las 10 corridas (utilidad y fidelidad)
├── analysis/
│   ├── stats_tests.py         # Pruebas pareadas, Holm, d_z e IC 95 % (Tabla IV)
│   ├── make_figures.py        # Figuras 3 y 5–14
│   └── fig4_fidelity_ds1.py   # Figura 4 (se ejecuta dentro de dataset_01_cdc_diabetes)
├── results/                   # stats_full.csv, summary_full.csv, figuras y tabla suplementaria
└── demo_sustentacion.ipynb    # Demo: protocolo completo, sintetizado, sobre un dataset
```

| Carpeta | Dataset | Dominio | Ratio | Fuente |
| --- | --- | --- | --- | --- |
| dataset_01_cdc_diabetes | Diabetes Health Indicators | Salud | 6.18:1 | UCI 891 |
| dataset_02_thyroid_cancer | Thyroid Cancer Recurrence | Salud | 2.55:1 | UCI 915 |
| dataset_03_aids_trial_175 | AIDS Clinical Trial 175 | Salud | 3.10:1 | UCI 890 |
| dataset_04_credit_default | Default of Credit Card Clients | Finanzas | 3.52:1 | UCI 350 |
| dataset_05_taiwanese_bankruptcy | Taiwanese Bankruptcy Prediction | Finanzas | 29.99:1 | UCI 572 |

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Se usó Python 3.10 y una GPU para CTGAN y TVAE (funciona también en CPU, más lento).

## Reproducir los resultados

Los resultados del artículo ya están en `dataset_*/outputs/metrics_v2/` y `results/`. Para regenerarlos:

1. **Preparar cada dataset.** Dentro de cada carpeta `dataset_0X_*`, ejecutar en orden `01_data_prep.ipynb` y `03_internal_split.ipynb`. Ejecuta el 03 una sola vez después del 01: reescribe la partición de entrenamiento con el subconjunto interno.
2. **Hiperparámetros (opcional).** Los notebooks `04` a `07` repiten la búsqueda en malla; sus resultados ya están en `outputs/*/best_config.json`.
3. **Protocolo de 10 corridas.** Desde la raíz: `bash run_all.sh` (usa `N_RUNS=10`; guarda cada semilla y se reanuda si se interrumpe). DS1 es el más lento.
4. **Estadística y figuras.**
   ```bash
   python analysis/stats_tests.py
   python analysis/make_figures.py
   ```

## Usar el pipeline con otro dataset

Crea una carpeta `dataset_06_<nombre>/` con un `dataset_config.yaml` como este y repite los pasos anteriores:

```yaml
dataset_name: mi_dataset
uci_id: 123              # o adapta 01_data_prep.ipynb para leer un CSV propio
dataset: 6
ratio: 4.2
target_column: objetivo
drop_columns: []
numerical_features: [edad, ingreso]
categorical_features: [sexo, region]
```

## Demo

`demo_sustentacion.ipynb` ejecuta el protocolo de la tesis de principio a fin sobre un dataset (DS2, DS3 o DS5): partición sin fuga, generación, fidelidad, utilidad con XGBoost y CatBoost, pruebas pareadas con Holm, gráficos y comparación con las 10 corridas publicadas.

```bash
jupyter notebook demo_sustentacion.ipynb
```

- `MODO = "rapido"`: SMOTE-NC, 5 corridas (1 a 2 minutos en CPU).
- `MODO = "completo"`: los cuatro métodos, 3 corridas (15 a 25 minutos en CPU).

La primera ejecución descarga el dataset de la UCI y lo guarda en `data/`; después funciona sin Internet.

## Datos

Los datasets se descargan del [UCI Machine Learning Repository](https://archive.ics.uci.edu) con `ucimlrepo` y no se incluyen en el repositorio. Cita las fuentes originales si los usas (ver referencias [29]–[36] del artículo).

## Licencia y cita

Código bajo licencia MIT. Si usas este trabajo, cita el artículo (ver `CITATION.cff`).
