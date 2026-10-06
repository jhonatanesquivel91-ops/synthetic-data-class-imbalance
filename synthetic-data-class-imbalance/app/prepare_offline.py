"""Descarga los datasets de ejemplo de la demo para usarla sin conexión.

Ejecutar una vez, con Internet, desde la raíz del repositorio:
    python app/prepare_offline.py
Guarda los archivos en app/data/; la app los usa automáticamente.
"""
import os

import pandas as pd
from ucimlrepo import fetch_ucirepo

EXAMPLES = {915: "Thyroid Cancer Recurrence", 890: "AIDS Clinical Trial 175"}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for uid, name in EXAMPLES.items():
        path = os.path.join(DATA_DIR, f"uci_{uid}.csv")
        if os.path.exists(path):
            print(f"Ya existe: {name} ({path})")
            continue
        d = fetch_ucirepo(id=uid)
        df = pd.concat([d.data.features, d.data.targets], axis=1)
        df.to_csv(path, index=False)
        print(f"Guardado: {name}, {len(df)} registros -> {path}")
    print("Listo. La demo puede usarse sin conexión.")


if __name__ == "__main__":
    main()
