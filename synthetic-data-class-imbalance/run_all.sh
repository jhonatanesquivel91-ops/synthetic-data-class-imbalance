#!/usr/bin/env bash
# Ejecuta el protocolo de 10 corridas (10_multirun_fair.py) en los cinco datasets, uno tras otro.
# Uso, desde la raíz del repositorio:  bash run_all.sh
# Requiere haber ejecutado antes los notebooks 01 a 07 de cada dataset.
# Las semillas ya completadas se omiten, por lo que se puede relanzar si se interrumpe.
export N_RUNS=10
export PYTHONUNBUFFERED=1
PROJ="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ" || exit 1
mkdir -p logs
# Orden: del dataset más rápido al más lento
for pref in dataset_02 dataset_03 dataset_05 dataset_04 dataset_01; do
  for ds in ${pref}_*/; do
    [ -f "$ds/dataset_config.yaml" ] || continue
    ds="${ds%/}"
    echo "[$(date '+%F %T')] >>> Iniciando $ds"
    cp "$PROJ/10_multirun_fair.py" "$PROJ/$ds/"
    ( cd "$PROJ/$ds" && python3 10_multirun_fair.py ) > "logs/$ds.log" 2>&1 \
      && echo "[$(date '+%F %T')] OK  $ds" \
      || echo "[$(date '+%F %T')] ERROR en $ds (ver logs/$ds.log); sigo con el siguiente"
  done
done
echo "[$(date '+%F %T')] Terminado."
