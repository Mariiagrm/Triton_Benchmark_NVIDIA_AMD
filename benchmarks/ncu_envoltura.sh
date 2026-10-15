#!/bin/bash
# Dentro del contenedor con TFM_NCU=1 (lo usa en_contenedor de ejecutar.sh): ejecuta '<cmd...>' y,
# si es un benchmark (run_*.py), bajo Nsight Compute; despues une sus metricas a los resultados
# (benchmarks/ncu.py --fusionar). El contenedor corre como root (los contadores de la GPU exigen
# administrador, RmProfilingAdminOnly=1): al acabar, todo lo escrito vuelve a ser del usuario.
set -u
es_benchmark=0
for a in "$@"; do
    case "$(basename "${a}")" in run_pico_mma.py) ;; run_*.py) es_benchmark=1 ;; esac
done
if [ "${es_benchmark}" = 1 ]; then
    mkdir -p "$(dirname "${TFM_NCU_CSV}")"
    metricas="$(python /workspace/tfm/benchmarks/ncu.py --metricas)"
    echo "== ncu: $(tr ',' '\n' <<< "${metricas}" | wc -l) metricas -> ${TFM_NCU_CSV#/workspace/tfm/} =="
    ncu --profile-from-start off --nvtx --clock-control none --print-units base --csv --page raw \
        --metrics "${metricas}" --log-file "${TFM_NCU_CSV}" "$@"
    rc=$?
    python /workspace/tfm/benchmarks/ncu.py --fusionar "${TFM_NCU_CSV}" || rc=1
else
    "$@"
    rc=$?
fi
chown -R "${TFM_UID}" /workspace/tfm/results /workspace/tfm/.cache /workspace/docs 2>/dev/null
exit "${rc}"
