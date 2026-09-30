#!/bin/bash
# ---------------------------------------------------------------------------
# Ejecutor ESTANDAR del TFM en hennessy: construye las imagenes (una por DSL) y
# lanza experimentos en contenedores efimeros con ~/tfm_entorno montado.
#
# DSLs / imagenes (targets del Dockerfile multi-stage, etiqueta tfm-<dsl>:ngc-arm64):
#   triton-tlx   Triton estandar + TLX   (src/*Kernels/triton/, triton_tlx/)
#   gluon        Gluon                   (src/*Kernels/gluon/)
#   helion       Helion                  (src/*Kernels/helion/)
#   cutlass      Cutlass/CuTe, C++/CUDA  (src/*Kernels/cutlass/, CMakeLists.txt)
#
# Uso (en cola, desde el login o desde hennessy):
#   sbatch ~/hennessy/tfm_entorno/ejecutar.sh imagen [dsl|todas]
#   sbatch ~/hennessy/tfm_entorno/ejecutar.sh validar [dsl|todas]
#   sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp run_matmul [args...]
#   sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp run_matmul_helion      # imagen helion
#   DSL=gluon sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp benchmarks/otro.py
#   (desde el login tambien vale: /machines/hennessy/home/mariag/tfm_entorno/ejecutar.sh)
#
# Uso directo (dentro de una sesion ./srun_hennessy.sh en hennessy):
#   ~/tfm_entorno/ejecutar.sh exp plantilla --sizes 1024
#   ~/tfm_entorno/ejecutar.sh shell helion   # bash interactivo en el contenedor
#
# Comandos:
#   imagen [dsl]      construye la imagen del DSL (por defecto todas) y la valida
#   validar [dsl]     ejecuta benchmarks/validar_gpu.py en la imagen (por defecto todas)
#   exp <exp> [args]  ejecuta un benchmark: nombre (benchmarks/<exp>.py) o ruta .py
#   shell [dsl]       bash interactivo en el contenedor (solo con srun --pty)
#
# Imagen de 'exp': la variable DSL o, si no, el nombre del benchmark
# (run_matmul_helion -> helion, run_rmsnorm_gluon -> gluon, *_cutlass -> cutlass);
# por defecto triton-tlx.
#
# Variables opcionales:
#   DSL=...           imagen para 'exp' (triton-tlx | gluon | helion | cutlass)
#   VALIDAR=0         'exp' no ejecuta la validacion previa (por defecto 1)
#   REBUILD=1         'exp' reconstruye la imagen antes (por defecto solo si falta)
#   IMAGE=...         otra etiqueta de imagen (por defecto tfm-<dsl>:ngc-arm64)
#
# Log:        ~/hennessy/tfm_entorno/logs/<nombre-job>-<JOBID>.out
# Resultados: ~/hennessy/tfm_entorno/results/<experimento>/<fecha>_job<JOBID>/
# ---------------------------------------------------------------------------

#SBATCH -p hennessy-benchmark
#SBATCH --chdir=/home/mariag/tfm_entorno
#SBATCH --cpus-per-task=14
#SBATCH -J tfm
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

DSLS=(triton-tlx gluon helion cutlass)
IMAGE_FIJA="${IMAGE:-}"
DSL="${DSL:-}"
IMAGE=""
VALIDAR="${VALIDAR:-1}"
REBUILD="${REBUILD:-0}"

# Con sbatch el script se copia al spool de Slurm: la raiz es el --chdir.
# Ejecutado a mano, la raiz es el directorio del propio script.
if [ -z "${SLURM_JOB_ID:-}" ] || [ -f "$(dirname "$0")/Dockerfile" ]; then
    RAIZ="$(cd "$(dirname "$0")" && pwd)"
else
    RAIZ="$(pwd)"
fi
cd "${RAIZ}"
mkdir -p logs results

uso() { sed -n '3,/^# ---/p' "${RAIZ}/ejecutar.sh" | sed '$d; s/^# \{0,1\}//'; exit "${1:-1}"; }

# Fija DSL e IMAGE para el DSL dado.
usar_dsl() {
    local d
    for d in "${DSLS[@]}"; do
        if [ "$1" = "${d}" ]; then
            DSL="${d}"
            IMAGE="${IMAGE_FIJA:-tfm-${DSL}:ngc-arm64}"
            return
        fi
    done
    echo "ERROR: DSL desconocido '$1' (validos: ${DSLS[*]})." >&2
    exit 1
}

# Lista de DSLs de 'imagen'/'validar': uno concreto o todos.
dsls_pedidos() {
    if [ -z "${1:-}" ] || [ "$1" = "todas" ]; then echo "${DSLS[@]}"; else echo "$1"; fi
}

# DSL de un benchmark por su nombre: run_matmul_helion -> helion. Por defecto triton-tlx.
dsl_de_exp() {
    local nombre="_$(basename "$1" .py)_" d
    for d in gluon helion cutlass; do
        if [[ "${nombre}" == *"_${d}_"* ]]; then echo "${d}"; return; fi
    done
    echo "triton-tlx"
}

cabecera() {
    echo "== ${1} | job ${SLURM_JOB_ID:-local} | nodo $(hostname) | $(date '+%F %T') =="
    echo "== Raiz: ${RAIZ} | DSL: ${DSL:-?} | imagen: ${IMAGE:-?} =="
}

comprobar_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: no puedo usar Docker en $(hostname). Revisa permisos (grupo docker)."
        exit 1
    fi
}

construir_imagen() {
    echo "== Construyendo imagen ${IMAGE} (target ${DSL}, contexto: ${RAIZ}) =="
    docker build --target "${DSL}" -t "${IMAGE}" "${RAIZ}"
}

asegurar_imagen() {
    if [ "${REBUILD}" = "1" ] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
        construir_imagen
    fi
}

# La GPU debe estar libre para que las medidas sean fiables.
estado_gpu() {
    echo "== Estado de la GPU =="
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,temperature.gpu --format=csv || true
    local ocupada
    ocupada=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
    if [ -n "${ocupada}" ]; then
        echo "AVISO: hay procesos usando la GPU; las medidas pueden no ser fiables:"
        echo "${ocupada}"
    fi
}

# Ejecuta un comando en un contenedor efimero con la raiz montada en /workspace/tfm.
# --user: resultados a nombre de mariag (no root). HOME=/tmp: caches escribibles.
# USER/LOGNAME: el UID no existe en el /etc/passwd de la imagen y getpass.getuser()
# (lo usa TorchInductor, que usa Helion) fallaria con "getpwuid(): uid not found".
# TRITON_CACHE_DIR persistente: no recompila kernels entre ejecuciones.
en_contenedor() {
    local tty=()
    [ -t 0 ] && [ -t 1 ] && tty=(-it)
    docker run --rm --gpus all ${tty[@]+"${tty[@]}"} \
        -v "${RAIZ}":/workspace/tfm -w /workspace/tfm \
        --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
        -e PYTHONPATH=/workspace/tfm/src:/workspace/tfm/benchmarks \
        -e TRITON_CACHE_DIR=/workspace/tfm/.cache/triton \
        -e TFM_RAIZ=/workspace/tfm \
        -e TFM_IMAGEN="${IMAGE}@$(docker image inspect -f '{{.Id}}' "${IMAGE}")" \
        -e TFM_HOST="$(hostname)" \
        -e TFM_DSL="${DSL}" \
        -e SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
        -e TFM_EXPERIMENTO="${TFM_EXPERIMENTO:-}" \
        "${IMAGE}" "$@"
}

validar() {
    echo "== Validacion GPU ${DSL} (benchmarks/validar_gpu.py) =="
    en_contenedor python /workspace/tfm/benchmarks/validar_gpu.py
}

# Acepta 'plantilla', 'plantilla.py' o 'benchmarks/plantilla.py'.
resolver_exp() {
    local e="$1"
    for c in "${e}" "benchmarks/${e}" "benchmarks/${e}.py" "${e}.py"; do
        if [ -f "${RAIZ}/${c}" ]; then echo "${c}"; return; fi
    done
    echo "ERROR: no encuentro el experimento '${e}' (ni en ${RAIZ} ni en ${RAIZ}/benchmarks)." >&2
    exit 1
}

COMANDO="${1:-}"
[ $# -gt 0 ] && shift

case "${COMANDO}" in
    imagen|validar)
        comprobar_docker
        fallos=()
        for d in $(dsls_pedidos "${1:-}"); do
            usar_dsl "${d}"
            if [ "${COMANDO}" = "imagen" ]; then
                cabecera "Construir imagen ${DSL}"
                construir_imagen
            else
                cabecera "Validar imagen ${DSL}"
                asegurar_imagen
            fi
            validar || fallos+=("${DSL}")
        done
        if [ ${#fallos[@]} -gt 0 ]; then
            echo "ERROR: validacion fallida en: ${fallos[*]}"
            exit 1
        fi
        ;;
    exp)
        [ $# -ge 1 ] || uso
        EXP="$(resolver_exp "$1")"
        shift
        export TFM_EXPERIMENTO="$(basename "${EXP}" .py)"
        usar_dsl "${DSL:-$(dsl_de_exp "${EXP}")}"
        cabecera "Experimento ${TFM_EXPERIMENTO}: ${EXP} $*"
        comprobar_docker
        asegurar_imagen
        estado_gpu
        if [ "${VALIDAR}" = "1" ] && ! validar; then
            echo "ERROR: la validacion GPU ha fallado; se aborta el experimento."
            exit 1
        fi
        echo "== Ejecutando ${EXP} $* =="
        t0=$(date +%s)
        en_contenedor python "/workspace/tfm/${EXP}" "$@"
        echo "== Experimento terminado en $(( $(date +%s) - t0 )) s =="
        ;;
    shell)
        usar_dsl "${1:-${DSL:-triton-tlx}}"
        comprobar_docker
        asegurar_imagen
        en_contenedor bash
        ;;
    -h|--help|help)
        uso 0
        ;;
    *)
        uso
        ;;
esac

echo "== Finalizado: $(date '+%F %T') =="
