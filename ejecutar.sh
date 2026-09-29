#!/bin/bash
# ---------------------------------------------------------------------------
# Ejecutor ESTANDAR del TFM en hennessy: construye la imagen y lanza
# experimentos en contenedores efimeros con ~/tfm_entorno montado.
#
# Uso (en cola, desde el login o desde hennessy):
#   sbatch ~/hennessy/tfm_entorno/ejecutar.sh imagen
#   sbatch ~/hennessy/tfm_entorno/ejecutar.sh validar
#   sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp experimento_cero [args...]
#   sbatch -J mi-exp ~/hennessy/tfm_entorno/ejecutar.sh exp src/otro.py --dtype bf16
#   (desde el login tambien vale: /machines/hennessy/home/mariag/tfm_entorno/ejecutar.sh)
#
# Uso directo (dentro de una sesion ./srun_hennessy.sh en hennessy):
#   ~/tfm_entorno/ejecutar.sh exp experimento_cero --sizes 1048576
#   ~/tfm_entorno/ejecutar.sh shell          # bash interactivo en el contenedor
#
# Comandos:
#   imagen            construye tfm:ngc-arm64 desde ./Dockerfile y la valida
#   validar           ejecuta src/validar_gpu.py (PyTorch/Triton/TLX/Helion)
#   exp <exp> [args]  ejecuta un experimento: nombre (src/<exp>.py) o ruta .py
#   shell             bash interactivo en el contenedor (solo con srun --pty)
#
# Variables opcionales:
#   VALIDAR=0         'exp' no ejecuta la validacion previa (por defecto 1)
#   REBUILD=1         'exp' reconstruye la imagen antes (por defecto solo si falta)
#   IMAGE=...         otra etiqueta de imagen (por defecto tfm:ngc-arm64)
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

IMAGE="${IMAGE:-tfm:ngc-arm64}"
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

uso() { sed -n '2,32p' "${RAIZ}/ejecutar.sh" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

cabecera() {
    echo "== ${1} | job ${SLURM_JOB_ID:-local} | nodo $(hostname) | $(date '+%F %T') =="
    echo "== Raiz: ${RAIZ} | imagen: ${IMAGE} =="
}

comprobar_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: no puedo usar Docker en $(hostname). Revisa permisos (grupo docker)."
        exit 1
    fi
}

construir_imagen() {
    echo "== Construyendo imagen ${IMAGE} (contexto: ${RAIZ}) =="
    docker build -t "${IMAGE}" "${RAIZ}"
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
        -e PYTHONPATH=/workspace/tfm/src \
        -e TRITON_CACHE_DIR=/workspace/tfm/.cache/triton \
        -e TFM_RAIZ=/workspace/tfm \
        -e TFM_IMAGEN="${IMAGE}@$(docker image inspect -f '{{.Id}}' "${IMAGE}")" \
        -e TFM_HOST="$(hostname)" \
        -e SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
        -e TFM_EXPERIMENTO="${TFM_EXPERIMENTO:-}" \
        "${IMAGE}" "$@"
}

validar() {
    echo "== Validacion GPU (src/validar_gpu.py) =="
    en_contenedor python /workspace/tfm/src/validar_gpu.py
}

# Acepta 'experimento_cero', 'experimento_cero.py' o 'src/experimento_cero.py'.
resolver_exp() {
    local e="$1"
    for c in "${e}" "src/${e}" "src/${e}.py" "${e}.py"; do
        if [ -f "${RAIZ}/${c}" ]; then echo "${c}"; return; fi
    done
    echo "ERROR: no encuentro el experimento '${e}' (ni en ${RAIZ} ni en ${RAIZ}/src)." >&2
    exit 1
}

COMANDO="${1:-}"
[ $# -gt 0 ] && shift

case "${COMANDO}" in
    imagen)
        cabecera "Construir imagen"
        comprobar_docker
        construir_imagen
        validar
        ;;
    validar)
        cabecera "Validar imagen"
        comprobar_docker
        asegurar_imagen
        validar
        ;;
    exp)
        [ $# -ge 1 ] || uso
        EXP="$(resolver_exp "$1")"
        shift
        export TFM_EXPERIMENTO="$(basename "${EXP}" .py)"
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
