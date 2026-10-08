#!/bin/bash
# ---------------------------------------------------------------------------
# Ejecutor ESTANDAR del TFM: construye las imagenes (una por DSL) y lanza
# experimentos en contenedores efimeros con el repo montado. Vale para cualquier
# nodo (hennessy, patterson...): nada de la maquina esta fijado en el script.
#
# DSLs / imagenes (targets del Dockerfile multi-stage, etiqueta tfm-<dsl>:ngc-arm64):
#   triton-tlx   Triton estandar + TLX   (src/*Kernels/triton/, triton_tlx/)
#   gluon        Gluon                   (src/*Kernels/gluon/)
#   helion       Helion                  (src/*Kernels/helion/)
#   cutlass      Cutlass/CuTe, C++/CUDA  (src/*Kernels/cutlass/, CMakeLists.txt)
#
# Uso en cola ('encolar' llama a sbatch con la particion y rutas del nodo del repo;
# desde el login, con el repo en /machines/<nodo>/..., o desde el propio nodo):
#   bash ~/hennessy/tfm_entorno/ejecutar.sh encolar imagen [dsl|todas]
#   bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul [args...]
#   bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul_helion     # imagen helion
#   PARTICION=hennessy-test bash ~/hennessy/tfm_entorno/ejecutar.sh encolar validar gluon
#   (tambien vale sbatch a mano: sbatch -p <particion> --chdir=<repo en el nodo> ejecutar.sh ...)
#
# Uso directo (dentro de una sesion srun en el nodo, sin cola):
#   ~/tfm_entorno/ejecutar.sh exp plantilla --sizes 1024
#   ~/tfm_entorno/ejecutar.sh exp triton-tlx       # todos los benchmarks Triton
#   ~/tfm_entorno/ejecutar.sh exp matmul           # matmul en TODOS los DSLs (cada uno en su imagen)
#   ~/tfm_entorno/ejecutar.sh exp gluon rmsnorm    # solo el rmsnorm de Gluon
#   ~/tfm_entorno/ejecutar.sh shell helion   # bash interactivo en el contenedor
#
# Comandos:
#   encolar <cmd...>  envia '<cmd...>' a la cola Slurm del nodo del repo (ver arriba)
#   imagen [dsl]      construye la imagen del DSL (por defecto todas) y la valida
#   validar [dsl]     ejecuta benchmarks/validar_gpu.py en la imagen (por defecto todas)
#   exp <exp> [args]  ejecuta un benchmark: nombre (se busca en benchmarks/ y
#                     benchmarks/<CB|MB>Kernels/<dsl>/) o ruta .py
#   exp <dsl> [args]  ejecuta TODOS los benchmarks de un DSL (CB y MB) en su imagen,
#                     p. ej. 'exp triton-tlx'; los args se pasan a todos
#   exp <alg> [args]  ejecuta los benchmarks de un algoritmo (run_<alg>.py, run_<alg>_*.py)
#                     de TODOS los DSLs, cada uno en su imagen: 'exp matmul', 'exp rmsnorm'
#   exp <dsl> <alg> [args]  solo los de ese algoritmo en ese DSL: 'exp helion matmul'
#   listar            lista los experimentos disponibles (run_*.py) y cuantos hay
#   shell [dsl]       bash interactivo en el contenedor (solo con srun --pty)
#
# Imagen de 'exp': la variable DSL o, si no, la carpeta del benchmark
# (benchmarks/<CB|MB>Kernels/<triton|gluon|cutlass|helion>/; triton -> triton-tlx).
# Fuera de esas carpetas: por el nombre (*_helion, *_gluon, *_cutlass) o triton-tlx.
#
# Variables opcionales:
#   PARTICION=...     'encolar': particion (por defecto <nodo>-benchmark)
#   TFM_MAQUINA=...   nombre del entorno en results/<maquina>/ (por defecto el nodo); p. ej.
#                     TFM_MAQUINA=hennessy-drv610 para separar otro driver en el mismo nodo
#   DSL=...           imagen para 'exp' (triton-tlx | gluon | helion | cutlass)
#   VALIDAR=0         'exp' no ejecuta la validacion previa (por defecto 1)
#   REBUILD=1         'exp' reconstruye la imagen antes (por defecto solo si falta)
#   IMAGE=...         otra etiqueta de imagen (por defecto tfm-<dsl>:ngc-arm64)
#   TFM_CACHE_FRIA=1  caches de compilacion vacias y efimeras (en el /tmp del contenedor):
#                     mide compilacion + autotuning en frio (t_compilacion_s, autotune_s).
#                     Usar con TFM_MAQUINA=<nodo>-frio, que el resumen de rendimiento ignora
#   TFM_DOCS=...      carpeta de documentos para la memoria (por defecto ../tfm/docs, fuera
#                     del repositorio)
#   TFM_ENERGIA=1     cada medida anade el regimen sostenido (potencia, J por llamada, GFLOP/J
#                     o GB/J; benchmarks/energia.py los reune). Entorno por defecto
#                     <nodo>-energia, que el resumen de rendimiento ignora. Duracion de cada
#                     medida: TFM_ENERGIA_SEGUNDOS (6) de los que se descartan
#                     TFM_ENERGIA_DESCARTE (2), mas 3 s de reposo
#
# Log (con 'encolar'): <repo>/logs/<nombre-job>-<JOBID>.out
# Resultados:         <repo>/results/<maquina>/<experimento>/<fecha>_job<JOBID>/
# ---------------------------------------------------------------------------

# Solo opciones independientes del nodo; particion, --chdir y log los pone 'encolar'.
#SBATCH --cpus-per-task=14
#SBATCH -J tfm
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
    if [ ! -f "${RAIZ}/Dockerfile" ]; then
        echo "ERROR: ${RAIZ} no es el repo (falta Dockerfile). Envia el trabajo con" \
             "'ejecutar.sh encolar ...' o con sbatch --chdir=<repo en el nodo>." >&2
        exit 1
    fi
fi
cd "${RAIZ}"
mkdir -p logs results
# Documentos para la memoria: fuera del repositorio, en ../tfm/docs (o TFM_DOCS). Se monta en el
# contenedor y benchmarks/informe.py escribe ahi tablas y figuras.
DOCS="${TFM_DOCS:-$(dirname "${RAIZ}")/tfm/docs}"
mkdir -p "${DOCS}"

# Envia este script a Slurm. El nodo sale de la ruta del repo: desde el login esta en
# /machines/<nodo>/<ruta en el nodo>; si no, es el nodo actual. Asi la particion
# (<nodo>-benchmark), el --chdir y el log apuntan a rutas validas DENTRO del nodo.
encolar() {
    [ $# -ge 1 ] || uso
    local real nodo ruta
    real="$(realpath "${RAIZ}")"
    if [[ "${real}" =~ ^/machines/([^/]+)(/.*)$ ]]; then
        nodo="${BASH_REMATCH[1]}"
        ruta="${BASH_REMATCH[2]}"
    else
        nodo="$(hostname -s)"
        ruta="${real}"
    fi
    local particion="${PARTICION:-${nodo}-benchmark}"
    local nombre="tfm-$1"
    [ "$1" = "exp" ] && [ -n "${2:-}" ] && nombre="$(basename "$2" .py)"
    echo "== Encolando '$*' en ${particion} (repo en el nodo: ${ruta}) =="
    sbatch -p "${particion}" -J "${nombre}" --chdir="${ruta}" \
        -o "${ruta}/logs/%x-%j.out" -e "${ruta}/logs/%x-%j.out" \
        "${real}/ejecutar.sh" "$@"
    echo "Log: logs/${nombre}-<JOBID>.out   Cola: squeue -p ${particion}"
}

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

# DSL de un benchmark: por su carpeta (benchmarks/CBKernels/helion/... -> helion) o, si
# no esta en una carpeta de DSL, por su nombre (run_matmul_helion -> helion). Por defecto triton-tlx.
dsl_de_exp() {
    local carpeta nombre="_$(basename "$1" .py)_" d
    carpeta="$(basename "$(dirname "$1")")"
    case "${carpeta}" in
        triton) echo "triton-tlx"; return ;;
        gluon|helion|cutlass) echo "${carpeta}"; return ;;
    esac
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
    local salida
    # Si nvidia-smi no puede hablar con el driver, Docker tampoco podra usar la GPU: abortar ya.
    if ! salida=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,temperature.gpu --format=csv 2>&1); then
        echo "${salida}"
        echo "ERROR: la GPU no es utilizable en $(hostname) (nvidia-smi falla)."
        if grep -qi "version mismatch" <<< "${salida}"; then
            echo "       'Driver/library version mismatch': el driver NVIDIA se ha actualizado y el"
            echo "       modulo del kernel cargado es el antiguo. Hay que reiniciar el nodo (o recargar"
            echo "       los modulos nvidia); compara: cat /proc/driver/nvidia/version"
        fi
        exit 1
    fi
    echo "${salida}"
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
# TRITON_CACHE_DIR persistente: no recompila kernels entre ejecuciones. Con TFM_CACHE_FRIA=1,
# TFM_CACHE (y con ella la de Triton) va al /tmp del contenedor: vacia en cada benchmark.
en_contenedor() {
    local tty=() cache=/workspace/tfm/.cache
    [ "${TFM_CACHE_FRIA:-0}" = "1" ] && cache=/tmp/tfm-cache
    [ -t 0 ] && [ -t 1 ] && tty=(-it)
    docker run --rm --gpus all ${tty[@]+"${tty[@]}"} \
        -v "${RAIZ}":/workspace/tfm -v "${DOCS}":/workspace/docs -e TFM_DOCS=/workspace/docs \
        -w /workspace/tfm \
        --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
        -e PYTHONPATH=/workspace/tfm/src:/workspace/tfm/benchmarks \
        -e TFM_CACHE="${cache}" -e TRITON_CACHE_DIR="${cache}/triton" \
        -e TFM_CACHE_FRIA="${TFM_CACHE_FRIA:-0}" \
        -e TFM_ENERGIA="${TFM_ENERGIA:-0}" \
        -e TFM_ENERGIA_SEGUNDOS="${TFM_ENERGIA_SEGUNDOS:-}" -e TFM_ENERGIA_DESCARTE="${TFM_ENERGIA_DESCARTE:-}" \
        -e TFM_RAIZ=/workspace/tfm \
        -e TFM_IMAGEN="${IMAGE}@$(docker image inspect -f '{{.Id}}' "${IMAGE}")" \
        -e TFM_HOST="$(hostname)" \
        -e TFM_MAQUINA="${TFM_MAQUINA:-$(hostname -s)${TFM_ENERGIA:+$([ "${TFM_ENERGIA}" = 1 ] && echo -energia)}}" \
        -e TFM_KERNEL="$(uname -r)" \
        -e TFM_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)" \
        -e TFM_DSL="${DSL}" \
        -e SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
        -e TFM_EXPERIMENTO="${TFM_EXPERIMENTO:-}" \
        "${IMAGE}" "$@"
}

validar() {
    echo "== Validacion GPU ${DSL} (benchmarks/validar_gpu.py) =="
    en_contenedor python /workspace/tfm/benchmarks/validar_gpu.py
}

# Carpeta de benchmarks de un DSL (triton-tlx/triton -> triton). Falla si no es un DSL.
carpeta_de_dsl() {
    case "$1" in
        triton-tlx|triton) echo "triton" ;;
        gluon|helion|cutlass) echo "$1" ;;
        *) return 1 ;;
    esac
}

# Benchmarks de un algoritmo (run_<alg>.py y run_<alg>_*.py) en la carpeta de DSL dada
# (o en todas con '*'). Imprime una ruta por linea; nada si no hay.
benchmarks_de_alg() {
    local f
    for f in "${RAIZ}"/benchmarks/*Kernels/$2/run_"$1".py "${RAIZ}"/benchmarks/*Kernels/$2/run_"$1"_*.py; do
        [ -f "${f}" ] && echo "${f#"${RAIZ}"/}"
    done | sort -t/ -k3,3 -k4,4
}

# Acepta 'run_matmul', 'run_matmul.py' o la ruta. Por nombre busca en benchmarks/ y en
# benchmarks/<CB|MB>Kernels/<dsl>/; si el nombre esta en varias carpetas, hay que dar la ruta.
resolver_exp() {
    local e="$1" c
    for c in "${e}" "${e}.py"; do
        if [ -f "${RAIZ}/${c}" ]; then echo "${c}"; return; fi
    done
    local hallados=()
    for c in benchmarks benchmarks/*Kernels/*; do
        [ -f "${RAIZ}/${c}/${e%.py}.py" ] && hallados+=("${c}/${e%.py}.py")
    done
    if [ ${#hallados[@]} -eq 1 ]; then echo "${hallados[0]}"; return; fi
    if [ ${#hallados[@]} -gt 1 ]; then
        echo "ERROR: '${e}' existe en varias carpetas; indica la ruta: ${hallados[*]}" >&2
    else
        echo "ERROR: no encuentro el experimento '${e}' (ni en ${RAIZ} ni en benchmarks/ o benchmarks/<CB|MB>Kernels/<dsl>/)." >&2
    fi
    exit 1
}

# Lista los benchmarks disponibles (run_*.py), agrupados por carpeta, y cuantos hay.
# Puro listado: no necesita Docker ni GPU (vale desde el login).
listar_experimentos() {
    local total=0 f rel dir prev=""
    echo "== Experimentos disponibles (benchmarks/<CB|MB>Kernels/<dsl>/run_*.py) =="
    while IFS= read -r f; do
        rel="${f#"${RAIZ}"/}"
        dir="$(dirname "${rel}")"
        [ "${dir}" != "${prev}" ] && { echo "  ${dir}/"; prev="${dir}"; }
        echo "      $(basename "${f}" .py)"
        total=$((total + 1))
    done < <(find "${RAIZ}/benchmarks" -type f -name 'run_*.py' 2>/dev/null | sort)
    echo
    echo "== ${total} experimentos en total =="
    echo "   Atajos: 'exp <alg>' (ese algoritmo en todos los DSLs), 'exp <dsl>' (todos los de"
    echo "   un DSL), 'exp <dsl> <alg>' (uno concreto). Encolar: 'encolar exp <nombre> [args]'."
}

COMANDO="${1:-}"
[ $# -gt 0 ] && shift

case "${COMANDO}" in
    encolar)
        encolar "$@"
        exit
        ;;
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
        # Sin argumentos: mostrar los experimentos disponibles y cuantos hay.
        if [ $# -eq 0 ]; then
            listar_experimentos
            exit 0
        fi
        # 'exp <dsl> [<alg>]': todos los benchmarks de ese DSL (CB y MB) o solo los de <alg>.
        # 'exp <alg>': los de ese algoritmo en todos los DSLs. Si no, un benchmark concreto.
        EXPS=()
        if CARPETA="$(carpeta_de_dsl "$1")"; then
            [ -z "${DSL}" ] && DSL="$1"
            [ "${DSL}" = "triton" ] && DSL=triton-tlx
            shift
            if [ -n "${1:-}" ] && [[ "$1" != -* ]]; then
                mapfile -t EXPS < <(benchmarks_de_alg "${1#run_}" "${CARPETA}")
                [ ${#EXPS[@]} -gt 0 ] || { echo "ERROR: no hay benchmarks run_${1#run_}*.py en benchmarks/<CB|MB>Kernels/${CARPETA}/." >&2; exit 1; }
                shift
            else
                for f in "${RAIZ}"/benchmarks/*Kernels/"${CARPETA}"/run_*.py; do
                    [ -f "${f}" ] && EXPS+=("${f#"${RAIZ}"/}")
                done
                if [ ${#EXPS[@]} -eq 0 ]; then
                    echo "No hay benchmarks en benchmarks/<CB|MB>Kernels/${CARPETA}/ (run_*.py)."
                    exit 0
                fi
            fi
        elif [[ "$1" != run_* ]] && [[ "$1" != *.py ]] && [[ "$1" != */* ]] \
                && mapfile -t EXPS < <(benchmarks_de_alg "$1" '*') && [ ${#EXPS[@]} -gt 0 ]; then
            shift
        else
            EXPS=("$(resolver_exp "$1")")
            shift
        fi
        # DSL de cada benchmark: el fijado (variable DSL o 'exp <dsl>') o el de su carpeta.
        DSL_FIJO="${DSL}"
        cabecera "Experimentos: ${EXPS[*]} $*"
        comprobar_docker
        estado_gpu
        fallos=()
        validados=()
        malos=()
        for EXP in "${EXPS[@]}"; do
            usar_dsl "${DSL_FIJO:-$(dsl_de_exp "${EXP}")}"
            export TFM_EXPERIMENTO="$(basename "${EXP}" .py)"
            # Imagen y validacion una sola vez por DSL; si falla, se saltan sus benchmarks.
            if [[ " ${validados[*]} ${malos[*]} " != *" ${DSL} "* ]]; then
                echo "== DSL ${DSL} | imagen ${IMAGE} =="
                if asegurar_imagen && { [ "${VALIDAR}" != "1" ] || validar; }; then
                    validados+=("${DSL}")
                else
                    echo "ERROR: imagen/validacion de ${DSL} fallida; se saltan sus benchmarks."
                    malos+=("${DSL}")
                fi
            fi
            if [[ " ${malos[*]} " == *" ${DSL} "* ]]; then
                fallos+=("${TFM_EXPERIMENTO}")
                continue
            fi
            echo "== Ejecutando ${EXP} [${DSL}] $* =="
            t0=$(date +%s)
            if en_contenedor python "/workspace/tfm/${EXP}" "$@"; then
                echo "== ${TFM_EXPERIMENTO} terminado en $(( $(date +%s) - t0 )) s =="
            else
                echo "== ERROR: ${TFM_EXPERIMENTO} ha fallado ($(( $(date +%s) - t0 )) s) =="
                fallos+=("${TFM_EXPERIMENTO}")
            fi
        done
        if [ ${#fallos[@]} -gt 0 ]; then
            echo "ERROR: han fallado: ${fallos[*]}"
            exit 1
        fi
        ;;
    listar|experimentos|exps)
        listar_experimentos
        exit 0
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
