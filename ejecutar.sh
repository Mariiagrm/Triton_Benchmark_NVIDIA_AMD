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
# Dos nodos (GB10 unidos por el cable QSFP, RoCE): 'encolar dist' copia el codigo al otro
# nodo (rsync, sin results/ ni logs/) y pide ambos en la cola hennessy-patterson (exclusiva;
# con COLA=benchmark|test, un trabajo heterogeneo con la cola de cada nodo). El
# nodo del repo es el rank 0 y guarda los resultados en results/<nodo0>-<nodo1>/:
#   bash ~/hennessy/tfm_entorno/ejecutar.sh encolar dist run_dist_matmul [--sizes ...]
#   NODO_PAR=patterson COLA=test bash ~/hennessy/tfm_entorno/ejecutar.sh encolar dist ...
#   (la primera vez, REBUILD=1 delante: la imagen del otro nodo debe salir de este repo)
#
# TODO en las dos GPUs con UN solo trabajo: cada nodo ejecuta 'todo' a la vez (sus resultados en
# results/<su nodo>/) y al final run_dist_matmul en los dos. Al terminar, 'traer' (desde el
# login) copia los resultados del otro nodo a este repo y regenera las tablas:
#   bash ~/hennessy/tfm_entorno/ejecutar.sh encolar ambos todo
#   bash ~/hennessy/tfm_entorno/ejecutar.sh traer
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
#   dist <exp> [args] (dentro de un trabajo heterogeneo de 2 nodos, lo pone 'encolar dist')
#                     ejecuta el benchmark con torchrun, un rank por nodo, NCCL por la red QSFP
#   todo              en este nodo: todos los benchmarks de cada DSL (con su validacion),
#                     run_pico_mma y run_sostenido; sigue aunque falle alguno
#   ambos <cmd...>    (dentro de un trabajo heterogeneo, lo pone 'encolar ambos') '<cmd...>' a la
#                     vez en los dos nodos; 'ambos todo' anade al final 'dist run_dist_matmul'
#   traer             (desde el login) copia results/<otro nodo>*/ del otro nodo y regenera
#                     results/*_metrics.csv y el resumen
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
#   NODO_PAR=...      'encolar dist|ambos', 'traer': el otro nodo (por defecto hennessy<->patterson)
#   PARTICION_DOS=... 'encolar dist|ambos': cola con los dos nodos (por defecto <nodo>-<otro>)
#   COLA=...          'encolar dist|ambos': benchmark | test -> heterogeneo con <nodo>-<COLA>
#   RED_DIST=172.16.  'dist': prefijo de IP de la red entre nodos (interfaz del cable QSFP)
#   NCCL_IB_HCA=...   'dist': dispositivos RoCE para NCCL (por defecto los detecta NCCL)
#   NCCL_DEBUG=...    'dist': por defecto INFO (subsistemas INIT,NET: muestra si usa RoCE)
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
nodo_del_repo() {
    real="$(realpath "${RAIZ}")"
    if [[ "${real}" =~ ^/machines/([^/]+)(/.*)$ ]]; then
        nodo="${BASH_REMATCH[1]}"
        ruta="${BASH_REMATCH[2]}"
    else
        nodo="$(hostname -s)"
        ruta="${real}"
    fi
}

encolar() {
    [ $# -ge 1 ] || uso
    local real nodo ruta
    nodo_del_repo
    if [ "$1" = "dist" ] || [ "$1" = "ambos" ]; then
        encolar_dos "${nodo}" "${ruta}" "${real}" "$@"
        return
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

# El otro nodo del par (NODO_PAR o hennessy<->patterson). Los nodos no se ven entre si: copiar
# codigo y resultados solo se puede desde el login, que ve ambos en /machines/<nodo>/.
nodo_par() {
    local otro="${NODO_PAR:-}"
    if [ -z "${otro}" ]; then
        case "$1" in hennessy) otro=patterson ;; patterson) otro=hennessy ;;
            *) echo "ERROR: indica el otro nodo con NODO_PAR=<nodo>." >&2; exit 1 ;; esac
    fi
    if [ ! -d "/machines/${otro}${2%/*}" ] || [[ "$3" != /machines/* ]]; then
        echo "ERROR: hay que lanzarlo desde el login, con el repo en /machines/<nodo>/." >&2
        exit 1
    fi
    echo "${otro}"
}

# 'traer': results/<otro>*/ del repo del otro nodo -> este repo, y regenera las tablas consolidadas.
traer() {
    local real nodo ruta otro origen d
    nodo_del_repo
    otro="$(nodo_par "${nodo}" "${ruta}" "${real}")"
    origen="/machines/${otro}${ruta}/results"
    shopt -s nullglob
    local dirs=("${origen}/${otro}"*/)
    shopt -u nullglob
    if [ ${#dirs[@]} -eq 0 ]; then
        echo "== No hay resultados de ${otro} que traer (${origen}/${otro}*/) =="
        return
    fi
    for d in "${dirs[@]}"; do
        echo "== Trayendo ${otro}:${ruta}/results/$(basename "${d}")/ =="
        rsync -a "${d}" "${real}/results/$(basename "${d}")/"
    done
    echo "== Regenerando results/*_metrics.csv y el resumen =="
    (cd "${real}/benchmarks" && python3.11 -c '
import informe, resumen
for fam in sorted({informe.familia(e) for _, e, _ in informe.ultimas_ejecuciones()} - {None}):
    informe.actualizar_metricas(fam)
resumen.generar(mostrar="todas")')
}

# 'encolar dist|ambos': copia el codigo al otro nodo y pide los dos en un trabajo heterogeneo
# (componente 0 = nodo del repo = rank 0; componente 1 = el otro nodo).
encolar_dos() {
    local nodo="$1" ruta="$2" real="$3"; shift 3
    [ -n "${2:-}" ] || uso
    local otro
    otro="$(nodo_par "${nodo}" "${ruta}" "${real}")"
    traer   # resultados de un trabajo anterior en el otro nodo, por si no se trajeron
    echo "== Copiando el codigo a ${otro}:${ruta} (sin results/, logs/, .cache/, .git/) =="
    mkdir -p "/machines/${otro}${ruta}"
    rsync -a --delete --exclude /results/ --exclude /logs/ --exclude /.cache/ --exclude /.git/ \
        "${real}/" "/machines/${otro}${ruta}/"
    local nombre="$1-$(basename "$2" .py)" comun=() part p
    comun=(-J "${nombre}" --chdir="${ruta}" -o "${ruta}/logs/%x-%j.out" -e "${ruta}/logs/%x-%j.out"
           --export=ALL,TFM_NODO0="${nodo}")
    # Cola con los dos nodos (hennessy-patterson, exclusiva) o, con COLA=benchmark|test o si no
    # existe, un trabajo heterogeneo con las colas de cada nodo.
    part="${PARTICION_DOS:-}"
    if [ -z "${part}" ] && [ -z "${COLA:-}" ]; then
        for p in "${nodo}-${otro}" "${otro}-${nodo}"; do
            [ -n "$(sinfo -h -p "${p}" -o %P 2>/dev/null)" ] && { part="${p}"; break; }
        done
    fi
    if [ -n "${part}" ]; then
        echo "== Encolando '$*' en ${part} (2 nodos; repo en el nodo: ${ruta}) =="
        sbatch "${comun[@]}" -p "${part}" -N 2 --ntasks-per-node=1 --cpus-per-task=14 \
            "${real}/ejecutar.sh" "$@"
    else
        local cola="${COLA:-benchmark}" cpus=14
        [ "${cola}" = "test" ] && cpus=2   # MaxCPUsPerNode=2 en las colas -test
        echo "== Encolando '$*' en ${nodo}-${cola} + ${otro}-${cola} (heterogeneo; repo en el nodo: ${ruta}) =="
        sbatch "${comun[@]}" -p "${nodo}-${cola}" --cpus-per-task="${cpus}" \
            : -p "${otro}-${cola}" --cpus-per-task="${cpus}" "${real}/ejecutar.sh" "$@"
    fi
    echo "Log: logs/${nombre}-<JOBID>.out (en ${nodo})   Cola: squeue -u $(id -un)"
}

# Nodos de un trabajo de 2 nodos (cola hennessy-patterson o heterogeneo): NODOS=(rank0 rank1),
# rank 0 = nodo del repo (TFM_NODO0, lo pone 'encolar').
nodos_del_trabajo() {
    local n0 n
    if [ "${SLURM_HET_SIZE:-1}" -ge 2 ]; then
        NODOS=("${SLURM_JOB_NODELIST_HET_GROUP_0}" "${SLURM_JOB_NODELIST_HET_GROUP_1}")
    elif [ "${SLURM_JOB_NUM_NODES:-1}" -ge 2 ]; then
        n0="${TFM_NODO0:-$(hostname -s)}"
        NODOS=("${n0}")
        for n in $(scontrol show hostnames "${SLURM_JOB_NODELIST}"); do
            [ "${n}" != "${n0}" ] && NODOS+=("${n}")
        done
    else
        echo "ERROR: hace falta un trabajo de 2 nodos: usa 'ejecutar.sh encolar dist|ambos ...'." >&2
        exit 1
    fi
}

# srun en el nodo del rank $1 (NODOS de nodos_del_trabajo).
srun_en() {
    local r="$1"; shift
    if [ "${SLURM_HET_SIZE:-1}" -ge 2 ]; then
        srun --het-group="${r}" "$@"
    else
        srun -N1 -n1 -w "${NODOS[r]}" "$@"
    fi
}

# En un trabajo de 2 nodos: 'ejecutar.sh <args>' a la vez en ambos (un srun por
# componente), salida con el nodo delante; '@R' en los args pasa a ser el rank (0 | 1).
# Con --parar, si uno falla se para el otro (p. ej. un rank de torchrun esperaria al otro).
en_los_dos() {
    local parar=0 pids=() r p fallo=0
    [ "$1" = "--parar" ] && { parar=1; shift; }
    nodos_del_trabajo
    for r in 0 1; do
        ( srun_en "${r}" bash "${RAIZ}/ejecutar.sh" "${@//@R/${r}}" 2>&1 \
            | sed -u "s/^/[${NODOS[r]}] /" ) &
        pids+=($!)
    done
    if [ "${parar}" = 1 ] && ! wait -n; then
        fallo=1
        echo "== ERROR: un nodo ha fallado; se para el otro =="
        pkill -TERM -P "$(IFS=,; echo "${pids[*]}")" 2>/dev/null || true   # los srun de cada nodo
    fi
    for p in "${pids[@]}"; do wait "${p}" || fallo=1; done
    return "${fallo}"
}

# Interfaz e IP de la red entre nodos (cable QSFP): la que tiene una IPv4 que empieza por RED_DIST.
iface_dist() { ip -o -4 addr show | awk -v red="${RED_DIST:-172.16.}" 'index($4, red) == 1 {print $2; exit}'; }
ip_dist() { ip -o -4 addr show | awk -v red="${RED_DIST:-172.16.}" 'index($4, red) == 1 {sub("/.*", "", $4); print $4; exit}'; }

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
    if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
        construir_imagen
    elif [ "${REBUILD}" = "1" ] && ! construir_imagen; then
        # P. ej. sin acceso a nvcr.io desde el nodo: mejor medir con la imagen que hay que no medir.
        echo "AVISO: no se pudo reconstruir ${IMAGE}; se usa la existente" \
             "($(docker image inspect -f '{{.Created}}' "${IMAGE}"))."
    fi
}

# Contenedores de trabajos anteriores que siguen vivos: si Slurm mata un trabajo (p. ej. NODE_FAIL
# del otro nodo y reencolado), muere el cliente 'docker run' pero NO el contenedor, que sigue
# usando la GPU y falsea las medidas del siguiente intento (job 21058: matmul a ~41 TFLOP/s
# en vez de ~92; descartados/LEEME.md). Cada contenedor lleva la etiqueta tfm.job=<job>.<intento>;
# se paran los que no son de este intento y cuyo trabajo ya no corre (o es un intento anterior).
ETIQUETA_JOB="${SLURM_JOB_ID:-local}.${SLURM_RESTART_COUNT:-0}"
limpiar_huerfanos() {
    local c etiqueta job
    command -v squeue >/dev/null 2>&1 || return 0
    for c in $(docker ps -q --filter label=tfm.job 2>/dev/null); do
        etiqueta="$(docker inspect -f '{{index .Config.Labels "tfm.job"}}' "${c}")"
        job="${etiqueta%%.*}"
        [ "${etiqueta}" = "${ETIQUETA_JOB}" ] && continue
        [ "${job}" = "local" ] && continue
        if [ "${job}" = "${SLURM_JOB_ID:-}" ] || [ -z "$(squeue -h -j "${job}" -t R 2>/dev/null)" ]; then
            echo "AVISO: parando contenedor huerfano ${c} (tfm.job=${etiqueta}; este es ${ETIQUETA_JOB})"
            docker rm -f "${c}" >/dev/null || true
        fi
    done
}

# La GPU debe estar libre para que las medidas sean fiables.
estado_gpu() {
    limpiar_huerfanos
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
    # 'dist': red del host (NCCL ve el ConnectX-7), RDMA (/dev/infiniband, memoria bloqueable).
    local red=()
    if [ "${TFM_DIST:-0}" = "1" ]; then
        red=(--network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864
             -e NCCL_SOCKET_IFNAME="${TFM_IFACE}" -e GLOO_SOCKET_IFNAME="${TFM_IFACE}"
             -e NCCL_DEBUG="${NCCL_DEBUG:-INFO}" -e NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}")
        [ -d /dev/infiniband ] && red+=(--device /dev/infiniband)
        [ -n "${NCCL_IB_HCA:-}" ] && red+=(-e NCCL_IB_HCA="${NCCL_IB_HCA}")
    fi
    docker run --rm --gpus all ${tty[@]+"${tty[@]}"} ${red[@]+"${red[@]}"} --label tfm.job="${ETIQUETA_JOB}" \
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
    dist)
        # En un trabajo de 2 nodos: un 'srun' por nodo, cada uno lanza '_rango <r>' (contenedor +
        # torchrun). Si un rank falla, se para el otro.
        [ $# -ge 1 ] || uso
        EXP="$(resolver_exp "$1")"
        shift
        usar_dsl "${DSL:-$(dsl_de_exp "${EXP}")}"
        nodos_del_trabajo
        # IP del rank 0 en la red QSFP (el script puede correr en el otro nodo del trabajo).
        if [ "${NODOS[0]}" = "$(hostname -s)" ]; then
            TFM_MASTER="$(ip_dist)"
        else
            TFM_MASTER="$(srun_en 0 bash "${RAIZ}/ejecutar.sh" _ip)"
        fi
        if [ -z "${TFM_MASTER}" ]; then
            echo "ERROR: ${NODOS[0]} no tiene IP en la red entre nodos (RED_DIST=${RED_DIST:-172.16.})." >&2
            exit 1
        fi
        export TFM_MASTER TFM_PUERTO="$((29500 + SLURM_JOB_ID % 1000))"
        export TFM_MAQUINA="${TFM_MAQUINA:-${NODOS[0]}-${NODOS[1]}}"
        cabecera "Experimento distribuido: ${EXP} $* | nodos ${NODOS[*]}"
        en_los_dos --parar _rango @R "${EXP}" "$@" || { echo "ERROR: ${EXP} distribuido ha fallado"; exit 1; }
        ;;
    todo)
        # Cada 'exp' valida su imagen antes de medir. run_pico_mma y run_sostenido usan la imagen
        # triton-tlx, ya construida (REBUILD=0).
        cabecera "Todo en $(hostname -s)"
        fallos=()
        for x in triton-tlx gluon helion cutlass run_pico_mma run_sostenido; do
            reb="${REBUILD}"
            [[ "${x}" == run_* ]] && reb=0
            REBUILD="${reb}" bash "${RAIZ}/ejecutar.sh" exp "${x}" || fallos+=("${x}")
        done
        if [ ${#fallos[@]} -gt 0 ]; then
            echo "ERROR: en $(hostname -s) han fallado: ${fallos[*]}"
            exit 1
        fi
        ;;
    ambos)
        [ $# -ge 1 ] || uso
        nodos_del_trabajo
        cabecera "En los dos nodos: $* | ${NODOS[*]}"
        fallo=0
        en_los_dos "$@" || fallo=1
        if [ "$1" = "todo" ]; then
            REBUILD=0 bash "${RAIZ}/ejecutar.sh" dist run_dist_matmul || fallo=1
        fi
        echo "== Trae los resultados de ${NODOS[1]} desde el login:" \
             "bash ~/${NODOS[0]}/$(basename "${RAIZ}")/ejecutar.sh traer =="
        [ "${fallo}" = 0 ] || { echo "ERROR: ha fallado algo; busca 'ERROR' en el log"; exit 1; }
        ;;
    traer)
        traer
        exit 0
        ;;
    _ip)
        ip_dist
        exit 0
        ;;
    _rango)
        # Interno ('dist'): rank $1 en este nodo.
        RANGO="$1" EXP="$2"
        shift 2
        usar_dsl "${DSL:-$(dsl_de_exp "${EXP}")}"
        export TFM_DIST=1 TFM_EXPERIMENTO="$(basename "${EXP}" .py)" TFM_IFACE="$(iface_dist)"
        if [ -z "${TFM_IFACE}" ]; then
            echo "ERROR: $(hostname -s) no tiene interfaz en la red entre nodos (RED_DIST=${RED_DIST:-172.16.})." >&2
            exit 1
        fi
        echo "== Rank ${RANGO} en $(hostname -s): interfaz ${TFM_IFACE} ($(ip_dist)), master ${TFM_MASTER}:${TFM_PUERTO} =="
        comprobar_docker
        estado_gpu
        asegurar_imagen
        en_contenedor python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank="${RANGO}" \
            --master_addr="${TFM_MASTER}" --master_port="${TFM_PUERTO}" "/workspace/tfm/${EXP}" "$@"
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
