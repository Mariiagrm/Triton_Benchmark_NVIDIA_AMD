"""Utilidades comunes a todos los experimentos del TFM: contexto, medida y guardado.

Estandar: toda operacion matricial se ejecuta en TENSOR CORES, no en CUDA cores.
    - Al importar este modulo, fp32 pasa a TF32 en cuBLAS/cuDNN (PyTorch lo
      ejecuta en CUDA cores por defecto).
    - exigir_tensor_cores(kernel_compilado) aborta si un kernel Triton no
      contiene instrucciones MMA de tensor core en su PTX.

Todo experimento deberia:
    1. medir con medir()   -> misma metodologia (triton.testing.do_bench, mediana), con
                              el reloj SM y la potencia de la GPU durante la medida
                              (reloj_mhz, potencia_w): el pico de los tensor cores es
                              proporcional al reloj (docs/TFM/pico_mma.md)
    2. guardar con guardar() -> results/<maquina>/<experimento>/<fecha>_job<JOBID>/
                                  resultados.csv  (una fila por medida)
                                  meta.json       (contexto + parametros + resumen)
       y results/<maquina>/<experimento>/ultimo -> la mas reciente; ademas regenera
       results/<familia>_metrics.csv (p. ej. matmul_metrics.csv) con la ultima de cada una.

El contexto (GPU, versiones, imagen, job de Slurm) lo rellenan las variables de
entorno que pasa ejecutar.sh, asi cada resultado es reproducible.
"""
import csv
import datetime
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

import torch
import triton

RAIZ = os.environ.get("TFM_RAIZ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Entorno de ejecucion: separa los resultados por maquina (results/<maquina>/...). Por defecto
# el nodo; TFM_MAQUINA permite distinguir entornos distintos en el mismo nodo.
MAQUINA = os.environ.get("TFM_MAQUINA") or os.environ.get("TFM_HOST") or platform.node().split(".")[0]
# tf32: tensores fp32 que los tensor cores multiplican en TF32 (tl.dot y cuBLAS).
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "tf32": torch.float32}

# Sin esto, torch.matmul en fp32 usa CUDA cores y la comparacion con cuBLAS no seria justa.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Instrucciones MMA de tensor core en PTX: mma.sync (sm_80+, la que usa sm_121/GB10),
# wgmma (sm_90a) y tcgen05.mma (sm_100a).
_MMA = re.compile(r"\b(?:tcgen05\.mma|wgmma\.mma_async|mma\.sync)[\w.]*")
# Instrucciones de copia asincrona en PTX:
#   cp.async.bulk.tensor -> TMA (Tensor Memory Accelerator, sm_90+), bloques VRAM->SRAM.
#   cp.async            -> copia asincrona "clasica" (sm_80+), el pipelining sin TMA.
_TMA = re.compile(r"\bcp\.async\.bulk\.tensor[\w.]*")
_CP_ASYNC = re.compile(r"\bcp\.async(?!\.bulk)[\w.]*")


def version(paquete):
    try:
        return md.version(paquete)
    except md.PackageNotFoundError:
        return None


def contexto():
    """Metadatos de la ejecucion: hardware, software e identificacion del job."""
    cap = torch.cuda.get_device_capability(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "sm": f"{cap[0]}{cap[1]}",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "helion": version("helion"),
        "triton_utlx": version("triton-utlx"),
        "python": platform.python_version(),
        "cublas_tf32": torch.backends.cuda.matmul.allow_tf32,
        "dsl": os.environ.get("TFM_DSL"),
        "imagen": os.environ.get("TFM_IMAGEN"),
        "maquina": MAQUINA,
        "host": os.environ.get("TFM_HOST", platform.node()),
        # Del nodo, no del contenedor: el driver y el kernel pueden cambiar entre ejecuciones.
        "driver_nvidia": os.environ.get("TFM_DRIVER") or None,
        "kernel_linux": os.environ.get("TFM_KERNEL") or None,
        "slurm_job": os.environ.get("SLURM_JOB_ID") or None,
        # True: caches de compilacion vacias (TFM_CACHE_FRIA=1); los tiempos de 1a llamada
        # (t_compilacion_s, autotune_s) son entonces compilacion + autotuning en frio.
        "cache_fria": os.environ.get("TFM_CACHE_FRIA") == "1",
        "potencia_fuente": Muestreador().fuente_potencia,
        # True: medir() anade el regimen sostenido y la eficiencia energetica (TFM_ENERGIA=1).
        "energia": ENERGIA,
        "energia_segundos": ENERGIA_SEGUNDOS if ENERGIA else None,
        "energia_descarte": ENERGIA_DESCARTE if ENERGIA else None,
        "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
        "comando": " ".join(sys.argv),
    }


def imprimir_contexto():
    c = contexto()
    print(f"GPU: {c['gpu']} (sm_{c['sm']}) | torch {c['torch']} | triton {c['triton']} | "
          f"helion {c['helion']} | job {c['slurm_job'] or 'local'}\n", flush=True)


# Reloj y potencia durante cada medida: el pico de los tensor cores es proporcional al reloj SM
# (docs/TFM/pico_mma.md), asi que sin el no se puede saber si un kernel pierde rendimiento por
# el propio kernel o porque la GPU baja la frecuencia (p. ej. por limite de potencia en GB10).
class Muestreador:
    """Muestrea reloj SM (MHz) y potencia (W) en segundo plano: NVML o, si no, nvidia-smi.

    Potencia: la INSTANTANEA de NVML (NVML_FI_DEV_POWER_INSTANT) si el driver la da; si no, la
    de nvmlDeviceGetPowerUsage, que es un promedio con retardo (en GB10 una medida de ~0.5 s
    puede reflejar aun la anterior: cuBLAS a 95 TFLOP/s leyo 10.8 W, sonda job 20844). La
    fuente usada queda en self.fuente_potencia ('instantanea' | 'promedio' | 'nvidia-smi').
    """

    def __init__(self, periodo=0.02):
        self.periodo, self.relojes, self.potencias = periodo, [], []
        self.serie = []  # (t desde el inicio en s, reloj MHz, potencia W)
        self._parar = threading.Event()
        self._nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device()))
            self.fuente_potencia = "instantanea" if self._potencia_instantanea() is not None else "promedio"
        except Exception:
            self.fuente_potencia = "nvidia-smi"
            self.periodo = max(periodo, 0.1)
            if not shutil.which("nvidia-smi"):
                self.periodo = None  # sin forma de muestrear

    def _potencia_instantanea(self):
        nv, h = self._nvml
        campo = getattr(nv, "NVML_FI_DEV_POWER_INSTANT", 186)
        try:
            v = nv.nvmlDeviceGetFieldValues(h, [campo])[0]
            if v.nvmlReturn == 0 and v.value.uiVal:
                return v.value.uiVal / 1000
        except Exception:
            pass
        return None

    def _leer(self):
        if self._nvml:
            nv, h = self._nvml
            reloj = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
            potencia = self._potencia_instantanea() if self.fuente_potencia == "instantanea" else None
            if potencia is None:
                try:
                    potencia = nv.nvmlDeviceGetPowerUsage(h) / 1000
                except Exception:
                    potencia = None
            return reloj, potencia
        s = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.split(",")
        num = lambda x: float(x) if x.strip().replace(".", "", 1).isdigit() else None  # noqa: E731
        return num(s[0]), num(s[1]) if len(s) > 1 else None

    def _bucle(self):
        t0 = time.perf_counter()
        while not self._parar.is_set():
            reloj, potencia = self._leer()
            self.serie.append((time.perf_counter() - t0, reloj, potencia))
            if reloj:
                self.relojes.append(reloj)
            if potencia:
                self.potencias.append(potencia)
            time.sleep(self.periodo)

    def __enter__(self):
        if self.periodo:
            self._hilo = threading.Thread(target=self._bucle, daemon=True)
            self._hilo.start()
        return self

    def __exit__(self, *exc):
        if self.periodo:
            self._parar.set()
            self._hilo.join()

    def mediana(self, desde=0.0):
        """Medianas de reloj y potencia, opcionalmente solo de las muestras con t >= desde (s)."""
        med = lambda xs: round(statistics.median(xs), 1) if xs else None  # noqa: E731
        if desde:
            tramo = [m for m in self.serie if m[0] >= desde]
            return med([m[1] for m in tramo if m[1]]), med([m[2] for m in tramo if m[2]])
        return med(self.relojes), med(self.potencias)

    def reloj_maximo(self):
        if self._nvml:
            nv, h = self._nvml
            return nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM)
        return None


def medir(fn, flops=None, bytes_movidos=None, warmup=100, rep=500):
    """Mide fn con do_bench. Devuelve ms (mediana, p20, p80), la mediana del reloj SM (MHz) y de la
    potencia (W) durante la medida y, si se dan, TFLOP/s y GB/s."""
    with Muestreador() as m:
        ms, p20, p80 = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
    reloj, potencia = m.mediana()
    r = {"ms": round(ms, 4), "ms_p20": round(p20, 4), "ms_p80": round(p80, 4),
         "reloj_mhz": reloj, "potencia_w": potencia}
    if flops is not None:
        r["tflops"] = round(flops / (ms * 1e-3) / 1e12, 2)
    if bytes_movidos is not None:
        r["gbs"] = round(bytes_movidos / (ms * 1e-3) / 1e9, 1)
    if ENERGIA:
        r.update(energia(fn, flops=flops, bytes_movidos=bytes_movidos))
    return r


# Eficiencia energetica (TFM_ENERGIA=1, lo pasa ejecutar.sh): medir() ejecuta ademas cada kernel
# en regimen sostenido (do_bench dura ~0.6 s y NVML solo actualiza cada ~0.5 s: la potencia de
# esa ventana es la del transitorio; docs/TFM/pico_mma.md) y anota la energia por llamada y el
# trabajo por julio. Coste por medida: ENERGIA_REPOSO + ENERGIA_SEGUNDOS.
ENERGIA = os.environ.get("TFM_ENERGIA") == "1"
ENERGIA_SEGUNDOS = float(os.environ.get("TFM_ENERGIA_SEGUNDOS") or 6)
ENERGIA_DESCARTE = float(os.environ.get("TFM_ENERGIA_DESCARTE") or 2)
ENERGIA_REPOSO = 3.0  # s de reposo antes de cada kernel: todos parten del mismo estado termico


def sostenido(fn, segundos, descarte):
    """Ejecuta fn en bucle `segundos` y mide solo tras `descarte` s.
    Devuelve (llamadas/s, reloj SM en MHz, potencia en W) del tramo medido."""
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    lote = max(1, int(0.05 / max(time.perf_counter() - t, 1e-6)))  # lotes de ~50 ms
    with Muestreador() as m:
        t0 = time.perf_counter()
        n, t_ini = 0, None
        while True:
            for _ in range(lote):
                fn()
            torch.cuda.synchronize()
            ahora = time.perf_counter() - t0
            if t_ini is None and ahora >= descarte:
                t_ini, n = ahora, 0
            elif t_ini is not None:
                n += lote
            if ahora >= segundos:
                break
    reloj, potencia = m.mediana(desde=t_ini)
    return n / (ahora - t_ini), reloj, potencia


_potencia_reposo = None


def potencia_reposo(segundos=4.0):
    """Potencia de la GPU sin carga (W): mediana de los ultimos 2 s. Una vez por proceso."""
    global _potencia_reposo
    if _potencia_reposo is None:
        torch.cuda.synchronize()
        with Muestreador() as m:
            time.sleep(segundos)
        _potencia_reposo = m.mediana(desde=segundos - 2)[1]
    return _potencia_reposo


def energia(fn, flops=None, bytes_movidos=None):
    """Regimen sostenido de fn: rendimiento, reloj, potencia y eficiencia energetica.

    Eficiencia = trabajo / energia de la GPU: GFLOP/J en los compute-bound (con flops) y GB/J en
    los memory-bound (solo bytes_movidos). La version *_din descuenta la potencia en reposo: la
    energia que cuesta el kernel por encima de tener la GPU encendida.
    """
    p0 = potencia_reposo()
    torch.cuda.synchronize()
    time.sleep(ENERGIA_REPOSO)
    ritmo, reloj, p = sostenido(fn, ENERGIA_SEGUNDOS, ENERGIA_DESCARTE)
    r = {"reloj_sost_mhz": reloj, "potencia_sost_w": p, "potencia_reposo_w": p0,
         "mj_llamada": round(1e3 * p / ritmo, 4) if p else None}
    din = p - p0 if p and p0 and p > p0 else None
    if flops is not None:
        tf = flops * ritmo / 1e12
        r["tflops_sost"] = round(tf, 2)
        r["gflop_j"] = round(1e3 * tf / p, 1) if p else None
        r["gflop_j_din"] = round(1e3 * tf / din, 1) if din else None
    elif bytes_movidos is not None:
        gb = bytes_movidos * ritmo / 1e9
        r["gbs_sost"] = round(gb, 1)
        r["gb_j"] = round(gb / p, 2) if p else None
        r["gb_j_din"] = round(gb / din, 2) if din else None
    return r


def instrucciones_tensor_core(compilado):
    """Instrucciones MMA de tensor core (sin repetir) en el PTX de un kernel Triton compilado."""
    asm = getattr(compilado, "asm", None)
    if asm is not None and "ptx" not in asm and "cubin" in asm:
        # Sin PTX (TRITON_STORE_BINARY_ONLY=1, p. ej. kernels autotuneados por Helion): se mira el
        # SASS del cubin (asm["sass"] lo genera Triton con nvdisasm), donde la MMA es HMMA.
        return sorted(set(re.findall(r"\bHMMA[\w.]*", asm["sass"])))
    if asm is None or "ptx" not in asm:
        raise TypeError(f"se esperaba un kernel Triton compilado (con .asm['ptx']), no {type(compilado).__name__}")
    return instrucciones_tensor_core_ptx(asm["ptx"])


def instrucciones_tensor_core_ptx(ptx):
    """Instrucciones MMA de tensor core (sin repetir) en un texto PTX (p. ej. el __ptx__ de CuTe DSL)."""
    return sorted(set(_MMA.findall(ptx)))


def kernels_triton_compilados(jit_fn):
    """Kernels ya compilados (CompiledKernel, con .asm['ptx']) de una funcion @triton.jit o de un
    @triton.autotune (se usa su .fn). Sirve cuando el codigo oficial lanza el kernel sin
    devolverlo, p. ej. el tutorial de atencion: JITFunction.device_caches[dev][0] es la cache
    {clave: CompiledKernel} de triton/runtime/jit.py (v3.8.0)."""
    fn = jit_fn if hasattr(jit_fn, "device_caches") else jit_fn.fn  # Autotuner -> su JITFunction
    return [k for caches in fn.device_caches.values() for k in caches[0].values()]


def exigir_tensor_cores(compilado, nombre="kernel"):
    """Aborta si el kernel no usa tensor cores. Devuelve sus instrucciones MMA."""
    mma = instrucciones_tensor_core(compilado)
    if not mma:
        raise RuntimeError(f"{nombre} NO usa tensor cores: su PTX no contiene mma.sync/wgmma/tcgen05.mma "
                           f"(se ejecutaria en CUDA cores)")
    return mma


def _ptx(compilado):
    asm = getattr(compilado, "asm", None)
    if asm is None or "ptx" not in asm:
        raise TypeError(f"se esperaba un kernel Triton compilado (con .asm['ptx']), no {type(compilado).__name__}")
    return asm["ptx"]


def instrucciones_tma(compilado):
    """Instrucciones TMA (cp.async.bulk.tensor) en el PTX de un kernel compilado."""
    return sorted(set(_TMA.findall(_ptx(compilado))))


def usa_tma(compilado):
    """True si el kernel emite TMA. Util para comprobar que la variante block-ptr
    aprovecha el acelerador en esta GPU (en sm_121/GB10 hay que verificarlo)."""
    return bool(instrucciones_tma(compilado))


def copias_asincronas(compilado):
    """Resumen de la estrategia de carga del kernel: TMA, cp.async clasico o sincrona.
    Devuelve un dict con las instrucciones detectadas y una etiqueta legible."""
    ptx = _ptx(compilado)
    tma = sorted(set(_TMA.findall(ptx)))
    cp = sorted(set(_CP_ASYNC.findall(ptx)))
    if tma:
        etiqueta = "TMA"
    elif cp:
        etiqueta = "cp.async"
    else:
        etiqueta = "sincrona"
    return {"carga": etiqueta, "tma": tma, "cp_async": cp}


def kernels_cuda(fn):
    """Nombres de los kernels CUDA que lanza fn, p. ej. el GEMM que elige cuBLAS (evidencia para la memoria)."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    fn()  # calentamiento: que cuBLAS elija su heuristica fuera del perfilado
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if e.device_type == DeviceType.CUDA})


def nombre_experimento():
    """Nombre del experimento: el que da ejecutar.sh o, si no, el del script principal."""
    return os.environ.get("TFM_EXPERIMENTO") or os.path.splitext(os.path.basename(sys.argv[0]))[0]


def guardar(filas, parametros=None, resumen=None, nombre=None, informe=None):
    """Guarda filas (lista de dicts) en CSV y el contexto en JSON. Devuelve el directorio.

    Al terminar genera tabla + grafica automaticamente: usa `informe` si se pasa
    (callable que recibe el directorio de la ejecucion) o, si no, el informe generico
    (informe.generar). Un fallo del informe nunca hace perder las medidas ya guardadas.
    """
    nombre = nombre or nombre_experimento()
    base = os.path.join(RAIZ, "results", MAQUINA, nombre)
    job = os.environ.get("SLURM_JOB_ID")
    ejecucion = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_job{job}" if job else "")
    destino = os.path.join(base, ejecucion)
    os.makedirs(destino, exist_ok=True)

    if filas:
        columnas = list(dict.fromkeys(k for f in filas for k in f))  # union, en orden de aparicion
        with open(os.path.join(destino, "resultados.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columnas)
            w.writeheader()
            w.writerows(filas)
    with open(os.path.join(destino, "meta.json"), "w") as f:
        json.dump({"experimento": nombre, "contexto": contexto(), "parametros": parametros or {},
                   "resumen": resumen or {}}, f, indent=2, default=str)

    # Enlace 'ultimo' -> ejecucion mas reciente (reemplazo atomico).
    ultimo = os.path.join(base, "ultimo")
    tmp = ultimo + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(ejecucion, tmp)
    os.replace(tmp, ultimo)

    print(f"Resultados guardados en {os.path.relpath(destino, RAIZ)}/", flush=True)

    # Informe automatico (tabla + grafica). No debe hacer perder las medidas.
    try:
        if informe is None:
            import informe as _informe
            informe = _informe.generar
        informe(destino)
    except Exception as e:
        print(f"AVISO: no se pudo generar el informe ({type(e).__name__}: {e}). "
              f"Regeneralo con: python3.11 benchmarks/informe.py {os.path.relpath(destino, RAIZ)}", flush=True)

    # Tabla consolidada results/<familia>_metrics.csv (p. ej. matmul_metrics.csv).
    try:
        import informe as _informe
        if _informe.familia(nombre):
            _informe.actualizar_metricas(_informe.familia(nombre))
    except Exception as e:
        print(f"AVISO: no se pudo actualizar la tabla de metricas ({type(e).__name__}: {e}).", flush=True)

    # Resumen por arquitectura (filas = DSL, columnas = algoritmo): docs/TFM/resultados/resumen*.
    try:
        import resumen as _resumen
        _resumen.generar(mostrar=destino)
    except Exception as e:
        print(f"AVISO: no se pudo actualizar el resumen ({type(e).__name__}: {e}). "
              f"Regeneralo con: python3.11 benchmarks/resumen.py", flush=True)
    # Eficiencia energetica (TFM_ENERGIA=1): docs/TFM/resultados/energia*.
    if ENERGIA:
        try:
            import energia as _energia
            _energia.generar()
        except Exception as e:
            print(f"AVISO: no se pudo actualizar la eficiencia energetica ({type(e).__name__}: {e}). "
                  f"Regenerala con: python3.11 benchmarks/energia.py", flush=True)

    return destino
