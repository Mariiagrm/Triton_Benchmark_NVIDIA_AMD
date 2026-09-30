"""Utilidades comunes a todos los experimentos del TFM: contexto, medida y guardado.

Estandar: toda operacion matricial se ejecuta en TENSOR CORES, no en CUDA cores.
    - Al importar este modulo, fp32 pasa a TF32 en cuBLAS/cuDNN (PyTorch lo
      ejecuta en CUDA cores por defecto).
    - exigir_tensor_cores(kernel_compilado) aborta si un kernel Triton no
      contiene instrucciones MMA de tensor core en su PTX.

Todo experimento deberia:
    1. medir con medir()   -> misma metodologia (triton.testing.do_bench, mediana)
    2. guardar con guardar() -> results/<experimento>/<fecha>_job<JOBID>/
                                  resultados.csv  (una fila por medida)
                                  meta.json       (contexto + parametros + resumen)
       y results/<experimento>/ultimo -> la ejecucion mas reciente; ademas regenera
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
import sys

import torch
import triton

RAIZ = os.environ.get("TFM_RAIZ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        "host": os.environ.get("TFM_HOST", platform.node()),
        "slurm_job": os.environ.get("SLURM_JOB_ID") or None,
        "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
        "comando": " ".join(sys.argv),
    }


def imprimir_contexto():
    c = contexto()
    print(f"GPU: {c['gpu']} (sm_{c['sm']}) | torch {c['torch']} | triton {c['triton']} | "
          f"helion {c['helion']} | job {c['slurm_job'] or 'local'}\n", flush=True)


def medir(fn, flops=None, bytes_movidos=None, warmup=100, rep=500):
    """Mide fn con do_bench. Devuelve ms (mediana, p20, p80) y, si se dan, TFLOP/s y GB/s."""
    ms, p20, p80 = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
    r = {"ms": round(ms, 4), "ms_p20": round(p20, 4), "ms_p80": round(p80, 4)}
    if flops is not None:
        r["tflops"] = round(flops / (ms * 1e-3) / 1e12, 2)
    if bytes_movidos is not None:
        r["gbs"] = round(bytes_movidos / (ms * 1e-3) / 1e9, 1)
    return r


def instrucciones_tensor_core(compilado):
    """Instrucciones MMA de tensor core (sin repetir) en el PTX de un kernel Triton compilado."""
    asm = getattr(compilado, "asm", None)
    if asm is None or "ptx" not in asm:
        raise TypeError(f"se esperaba un kernel Triton compilado (con .asm['ptx']), no {type(compilado).__name__}")
    return sorted(set(_MMA.findall(asm["ptx"])))


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
    base = os.path.join(RAIZ, "results", nombre)
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

    return destino
