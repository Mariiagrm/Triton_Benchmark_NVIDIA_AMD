"""Pico de los tensor cores con mma.sync: compila pico_mma_ext.cu y lo expone a Python.

Se compila con nvcc a una biblioteca compartida (ctypes, sin pybind) para controlar la
arquitectura: las MMA block-scaled de FP4/MXFP8 solo existen en la variante especifica
sm_120a/sm_121a. Antes, cada instruccion se compila sola (-cubin, segundos): las que ptxas
rechaza quedan fuera con su error, sin impedir medir las demas.

La biblioteca se guarda en .cache/pico_mma/<huella>/ (huella = fuente + arquitectura + nvcc).
"""
import ctypes
import hashlib
import os
import re
import shutil
import subprocess

import torch

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.environ.get("TFM_RAIZ", os.path.dirname(os.path.dirname(_AQUI)))
_FUENTE = os.path.join(_AQUI, "pico_mma_ext.cu")

# id (enum del .cu), macro, instruccion PTX, (M, N, K), tipo de las entradas en torch para
# rellenar los fragmentos con datos reales, unidad. FLOP por instruccion = 2*M*N*K.
VARIANTES = {
    "f16·f32acc": (0, "F16_F32", "mma.m16n8k16.f32.f16.f16.f32", (16, 8, 16), "fp16", "TFLOP/s"),
    "f16·f16acc": (1, "F16_F16", "mma.m16n8k16.f16.f16.f16.f16", (16, 8, 16), "fp16", "TFLOP/s"),
    "bf16·f32acc": (2, "BF16_F32", "mma.m16n8k16.f32.bf16.bf16.f32", (16, 8, 16), "bf16", "TFLOP/s"),
    "tf32·f32acc": (3, "TF32_F32", "mma.m16n8k8.f32.tf32.tf32.f32", (16, 8, 8), "fp32", "TFLOP/s"),
    "e4m3·f32acc": (4, "E4M3_F32", "mma.m16n8k32.f32.e4m3.e4m3.f32", (16, 8, 32), "fp8", "TFLOP/s"),
    "e4m3·f16acc": (5, "E4M3_F16", "mma.m16n8k32.f16.e4m3.e4m3.f16", (16, 8, 32), "fp8", "TFLOP/s"),
    "s8·s32acc": (6, "S8_S32", "mma.m16n8k32.s32.s8.s8.s32", (16, 8, 32), "int8", "TOP/s"),
    "e2m1·f32acc": (7, "F8F6F4_E2M1", "mma.kind::f8f6f4.m16n8k32.f32.e2m1.e2m1.f32", (16, 8, 32), "fp4x8", "TFLOP/s"),
    "mxf8 e4m3": (8, "MXF8_E4M3", "mma.kind::mxf8f6f4.block_scale.1X.m16n8k32.e4m3.ue8m0", (16, 8, 32), "fp8", "TFLOP/s"),
    "mxf4 e2m1": (9, "MXF4_E2M1", "mma.kind::mxf4nvf4.block_scale.2X.m16n8k64.e2m1.ue8m0", (16, 8, 64), "fp4", "TFLOP/s"),
    "nvf4 e2m1": (10, "NVF4_E2M1", "mma.kind::mxf4nvf4.block_scale.4X.m16n8k64.e2m1.ue4m3", (16, 8, 64), "fp4", "TFLOP/s"),
}

_lib = None
_errores = {}  # variante -> error de compilacion (las que ptxas no acepta)


def nvcc():
    for c in (os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc"), shutil.which("nvcc")):
        if c and os.path.isfile(c):
            return c
    raise RuntimeError("nvcc no encontrado (CUDA_HOME o PATH)")


def arquitectura():
    """sm_XYa: la variante especifica de la arquitectura (necesaria para las block-scaled)."""
    cap = torch.cuda.get_device_capability(0)
    return f"sm_{cap[0]}{cap[1]}a"


def _gencode(arch):
    """-gencode explicito: con '-arch=sm_121a -shared' nvcc genera PTX para compute_121 (sin la
    'a') y ptxas rechaza las MMA block-scaled; asi el PTX es el de la variante especifica."""
    n = arch[3:]
    return f"-gencode=arch=compute_{n},code=sm_{n}"


def _version_nvcc(exe):
    return subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]


def _compila_sola(exe, arch, macro, destino):
    """Compila solo esa instruccion a cubin. Devuelve None o el error de ptxas."""
    r = subprocess.run([exe, "-cubin", _gencode(arch), "-O3", "-std=c++17", f"-DCON_{macro}",
                        _FUENTE, "-o", os.path.join(destino, f"prueba_{macro}.cubin")],
                       capture_output=True, text=True)
    if r.returncode == 0:
        return None
    lineas = [l for l in (r.stderr + r.stdout).splitlines() if "error" in l.lower()]
    return " | ".join(lineas[:3]) or f"nvcc devolvio {r.returncode}"


def biblioteca():
    """Compila (o reutiliza) la biblioteca con todas las instrucciones que compilan."""
    global _lib
    if _lib is not None:
        return _lib
    exe, arch = nvcc(), arquitectura()
    with open(_FUENTE, "rb") as f:
        huella = hashlib.sha1(f.read() + _gencode(arch).encode() + _version_nvcc(exe).encode()).hexdigest()[:12]
    destino = os.path.join(os.environ.get("TFM_CACHE") or os.path.join(_RAIZ, ".cache"), "pico_mma", huella)
    so = os.path.join(destino, "libpico_mma.so")
    errores_txt = os.path.join(destino, "errores.txt")
    os.makedirs(destino, exist_ok=True)

    if not os.path.isfile(so):
        _errores.clear()
        for nombre, (_, macro, *_r) in VARIANTES.items():
            err = _compila_sola(exe, arch, macro, destino)
            if err:
                _errores[nombre] = err
                print(f"  [{nombre}] no compila en {arch}: {err}", flush=True)
        macros = [f"-DCON_{v[1]}" for n, v in VARIANTES.items() if n not in _errores]
        r = subprocess.run([exe, "-shared", "-Xcompiler", "-fPIC", _gencode(arch), "-O3", "-std=c++17",
                            *macros, _FUENTE, "-o", so], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"no se pudo compilar {so}:\n{r.stderr}")
        with open(errores_txt, "w") as f:
            f.writelines(f"{n}\t{e}\n" for n, e in _errores.items())
    elif os.path.isfile(errores_txt):
        with open(errores_txt) as f:
            _errores.update(l.rstrip("\n").split("\t", 1) for l in f if "\t" in l)

    lib = ctypes.CDLL(so)
    lib.disponible.argtypes = [ctypes.c_int]
    lib.lanzar.argtypes = [ctypes.c_int] * 5 + [ctypes.c_void_p] * 3
    lib.nombre_error.restype = ctypes.c_char_p
    lib.ruta = so
    _lib = lib
    return lib


def errores():
    """Instrucciones que ptxas no acepta en esta GPU: {variante: error}."""
    biblioteca()
    return dict(_errores)


def disponible(variante):
    return bool(biblioteca().disponible(VARIANTES[variante][0]))


def flop_por_mma(variante):
    m, n, k = VARIANTES[variante][3]
    return 2 * m * n * k


def fragmentos(variante, dispositivo="cuda"):
    """Buffer de 4096 palabras de 32 bits con datos aleatorios del tipo de entrada."""
    tipo = VARIANTES[variante][4]
    g = torch.Generator(device=dispositivo).manual_seed(0)
    if tipo in ("fp16", "bf16", "fp32"):
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[tipo]
        x = torch.randn(4096 * 4 // dt.itemsize, device=dispositivo, generator=g).to(dt)
    elif tipo == "fp8":
        x = torch.randn(4096 * 4, device=dispositivo, generator=g).to(torch.float8_e4m3fn)
    elif tipo == "int8":
        x = torch.randint(-128, 128, (4096 * 4,), device=dispositivo, generator=g, dtype=torch.int8)
    else:  # fp4 empaquetado (2 por byte) o fp4 en contenedor de 8 bits: bytes aleatorios
        x = torch.randint(0, 256, (4096 * 4,), device=dispositivo, generator=g, dtype=torch.uint8)
        if tipo == "fp4x8":
            x = x & 0x0F  # e2m1 en los 4 bits bajos de cada byte
    return x.view(torch.int32).contiguous()


def lanzador(variante, ilp, bloques, hilos, iters):
    """Funcion sin argumentos que lanza el kernel en el stream actual (para do_bench)."""
    lib = biblioteca()
    v = VARIANTES[variante][0]
    buf = fragmentos(variante)
    out = torch.empty(bloques * hilos, device="cuda", dtype=torch.int32)

    def fn():
        e = lib.lanzar(v, ilp, bloques, hilos, iters, buf.data_ptr(), out.data_ptr(),
                       torch.cuda.current_stream().cuda_stream)
        if e:
            raise RuntimeError(f"{variante}: {lib.nombre_error(e).decode()}")

    return fn


def sass_mma():
    """Instrucciones MMA del SASS por variante: {variante: [opcodes]} (cuobjdump -sass)."""
    lib = biblioteca()
    cuobjdump = os.path.join(os.path.dirname(nvcc()), "cuobjdump")
    texto = subprocess.run([cuobjdump, "-sass", lib.ruta], capture_output=True, text=True).stdout
    por_id = {}
    actual = None
    for linea in texto.splitlines():
        m = re.search(r"Function : \S*bucle_mma\S*?ILi(\d+)ELi\d+E", linea)
        if m:
            actual = int(m.group(1))
            continue
        if actual is not None:
            por_id.setdefault(actual, set()).update(re.findall(r"\b[A-Z]*MMA(?:\.[\w]+)*", linea))
    return {n: sorted(por_id.get(v[0], ())) for n, v in VARIANTES.items()}
