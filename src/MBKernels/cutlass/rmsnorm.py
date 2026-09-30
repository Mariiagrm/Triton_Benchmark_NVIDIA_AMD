"""RMSNorm de CUTLASS (cutlass::rmsnorm), compilado como extension de PyTorch.

La primera llamada compila rmsnorm_ext.cu con nvcc (torch.utils.cpp_extension, ~1 min);
las siguientes reutilizan el binario de .cache/torch_extensions/. Requiere CUTLASS_DIR
(la imagen cutlass lo define: /opt/cutlass).
"""
import os

import torch
from torch.utils import cpp_extension

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.environ.get("TFM_RAIZ", os.path.dirname(os.path.dirname(os.path.dirname(_AQUI))))
_ext = None


def extension():
    """Compila (o carga de la cache) la extension. Arquitectura: la de la GPU actual."""
    global _ext
    if _ext is None:
        cutlass = os.environ.get("CUTLASS_DIR", "")
        if not os.path.isfile(os.path.join(cutlass, "include", "cutlass", "cutlass.h")):
            raise RuntimeError(f"CUTLASS no encontrado en CUTLASS_DIR={cutlass!r} (usa la imagen cutlass)")
        cap = torch.cuda.get_device_capability(0)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}")  # GB10: 12.1 -> sm_121
        build = os.path.join(_RAIZ, ".cache", "torch_extensions", "tfm_cutlass_rmsnorm")
        os.makedirs(build, exist_ok=True)
        _ext = cpp_extension.load(
            name="tfm_cutlass_rmsnorm",
            sources=[os.path.join(_AQUI, "rmsnorm_ext.cu")],
            extra_include_paths=[os.path.join(cutlass, "include"),
                                 os.path.join(cutlass, "tools", "util", "include")],
            # PyTorch pasa por defecto -D__CUDA_NO_HALF_CONVERSIONS__ (y afines), que quitan
            # la conversion implicita __half -> float de la que depende device_rmsnorm.h
            # ("no suitable conversion function from const __half to float"). Se anulan
            # aqui (-U va despues de los -D de PyTorch); la cabecera de CUTLASS no se toca.
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                               "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                               "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
            build_directory=build,
            verbose=False,
        )
    return _ext


def rmsnorm(x, w, eps=1e-5):
    """RMSNorm sobre la ultima dimension de x (M x N) con pesos w (N,)."""
    return extension().rmsnorm(x, w, eps)


def ruta(x):
    """Kernel de CUTLASS que se usa para x: rmsnorm_twoPassAlgo_e8 (vectorizado) o _e1."""
    return extension().ruta(x)
