"""Matmul denso fp16 de CUTLASS (CuTe DSL) para Blackwell GeForce (sm_120/121).

Usa SIN MODIFICAR el ejemplo oficial de NVIDIA
    $CUTLASS_DIR/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py
(clase Sm120GemmKernel: TMA, pipeline multietapa, MMA de Blackwell, acumulacion fp32). Este
modulo solo lo carga y le pasa los tensores de PyTorch, igual que hace su funcion run():
tensores CuTe con cutlass.cute.runtime.from_dlpack(...).mark_layout_dynamic(...), como
cutlass.torch.cute_tensor_like, y cute.compile(...) del kernel.

El kernel trabaja con tensores (modo0, modo1, L) y deja elegir el orden de memoria; se usan
las matrices de PyTorch tal cual, sin copias:
    A (M, K) fila  -> (M, K, 1), K contigua   ("k-major", a_major="k")
    B (K, N) fila  -> (N, K, 1), N contigua   ("n-major", b_major="n")
    C (M, N) fila  -> (M, N, 1), N contigua   ("n-major", c_major="n")
Solo fp16 (el kernel rechaza bf16: "a_dtype should be float16 or float8").

Requiere la imagen cutlass (CUTLASS_DIR y el paquete nvidia-cutlass-dsl).
"""
import importlib.util
import os

import torch

_EJEMPLO = os.path.join("examples", "python", "CuTeDSL", "cute", "blackwell_geforce", "kernel",
                        "dense_gemm", "dense_gemm.py")
TILE_POR_DEFECTO = (128, 128, 64)  # el valor por defecto del propio ejemplo
_modulo = None


def ejemplo():
    """Modulo dense_gemm.py oficial (cargado una vez desde $CUTLASS_DIR)."""
    global _modulo
    if _modulo is None:
        ruta = os.path.join(os.environ.get("CUTLASS_DIR", "/opt/cutlass"), _EJEMPLO)
        if not os.path.isfile(ruta):
            raise RuntimeError(f"No encuentro el ejemplo de CUTLASS: {ruta} (usa la imagen cutlass)")
        spec = importlib.util.spec_from_file_location("cutlass_ejemplo_dense_gemm", ruta)
        _modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_modulo)
    return _modulo


def _tensor_cute(t):
    from cutlass.cute.runtime import from_dlpack

    leading = next(i for i, s in enumerate(t.stride()) if s == 1)  # cutlass.torch.get_leading_dim
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=leading)


def preparar(a, b, tile=TILE_POR_DEFECTO, epi_stage=4):
    """Compila el kernel para A (M,K) y B (K,N) fp16 y devuelve fn() -> C (M,N).

    La compilacion (cute.compile) se hace aqui, fuera de fn: fn solo lanza el kernel.
    """
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch

    assert a.dtype == torch.float16 and b.dtype == torch.float16, "el kernel CuTe solo admite fp16"
    assert a.is_contiguous() and b.is_contiguous(), "A y B deben ser contiguas (fila)"
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), dtype=torch.float16, device=a.device)
    ta = _tensor_cute(a.unsqueeze(0).permute(1, 2, 0))   # (M, K, 1)
    tb = _tensor_cute(b.unsqueeze(0).permute(2, 1, 0))   # (N, K, 1)
    tc = _tensor_cute(c.unsqueeze(0).permute(1, 2, 0))   # (M, N, 1)

    gemm = ejemplo().Sm120GemmKernel(cutlass.Float32, tuple(tile), epi_stage)
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)  # cluster (1, 1, 1)
    stream = cutlass_torch.default_stream()
    compilado = cute.compile(gemm, ta, tb, tc, max_clusters, stream)

    def fn():
        compilado(ta, tb, tc, stream)
        return c

    return fn
