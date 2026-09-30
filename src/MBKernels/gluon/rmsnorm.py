"""RMSNorm en Gluon: y = x / sqrt(mean(x^2) + eps) * w.

Mismo algoritmo que MBKernels/triton/rmsnorm_baseline.py (un programa por fila, fila
entera en registros, reduccion en fp32), pero en Gluon el reparto de la fila entre
hilos NO lo decide el compilador: se fija con un BlockedLayout explicito.

Layout: cada hilo lleva VEC elementos contiguos (16 bytes en fp16/bf16 -> una carga
vectorizada ld.global.v4), 32 hilos por warp y num_warps warps en la dimension de la
fila. Si la fila es mas larga que el layout (VEC*32*num_warps), el layout se repite y
cada hilo procesa varios bloques de VEC.

API de Triton 3.8 (python/tutorials/gluon/02-layouts.py). Sin tensor cores: no hay
producto matricial, se mide en GB/s.
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

MAX_FUSED_SIZE = 65536


@gluon.jit
def rmsnorm_kernel(x_ptr, y_ptr, w_ptr, stride_x_row, stride_y_row, N, eps,
                   BLOCK_SIZE: gl.constexpr, layout: gl.constexpr):
    row = gl.program_id(0)
    cols = gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = cols < N

    x = gl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(gl.float32)
    var = gl.sum(x * x, axis=0) / N
    rstd = gl.rsqrt(var + eps)

    w_raw = gl.load(w_ptr + cols, mask=mask, other=0.0)
    y = x * rstd * w_raw.to(gl.float32)
    gl.store(y_ptr + row * stride_y_row + cols, y.to(w_raw.dtype), mask=mask)  # dtype de entrada


def configuracion(N, elem_bytes):
    """BLOCK_SIZE, num_warps y layout para una fila de N elementos."""
    block = triton.next_power_of_2(N)
    num_warps = min(max(block // 256, 1), 16)          # misma heuristica que el kernel Triton
    vec = max(1, min(16 // elem_bytes, block // (32 * num_warps)))  # 16 B por hilo si cabe
    layout = gl.BlockedLayout(size_per_thread=[vec], threads_per_warp=[32],
                              warps_per_cta=[num_warps], order=[0])
    return block, num_warps, layout


def rmsnorm(x, w, eps=1e-5, devolver_kernel=False):
    """RMSNorm sobre la ultima dimension de x (M x N) con pesos w (N,)."""
    assert x.ndim == 2 and w.shape == (x.shape[1],), "x debe ser (M, N) y w (N,)"
    assert x.stride(1) == 1, "las filas de x deben ser contiguas"
    M, N = x.shape
    y = torch.empty_like(x)
    block, num_warps, layout = configuracion(N, x.element_size())
    assert block <= MAX_FUSED_SIZE, f"N={N} es demasiado grande para fusionar la fila (max {MAX_FUSED_SIZE})"
    kernel = rmsnorm_kernel[(M,)](x, y, w, x.stride(0), y.stride(0), N, eps,
                                  BLOCK_SIZE=block, layout=layout, num_warps=num_warps)
    return (y, kernel) if devolver_kernel else y
