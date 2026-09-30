"""RMSNorm en Triton (baseline memory-bound): y = x / sqrt(mean(x^2) + eps) * w.

Un programa por fila: carga la fila entera en registros (BLOCK_SIZE = siguiente
potencia de 2 de N), reduce x^2 en fp32 y escribe la fila normalizada. Cada byte se
lee y escribe una sola vez, asi que el limite es el ancho de banda de la memoria
(LPDDR5X unificada en GB10), no el computo.

No usa tensor cores: no hay producto matricial (tl.dot), solo una reduccion por fila.
Por eso los experimentos de RMSNorm miden GB/s y no llaman a comun.exigir_tensor_cores.
"""
import torch
import triton
import triton.language as tl

# Fila maxima fusionable en un programa: 64K elementos (limite de tl.arange y de registros).
MAX_FUSED_SIZE = 65536


@triton.jit
def rmsnorm_kernel(
    x_ptr, y_ptr, w_ptr,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Un programa por fila (token).
    row_idx = tl.program_id(0)
    x_ptr += row_idx * stride_x_row
    y_ptr += row_idx * stride_y_row

    # Mascara para N que no son potencia de 2.
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Reduccion en fp32 para evitar desbordamiento/perdida de precision con fp16/bf16.
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(y_ptr + cols, y.to(y_ptr.dtype.element_ty), mask=mask)  # mismo dtype que la salida


def num_warps_para(block_size):
    """Mas warps para filas largas: con 4 fijos, BLOCK_SIZE grande desborda registros."""
    return min(max(block_size // 256, 1), 16)


def rmsnorm(x, w, eps=1e-5, devolver_kernel=False):
    """RMSNorm sobre la ultima dimension de x (M x N) con pesos w (N,)."""
    assert x.ndim == 2 and w.shape == (x.shape[1],), "x debe ser (M, N) y w (N,)"
    assert x.stride(1) == 1, "las filas de x deben ser contiguas"
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    assert BLOCK_SIZE <= MAX_FUSED_SIZE, f"N={N} es demasiado grande para fusionar la fila (max {MAX_FUSED_SIZE})"

    kernel = rmsnorm_kernel[(M,)](
        x, y, w,
        x.stride(0), y.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps_para(BLOCK_SIZE),
    )
    return (y, kernel) if devolver_kernel else y
