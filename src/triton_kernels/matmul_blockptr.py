"""Matmul con block-pointers (INTENTO documentado de activar TMA vía tl.make_block_ptr).

>>> Resultado del intento (para la memoria del TFM):
    En Triton 3.x, `tl.make_block_ptr` + `tl.advance` NO garantizan que el backend
    emita instrucciones TMA (cp.async.bulk.tensor). En sm_121/GB10 el compilador
    baja estas cargas a `cp.async` "clasico": se obtiene pipelining pero NO se activa
    el Tensor Memory Accelerator. Es un ejemplo de que las abstracciones de alto nivel
    del compilador no siempre se traducen en la aceleracion de hardware esperada sin la
    sintaxis correcta. La activacion real de TMA se hace con `tl.make_tensor_descriptor`
    (ver triton_kernels/matmul_tma.py).

Se conserva este kernel como variante de comparacion: `experimento_tma` lo mide junto
al baseline y al kernel con descriptores TMA, y reporta la estrategia de carga de cada
uno (comun.copias_asincronas) para evidenciar la diferencia (cp.async vs TMA).

El resto (swizzling L2, tl.dot -> tensor cores, acumulacion fp32) es identico al baseline.
"""
import itertools

import torch
import triton
import triton.language as tl


def configs_autotune():
    """Como el baseline pero con num_stages [2,3,4,5]."""
    configs = []
    for bm, bn, bk, warps, stages in itertools.product(
        [64, 128, 256],   # BLOCK_SIZE_M
        [64, 128, 256],   # BLOCK_SIZE_N
        [32, 64],         # BLOCK_SIZE_K
        [4, 8],           # num_warps
        [2, 3, 4, 5],     # num_stages
    ):
        if bm * bn >= 256 * 256:
            continue
        configs.append(triton.Config(
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8},
            num_warps=warps, num_stages=stages))
    return configs


@triton.autotune(configs=configs_autotune(), key=["M", "N", "K"])
@triton.jit
def matmul_blockptr_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Swizzling L2: grupos de GROUP_SIZE_M filas de bloques (igual que el baseline).
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block-pointers: describen el bloque y DEJAN al backend elegir la carga.
    # En sm_121 el resultado es cp.async, no TMA (ver docstring).
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0))

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = acc.to(c_ptr.dtype.element_ty)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def matmul_blockptr(a, b, devolver_kernel=False):
    """C = A @ B con block-pointers. devolver_kernel=True -> tambien el kernel compilado."""
    assert a.shape[1] == b.shape[0], "dimensiones incompatibles"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    kernel = matmul_blockptr_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return (c, kernel) if devolver_kernel else c
