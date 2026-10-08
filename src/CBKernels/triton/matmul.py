"""Matmul en Triton con @triton.autotune (baseline del TFM).

Para cada (M, N, K) el autotuner prueba todas las configuraciones (BLOCK_SIZE_M/N/K,
GROUP_SIZE_M, num_warps, num_stages) y se queda con la mas rapida; la elegida queda
en matmul_kernel.best_config.

tl.dot se compila a instrucciones MMA de tensor core (mma.sync.m16n8k16 en sm_121):
fp16/bf16 directamente y fp32 en TF32 (input_precision por defecto de Triton).
Compruebalo con comun.exigir_tensor_cores(matmul(a, b, devolver_kernel=True)[1]).
"""
import itertools

import torch
import triton
import triton.language as tl


def configs_autotune():
    """Espacio de busqueda: 3 x 3 x 2 x 2 x 2 = 72 combinaciones, menos las podadas."""
    configs = []
    for bm, bn, bk, warps, stages in itertools.product(
        [64, 128, 256],  # BLOCK_SIZE_M
        [64, 128, 256],  # BLOCK_SIZE_N
        [32, 64],        # BLOCK_SIZE_K
        [4, 8],          # num_warps
        [3, 4],          # num_stages
    ):
        if bm * bn >= 256 * 256:  # 256x256 no cabe en registros/memoria compartida
            continue
        configs.append(triton.Config(
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8},
            num_warps=warps, num_stages=stages))
    return configs


# Las configuraciones que no caben en memoria compartida (OutOfResources) las
# descarta el propio autotuner. key: se re-afina para cada (M, N, K).
@triton.autotune(configs=configs_autotune(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Reordenacion en grupos de GROUP_SIZE_M filas de bloques para mejorar la reutilizacion en L2.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # tl.dot -> tensor cores; acumulacion en fp32 para precision.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul(a, b, devolver_kernel=False):
    """C = A @ B. Con devolver_kernel=True devuelve tambien el kernel compilado (para ver su PTX)."""
    assert a.shape[1] == b.shape[0], "dimensiones incompatibles"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    kernel = matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return (c, kernel) if devolver_kernel else c
