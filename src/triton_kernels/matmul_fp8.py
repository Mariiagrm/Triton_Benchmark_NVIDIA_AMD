"""Matmul en Triton con entradas FP8 (e4m3), acumulacion FP32.

Los tensor cores de Blackwell (sm_120/121) ejecutan MMA en FP8 al doble de throughput
que en FP16. Este kernel carga A y B en FP8 (torch.float8_e4m3fn), multiplica con
tl.dot (que emite la MMA en FP8) y acumula en FP32; la salida se escribe en el dtype
de C (por defecto fp16). Es el kernel para intentar superar el techo de ~100 TFLOP/s
que el matmul denso fp16 no rebasa (ver docs/tma.md: el limite es la MMA, no la memoria).

Mismo esqueleto que el baseline (swizzling L2 por grupos, aritmetica de punteros) para
que la comparacion aisle el efecto de la precision.

Nota de precision: FP8 e4m3 tiene ~3 bits de mantisa; el error relativo del matmul es
de varios %. El experimento (experimento_fp8.py) lo mide y reporta explicitamente.
"""
import itertools

import torch
import triton
import triton.language as tl


def configs_autotune():
    configs = []
    for bm, bn, bk, warps, stages in itertools.product(
        [64, 128, 256],   # BLOCK_SIZE_M
        [64, 128, 256],   # BLOCK_SIZE_N
        [64, 128],        # BLOCK_SIZE_K (mayor en fp8: mas K por MMA)
        [4, 8],           # num_warps
        [3, 4],           # num_stages
    ):
        if bm * bn >= 256 * 256:
            continue
        configs.append(triton.Config(
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8},
            num_warps=warps, num_stages=stages))
    return configs


@triton.autotune(configs=configs_autotune(), key=["M", "N", "K"])
@triton.jit
def matmul_fp8_kernel(
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

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # A y B son fp8; tl.dot emite la MMA en fp8 y acumula en fp32.
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


def matmul_fp8(a, b, out_dtype=torch.float16, devolver_kernel=False):
    """C = A @ B con A, B en FP8 (float8_e4m3fn/e5m2). C en out_dtype (fp16 por defecto)."""
    assert a.shape[1] == b.shape[0], "dimensiones incompatibles"
    assert a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "a debe ser fp8"
    assert b.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "b debe ser fp8"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    kernel = matmul_fp8_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return (c, kernel) if devolver_kernel else c
