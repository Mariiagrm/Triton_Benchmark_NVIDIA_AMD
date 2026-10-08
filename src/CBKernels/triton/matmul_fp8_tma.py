"""Matmul FP8 (e4m3) con TMA (descriptores) — variante optimizada para cerrar el hueco
frente a cuBLASLt.

En FP8 se mueve la mitad de bytes que en FP16, asi que el kernel puede volverse
memory-bound; a diferencia del caso FP16 denso (docs/TFM/tma.md, donde TMA no aportaba),
aqui el transporte por TMA (cp.async.bulk.tensor) puede alimentar la MMA mas rapido y
acercarse a cuBLASLt (torch._scaled_mm).

Combina:
    - Entradas FP8 e4m3, tl.dot -> MMA fp8 (m16n8k32), acumulacion FP32.
    - Cargas por descriptor (tl.make_tensor_descriptor) -> TMA real.
    - Espacio de autotuning mas amplio: num_stages [3,4,5] y BLOCK_K hasta 256.

Verifica en el PTX que emite TMA con comun.usa_tma(kernel).
"""
import itertools

import torch
import triton
import triton.language as tl


# Los descriptores on-device necesitan scratch en memoria global (allocator global).
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)


def configs_autotune():
    """Tiles grandes (mejor para matmul grande) + pipeline profundo + BLOCK_K amplio."""
    configs = []
    for bm, bn, bk, warps, stages in itertools.product(
        [128, 256],        # BLOCK_SIZE_M
        [128, 256],        # BLOCK_SIZE_N
        [64, 128, 256],    # BLOCK_SIZE_K
        [4, 8],            # num_warps
        [3, 4, 5],         # num_stages
    ):
        if bm * bn >= 256 * 256:  # 256x256 no cabe en memoria compartida
            continue
        configs.append(triton.Config(
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8},
            num_warps=warps, num_stages=stages))
    return configs


@triton.autotune(configs=configs_autotune(), key=["M", "N", "K"])
@triton.jit
def matmul_fp8_tma_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_bk, stride_cm,  # strides de la dim externa (interna = 1: contiguo)
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Descriptores TMA (A y B en fp8, C en fp16); el dtype lo infiere del puntero.
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[stride_bk, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

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

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = a_desc.load([offs_am, k])     # fp8, vía TMA
        b = b_desc.load([k, offs_bn])     # fp8, vía TMA
        acc = tl.dot(a, b, acc)           # MMA fp8, acumula fp32
    c = acc.to(c_ptr.dtype.element_ty)

    c_desc.store([offs_am, offs_bn], c)


def matmul_fp8_tma(a, b, out_dtype=torch.float16, devolver_kernel=False):
    """C = A @ B con A, B en FP8 y carga por descriptor TMA. C en out_dtype (fp16 por defecto)."""
    assert a.shape[1] == b.shape[0], "dimensiones incompatibles"
    assert a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "a debe ser fp8"
    assert b.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "b debe ser fp8"
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    kernel = matmul_fp8_tma_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
    )
    return (c, kernel) if devolver_kernel else c
