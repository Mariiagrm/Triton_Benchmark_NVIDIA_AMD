"""Matmul en Triton con TMA real vía descriptores de tensor (tl.make_tensor_descriptor).

Contexto (para la memoria del TFM):
    El intento previo con `tl.make_block_ptr` (ver triton_kernels/matmul_blockptr.py)
    NO activa el Tensor Memory Accelerator en sm_121/GB10: el backend baja las cargas
    a `cp.async` clasico. La API que emite TMA de verdad (cp.async.bulk.tensor) es la
    de DESCRIPTORES DE TENSOR: `tl.make_tensor_descriptor` + desc.load/desc.store.

Este kernel:
    - Crea descriptores de A, B y C DENTRO del kernel con `tl.make_tensor_descriptor`
      (block_shape en constexpr -> compatible con @triton.autotune).
    - Carga bloques con desc.load([off_fila, off_col]); TMA gestiona el fuera-de-rango
      rellenando con ceros, asi que NO hacen falta mascaras ni boundary_check.
    - Los descriptores on-device necesitan memoria scratch: se registra un allocator
      global con `triton.set_allocator` (obligatorio, si no falla en tiempo de ejecucion).

Requiere Triton 3.x con `tl.make_tensor_descriptor` (presente en la imagen NGC 25.10).
Verifica en el PTX que aparece `cp.async.bulk.tensor` con comun.usa_tma(kernel).

El resto (swizzling L2, tl.dot -> tensor cores, acumulacion fp32) es identico al baseline.
"""
import itertools

import torch
import triton
import triton.language as tl


# Los descriptores de tensor on-device se materializan en memoria global: Triton pide
# ese scratch a traves de un allocator que hay que registrar una vez.
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)


def configs_autotune():
    """Como el baseline pero con num_stages [2,3,4,5] (el pipelining es el objetivo)."""
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
def matmul_tma_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_bk, stride_cm,  # strides de la dim externa (la interna es 1: contiguo)
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Descriptores TMA: base, forma global, strides (dim interna = 1) y forma del bloque.
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

    # tl.dot -> tensor cores; acumulacion fp32. TMA rellena con ceros el fuera-de-rango.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = a_desc.load([offs_am, k])            # bloque [BLOCK_M, BLOCK_K] vía TMA
        b = b_desc.load([k, offs_bn])            # bloque [BLOCK_K, BLOCK_N] vía TMA
        acc = tl.dot(a, b, acc)
    c = acc.to(c_ptr.dtype.element_ty)

    c_desc.store([offs_am, offs_bn], c)          # escritura del bloque vía TMA


def matmul_tma(a, b, devolver_kernel=False):
    """C = A @ B con descriptores TMA. devolver_kernel=True -> tambien el kernel compilado."""
    assert a.shape[1] == b.shape[0], "dimensiones incompatibles"
    # TMA exige la dimension interna contigua (stride 1); lo forzamos para A, B y C.
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    kernel = matmul_tma_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
    )
    return (c, kernel) if devolver_kernel else c
