"""Matmul en Gluon para sm_80+ (incluye sm_120/121, GB10 y RTX 5090): C = A @ B, acumulacion fp32.

Gluon no trae un matmul oficial para esta arquitectura: sus tutoriales de matmul usan
wgmma (05-wgmma, Hopper sm_90) o tcgen05 (06-tcgen05, Blackwell de centro de datos sm_100),
instrucciones que NO existen en sm_120/121, que usa mma.sync. Este kernel es PROPIO y se
construye con las piezas oficiales que si funcionan aqui (Triton v3.8.0):
    - MMA de tensor core: mma_v2 + NVMMADistributedLayout(version=[2, 0]) + DotOperandLayout
      (python/test/gluon/test_core.py::test_mma_v2).
    - Pipeline de copias asincronas a memoria compartida con varios buffers: cp.async_load +
      commit_group/wait_group (python/tutorials/gluon/03-async-copy.py).
    - Esqueleto del matmul (orden de programas agrupado para la L2, GROUP_M) igual que el
      baseline Triton (CBKernels/triton/matmul.py, tutorial 03-matrix-multiplication).

Funcionamiento: cada programa calcula un bloque BM x BN de C. Un anillo de STAGES buffers en
memoria compartida (gl.allocate_shared_memory) recibe los bloques de A (BM x BK) y B (BK x BN)
con cp.async; mientras mma_v2 multiplica el bloque K actual (leido de compartida al layout
de operando), los STAGES-1 siguientes ya se estan copiando. A diferencia de Triton, todos los
layouts (registros, compartida, operandos de la MMA) son explicitos.
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_cm,
                  BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr,
                  STAGES: gl.constexpr, acc_layout: gl.constexpr, a_load: gl.constexpr,
                  b_load: gl.constexpr, a_smem_layout: gl.constexpr, b_smem_layout: gl.constexpr):
    lhs: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=0, k_width=8)
    rhs: gl.constexpr = gl.DotOperandLayout(parent=acc_layout, operand_index=1, k_width=8)

    # Orden de programas agrupado para reutilizar A en la L2 (como el baseline Triton).
    pid = gl.program_id(0)
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, a_load))
    offs_ak = gl.arange(0, BK, layout=gl.SliceLayout(0, a_load))
    offs_bk = gl.arange(0, BK, layout=gl.SliceLayout(1, b_load))
    offs_bn = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, b_load))
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :]
    b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :]
    mask_am = (offs_am < M)[:, None]
    mask_bn = (offs_bn < N)[None, :]

    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [STAGES, BM, BK], layout=a_smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [STAGES, BK, BN], layout=b_smem_layout)
    acc = gl.zeros([BM, BN], gl.float32, layout=acc_layout)

    # Prologo: lanzar los STAGES-1 primeros bloques K (un grupo de copias por bloque).
    for s in gl.static_range(STAGES - 1):
        k0 = s * BK
        cp.async_load(a_smem.index(s), a_ptrs + k0, mask=mask_am & (offs_ak[None, :] + k0 < K))
        cp.async_load(b_smem.index(s), b_ptrs + k0 * stride_bk, mask=(offs_bk[:, None] + k0 < K) & mask_bn)
        cp.commit_group()

    for kb in range(gl.cdiv(K, BK)):
        # Copia del bloque kb+STAGES-1 en el buffer que se libero en la iteracion anterior
        # (fuera de K la mascara es falsa: cp.async rellena con ceros).
        nxt = kb + STAGES - 1
        k0 = nxt * BK
        cp.async_load(a_smem.index(nxt % STAGES), a_ptrs + k0, mask=mask_am & (offs_ak[None, :] + k0 < K))
        cp.async_load(b_smem.index(nxt % STAGES), b_ptrs + k0 * stride_bk,
                      mask=(offs_bk[:, None] + k0 < K) & mask_bn)
        cp.commit_group()
        # Quedan como mucho STAGES-1 grupos pendientes -> el bloque kb ya esta en compartida.
        cp.wait_group(STAGES - 1)
        a = a_smem.index(kb % STAGES).load(lhs)
        b = b_smem.index(kb % STAGES).load(rhs)
        acc = mma_v2(a, b, acc)
    cp.wait_group(0)

    c = gl.convert_layout(acc.to(dtype), b_load)
    offs_cm = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, b_load))
    offs_cn = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, b_load))
    gl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :], c,
             mask=(offs_cm < M)[:, None] & (offs_cn < N)[None, :])


# Configuraciones que prueba el banco: (BM, BN, BK, STAGES, WARPS_M, WARPS_N). Todas caben en
# los 99 KiB de memoria compartida por bloque de sm_120/121 en 16 bits.
CONFIGS = [
    (128, 128, 32, 3, 2, 2), (128, 128, 32, 4, 2, 2), (128, 128, 64, 3, 2, 2),
    (128, 256, 32, 3, 2, 4), (256, 128, 32, 3, 4, 2), (128, 64, 32, 4, 2, 2),
    (64, 128, 32, 4, 2, 2),
]


def layouts(BM, BN, BK, WARPS_M, WARPS_N, elem_bytes):
    """Layouts explicitos del kernel para una configuracion."""
    num_warps = WARPS_M * WARPS_N
    acc = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[WARPS_M, WARPS_N], instr_shape=[16, 8])

    def carga(cols):  # global -> registros: 16 bytes contiguos por hilo en la fila
        vec = 16 // elem_bytes
        hilos_col = min(32, cols // vec)
        return gl.BlockedLayout([1, vec], [32 // hilos_col, hilos_col], [num_warps, 1], [1, 0])

    def compartida(cols):  # swizzle por filas de 128 B para leer sin conflictos de banco
        per_phase = max(1, 128 // (cols * elem_bytes))
        return gl.SwizzledSharedLayout(vec=16 // elem_bytes, per_phase=per_phase,
                                       max_phase=max(1, 8 // per_phase), order=[1, 0])

    return num_warps, acc, carga(BK), carga(BN), compartida(BK), compartida(BN)


def matmul(a, b, config=CONFIGS[0], group_m=8, devolver_kernel=False):
    """C = A @ B con A (M, K) y B (K, N) en fp16/bf16, filas contiguas."""
    assert a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16), "fp16 o bf16"
    assert a.shape[1] == b.shape[0] and a.stride(1) == 1 and b.stride(1) == 1
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    BM, BN, BK, STAGES, WM, WN = config
    num_warps, acc, a_load, b_load, a_sm, b_sm = layouts(BM, BN, BK, WM, WN, a.element_size())
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    kernel = matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0),
                                 BM, BN, BK, group_m, STAGES, acc, a_load, b_load, a_sm, b_sm,
                                 num_warps=num_warps)
    return (c, kernel) if devolver_kernel else c
