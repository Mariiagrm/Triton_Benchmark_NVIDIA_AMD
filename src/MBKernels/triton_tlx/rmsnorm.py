"""RMSNorm en Triton + TLX: y = x / sqrt(mean(x^2) + eps) * w, con prefetch asincrono explicito.

El kernel Triton base (MBKernels/triton/rmsnorm_baseline.py) lanza un programa por fila y
deja las cargas al compilador. Aqui TLX (Triton Language Extensions, paquete triton-utlx)
permite gestionar la memoria a mano, que es lo que importa en un operador memory-bound:

    - Kernel PERSISTENTE: CTAS_POR_SM programas por SM; cada uno recorre las filas
      pid, pid + nprog, pid + 2*nprog, ...
    - Pipeline de NUM_STAGES buffers en memoria compartida (tlx.local_alloc): mientras se
      normaliza la fila i, las filas i+1 .. i+NUM_STAGES-1 ya se estan copiando de
      memoria global a compartida con cp.async (tlx.async_load + commit/wait_group).
      Asi la latencia de la memoria queda oculta tras el computo de la fila anterior.
    - w se carga UNA vez por programa, no una por fila.

Las barreras entre la escritura asincrona de un buffer y su lectura (tlx.local_load) las
inserta el analisis de memoria compartida de Triton (Membar), como en los tutoriales TLX.

API de triton-ext/extensions/utlx (test/test_tlx.py::test_local_load y el tutorial
blackwell-multi-cta-layernorm). Sin tensor cores: se mide en GB/s.
"""
import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

MAX_FUSED_SIZE = 65536
# Memoria compartida que se reparte entre los CTAs de un SM (GB10/sm_121: ~100 KB por SM).
SMEM_POR_SM = 96 * 1024


@triton.jit
def rmsnorm_kernel(x_ptr, y_ptr, w_ptr, stride_x_row, stride_y_row, M, N, eps,
                   BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask_cols = cols < N

    w = tl.load(w_ptr + cols, mask=mask_cols, other=0.0).to(tl.float32)
    buffers = tlx.local_alloc((BLOCK_SIZE,), x_ptr.dtype.element_ty, NUM_STAGES)

    # Prologo: lanzar las NUM_STAGES-1 primeras filas de este programa (un grupo por fila).
    for s in tl.static_range(NUM_STAGES - 1):
        fila = pid + s * nprog
        tlx.async_load(x_ptr + fila * stride_x_row + cols, tlx.local_view(buffers, s),
                       mask=mask_cols & (fila < M))
        tlx.async_load_commit_group()

    n_filas = tl.cdiv(M - pid, nprog)
    for i in range(0, n_filas):
        # Prefetch de la fila i+NUM_STAGES-1 en el buffer que se libero en la iteracion i-1.
        j = i + NUM_STAGES - 1
        fila_j = pid + j * nprog
        tlx.async_load(x_ptr + fila_j * stride_x_row + cols, tlx.local_view(buffers, j % NUM_STAGES),
                       mask=mask_cols & (fila_j < M))
        tlx.async_load_commit_group()
        # Quedan como mucho NUM_STAGES-1 grupos pendientes -> el de la fila i ya ha llegado.
        tlx.async_load_wait_group(NUM_STAGES - 1)

        x = tlx.local_load(tlx.local_view(buffers, i % NUM_STAGES)).to(tl.float32)
        rstd = 1 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
        y = x * rstd * w
        fila = pid + i * nprog
        tl.store(y_ptr + fila * stride_y_row + cols, y.to(y_ptr.dtype.element_ty), mask=mask_cols)


def configuracion(N, elem_bytes, num_stages):
    """BLOCK_SIZE, num_warps y CTAs por SM (los que caben en la memoria compartida)."""
    block = triton.next_power_of_2(N)
    num_warps = min(max(block // 256, 1), 16)  # misma heuristica que el kernel Triton
    smem_cta = num_stages * block * elem_bytes
    ctas_por_sm = max(1, min(4, SMEM_POR_SM // smem_cta))
    return block, num_warps, ctas_por_sm


def rmsnorm(x, w, eps=1e-5, num_stages=3, devolver_kernel=False):
    """RMSNorm sobre la ultima dimension de x (M x N) con pesos w (N,)."""
    assert x.ndim == 2 and w.shape == (x.shape[1],), "x debe ser (M, N) y w (N,)"
    assert x.stride(1) == 1, "las filas de x deben ser contiguas"
    assert num_stages >= 2, "el pipeline necesita al menos 2 buffers"
    M, N = x.shape
    y = torch.empty_like(x)
    block, num_warps, ctas_por_sm = configuracion(N, x.element_size(), num_stages)
    assert block <= MAX_FUSED_SIZE, f"N={N} es demasiado grande para fusionar la fila (max {MAX_FUSED_SIZE})"
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    grid = (min(M, sms * ctas_por_sm),)
    kernel = rmsnorm_kernel[grid](x, y, w, x.stride(0), y.stride(0), M, N, eps,
                                  BLOCK_SIZE=block, NUM_STAGES=num_stages, num_warps=num_warps)
    return (y, kernel) if devolver_kernel else y
