"""FlashAttention (forward) en Triton frente a SDPA de PyTorch. Imagen: triton-tlx.

Kernel: CBKernels/triton/fused_attention.py, copia LITERAL del tutorial oficial de Triton
python/tutorials/06-fused-attention.py (tag v3.8.0, c01b677): FlashAttention-2 con autotuning
de BLOCK_M/BLOCK_N/num_stages/num_warps y descriptores TMA de host (capability >= 9). Se llama
con warp_specialize=False: el propio benchmark del tutorial solo lo activa en Hopper y en
Blackwell de centro de datos (is_hopper / is_blackwell: capability 9 y 10), no en sm_120/121.
Comprueba en el PTX de cada kernel compilado que usa tensor cores (comun.exigir_tensor_cores).
Banco comun a todos los DSLs: benchmarks/banco_attention.py.

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_attention_triton [--seqlens 1024 4096 16384] [--head-dims 64 128]
Resultados: results/<maquina>/run_attention_triton/<fecha>_job<JOBID>/ y results/attention_metrics.csv
"""
import banco_attention
import comun
from CBKernels.triton import fused_attention as fa


def preparar(q, k, v, causal, escala):
    return lambda: fa.attention(q, k, v, causal, escala, False)


def detalle(q, k, v, causal, fn):
    # Todos los kernels compilados (cada configuracion probada por el autotuning) deben usar MMA.
    mma = set()
    for kernel in comun.kernels_triton_compilados(fa._attn_fwd):
        mma.update(comun.exigir_tensor_cores(kernel, "attention Triton"))
    cfg = fa._attn_fwd.best_config  # la que el autotuning eligio en la 1a llamada (esta forma)
    return (f"BM={cfg.kwargs['BLOCK_M']} BN={cfg.kwargs['BLOCK_N']} stages={cfg.num_stages} "
            f"num_warps={cfg.num_warps} mma={';'.join(sorted(mma))}")


if __name__ == "__main__":
    args = banco_attention.parser(__doc__).parse_args()
    banco_attention.ejecutar(args, "triton", preparar, detalle)
