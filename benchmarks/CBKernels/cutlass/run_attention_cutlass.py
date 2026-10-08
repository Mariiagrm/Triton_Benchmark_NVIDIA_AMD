"""FlashAttention-2 (forward) de CUTLASS (CuTe DSL) frente a SDPA de PyTorch. Imagen: cutlass.

Kernel: el ejemplo oficial ampere/kernel/attention/flash_attention_v2.py de CUTLASS v4.8.0
(FlashAttentionForwardAmpere), sin modificar, cargado por CBKernels/cutlass/attention_fa2.py.
El ejemplo no tiene autotuning (Triton y Helion si): por defecto se barren sus parametros
publicos m_block_size, n_block_size en {64, 128} y num_threads en {128, 256}, solo las
combinaciones que acepta su propio FlashAttentionForwardAmpere.can_implement, y se usa la mas
rapida (como el matmul de Gluon). --bloques M N HILOS fija una (la del ejemplo: 128 128 128).
La 1a llamada incluye cute.compile de cada candidata y el barrido. Comprueba en el PTX del kernel (__ptx__, con
CUTE_DSL_KEEP=ptx) que usa tensor cores.
Banco comun a todos los DSLs: benchmarks/banco_attention.py.

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_attention_cutlass [--bloques 128 64 128] [--seqlens ...]
Resultados: results/<maquina>/run_attention_cutlass/<fecha>_job<JOBID>/ y results/attention_metrics.csv
"""
import itertools

import torch

import banco_attention
import comun
import validation
from CBKernels.cutlass.attention_fa2 import ejemplo, preparar

CANDIDATAS = list(itertools.product((64, 128), (64, 128), (128, 256)))

if __name__ == "__main__":
    p = banco_attention.parser(__doc__)
    p.add_argument("--bloques", type=int, nargs=3, default=None, metavar=("M_BLOCK", "N_BLOCK", "HILOS"),
                   help="una configuracion fija (por defecto: barrido de CANDIDATAS)")
    args = p.parse_args()
    compilados = {}
    elegidas = {}

    def preparar_fn(q, k, v, causal, escala):
        import cutlass

        if args.bloques:
            candidatas = [tuple(args.bloques)]
        else:
            fa2 = ejemplo().FlashAttentionForwardAmpere
            candidatas = [c for c in CANDIDATAS if fa2.can_implement(cutlass.Float16, q.shape[-1], *c, causal)]
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=escala)
        mejor = None
        for c in candidatas:
            fn, compilado = preparar(q, k, v, causal, escala, c)
            validation.comprobar(fn(), ref, nombre=f"attention CuTe bloques={c}", **banco_attention.TOL_ATTENTION)
            ms = comun.medir(fn, warmup=10, rep=50)["ms"] if len(candidatas) > 1 else 0.0
            print(f"   candidata BM={c[0]} BN={c[1]} hilos={c[2]}: {ms:.3f} ms", flush=True)
            if mejor is None or ms < mejor[0]:
                mejor = (ms, c, fn, compilado)
        _, c, fn, compilado = mejor
        compilados[fn], elegidas[fn] = compilado, c
        return fn

    def detalle(q, k, v, causal, fn):
        mma = comun.instrucciones_tensor_core_ptx(compilados[fn].__ptx__)
        if not mma:
            raise RuntimeError("attention CuTe NO usa tensor cores: su PTX no contiene mma.sync/wgmma/tcgen05.mma")
        m, n, h = elegidas[fn]
        return f"FlashAttentionForwardAmpere BM={m} BN={n} hilos={h} mma={';'.join(mma)}"

    banco_attention.ejecutar(args, "cutlass", preparar_fn, detalle)
