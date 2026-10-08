"""FlashAttention (forward) en Helion frente a SDPA de PyTorch. Imagen: helion.

Kernel: CBKernels/helion/attention.py, que carga SIN MODIFICAR el ejemplo oficial de Helion
examples/attention.py (attention_output / causal_attention_output). Helion autotunea cada
forma en su 1a llamada (columna t_compilacion_s); --autotune fija el esfuerzo: none (config
por defecto), quick (por defecto aqui) o full. Comprueba en el PTX de los kernels Triton que
genera Helion que usan tensor cores (comun.exigir_tensor_cores).
Banco comun a todos los DSLs: benchmarks/banco_attention.py.

Nota: el kernel causal del ejemplo no salta los bloques del triangulo superior (los enmascara),
asi que hace el mismo trabajo que el no causal; los TFLOP/s causales se calculan, como en todos
los DSLs, con los FLOPs utiles (la mitad).

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_attention_helion [--autotune full] [--seqlens ...]
Resultados: results/<maquina>/run_attention_helion/<fecha>_job<JOBID>/ y results/attention_metrics.csv
"""
import gc
import os

import banco_attention
import comun

if __name__ == "__main__":
    p = banco_attention.parser(__doc__)
    p.add_argument("--autotune", choices=["none", "quick", "full"], default="quick",
                   help="esfuerzo de autotuning de Helion (HELION_AUTOTUNE_EFFORT)")
    p.add_argument("--benchmark-timeout", type=int, default=300,
                   help="s por configuracion al autotunear (HELION_AUTOTUNE_BENCHMARK_TIMEOUT; Helion: 30). "
                        "En GB10, con N=16384 y D=128, la configuracion inicial de Helion tarda mas de 30 s "
                        "y el autotuning acaba en NoConfigFound (job 20649)")
    args = p.parse_args()
    # Helion lee el esfuerzo al definir el kernel: fijarlo antes de importarlo.
    os.environ["HELION_AUTOTUNE_EFFORT"] = args.autotune
    # Helion pone TRITON_STORE_BINARY_ONLY=1 al autotunear si no esta definida
    # (helion/autotuner/base_search.py) y entonces Triton no guarda el PTX que se revisa abajo.
    os.environ.setdefault("TRITON_STORE_BINARY_ONLY", "0")
    os.environ["HELION_AUTOTUNE_BENCHMARK_TIMEOUT"] = str(args.benchmark_timeout)
    import triton

    from CBKernels.helion import config_elegida
    from CBKernels.helion.attention import kernel

    def preparar(q, k, v, causal, escala):
        kern = kernel(causal)  # escala 1/sqrt(D) fija en el ejemplo: la misma que usa el banco
        return lambda: kern(q, k, v)

    def detalle(q, k, v, causal, fn):
        # Helion genera funciones @triton.jit (_helion_<kernel>); se buscan entre los objetos vivos.
        jits = [o for o in gc.get_objects()
                if isinstance(o, triton.runtime.jit.JITFunction) and "attention" in o.__name__]
        mma = set()
        for jit in jits:
            for compilado in comun.kernels_triton_compilados(jit):
                mma.update(comun.exigir_tensor_cores(compilado, f"attention Helion {jit.__name__}"))
        if not mma:
            raise RuntimeError("no encuentro el kernel Triton compilado por Helion para comprobar sus MMA")
        return f"autotune={args.autotune} {config_elegida(kernel(causal), (q, k, v))} mma={';'.join(sorted(mma))}"

    banco_attention.ejecutar(args, "helion", preparar, detalle)
