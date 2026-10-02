"""Matmul en Helion frente a cuBLAS. Imagen: helion.

Kernel: CBKernels/helion/matmul.py (el `matmul` del ejemplo oficial de Helion). Helion
autotunea cada forma en su 1a llamada (columna t_compilacion_s); --autotune fija el
esfuerzo: none (config por defecto), quick (por defecto aqui) o full (mas lento, mejor).
Banco comun a todos los DSLs: benchmarks/banco_matmul.py.

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul_helion [--autotune full] [--sizes 4096 8192]
Resultados: results/<maquina>/run_matmul_helion/<fecha>_job<JOBID>/ y results/matmul_metrics.csv
"""
import os

import banco_matmul

if __name__ == "__main__":
    p = banco_matmul.parser(__doc__)
    p.add_argument("--autotune", choices=["none", "quick", "full"], default="quick",
                   help="esfuerzo de autotuning de Helion (HELION_AUTOTUNE_EFFORT)")
    args = p.parse_args()
    # Helion lee el esfuerzo al definir el kernel: fijarlo antes de importarlo.
    os.environ["HELION_AUTOTUNE_EFFORT"] = args.autotune
    from CBKernels.helion.matmul import matmul

    def preparar(a, b):
        matmul(a, b)  # autotuning de esta forma (static_shapes=True)
        return lambda: matmul(a, b)

    banco_matmul.ejecutar(args, "helion", preparar, lambda a, b, fn: f"autotune={args.autotune}")
