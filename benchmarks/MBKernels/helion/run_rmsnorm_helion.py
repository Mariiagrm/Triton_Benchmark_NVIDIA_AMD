"""RMSNorm en Helion frente a PyTorch (operador memory-bound). Imagen: helion.

Kernel: MBKernels/helion/rmsnorm.py (ejemplo oficial de Helion, solo forward). Helion
autotunea cada forma en su 1a llamada (columna t_compilacion_s); --autotune fija el
esfuerzo: none (config por defecto), quick (por defecto aqui) o full (mas lento, mejor).
Banco comun a todos los DSLs: benchmarks/banco_rmsnorm.py.

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_rmsnorm_helion [--autotune full] [--rows ...]
Resultados: results/run_rmsnorm_helion/<fecha>_job<JOBID>/ y results/rmsnorm_metrics.csv
"""
import os

import banco_rmsnorm

if __name__ == "__main__":
    p = banco_rmsnorm.parser(__doc__)
    p.add_argument("--autotune", choices=["none", "quick", "full"], default="quick",
                   help="esfuerzo de autotuning de Helion (HELION_AUTOTUNE_EFFORT)")
    args = p.parse_args()
    # Helion lee el esfuerzo al definir el kernel: fijarlo antes de importarlo.
    os.environ["HELION_AUTOTUNE_EFFORT"] = args.autotune
    from MBKernels.helion.rmsnorm import rmsnorm

    banco_rmsnorm.ejecutar(args, "helion", rmsnorm, lambda x, w: f"autotune={args.autotune}")
