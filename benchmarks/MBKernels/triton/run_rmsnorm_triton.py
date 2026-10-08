"""RMSNorm en Triton frente a PyTorch (operador memory-bound).

Kernel: MBKernels/triton/rmsnorm_baseline.py (un programa por fila, cargas por
punteros). Banco comun a todos los DSLs: benchmarks/banco_rmsnorm.py.

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_rmsnorm_triton [--rows ...] [--cols ...] [--dtypes ...]
Resultados: results/run_rmsnorm_triton/<fecha>_job<JOBID>/ y results/rmsnorm_metrics.csv
"""
import banco_rmsnorm
from MBKernels.triton.rmsnorm_baseline import rmsnorm, num_warps_para

import triton


def detalle(x, w):
    return f"num_warps={num_warps_para(triton.next_power_of_2(x.shape[1]))}"


if __name__ == "__main__":
    args = banco_rmsnorm.parser(__doc__).parse_args()
    banco_rmsnorm.ejecutar(args, "triton", rmsnorm, detalle)
