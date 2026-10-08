"""RMSNorm en Gluon frente a PyTorch (operador memory-bound). Imagen: gluon.

Kernel: MBKernels/gluon/rmsnorm.py (mismo algoritmo que el de Triton, con el layout de
registros explicito). Banco comun a todos los DSLs: benchmarks/banco_rmsnorm.py.

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_rmsnorm_gluon [--rows ...] [--cols ...] [--dtypes ...]
Resultados: results/run_rmsnorm_gluon/<fecha>_job<JOBID>/ y results/rmsnorm_metrics.csv
"""
import banco_rmsnorm
from MBKernels.gluon.rmsnorm import configuracion, rmsnorm


def detalle(x, w):
    _, num_warps, layout = configuracion(x.shape[1], x.element_size())
    return f"num_warps={num_warps} vec={layout.size_per_thread[0]}"


if __name__ == "__main__":
    args = banco_rmsnorm.parser(__doc__).parse_args()
    banco_rmsnorm.ejecutar(args, "gluon", rmsnorm, detalle)
