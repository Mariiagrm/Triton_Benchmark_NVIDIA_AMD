"""RMSNorm de CUTLASS frente a PyTorch (operador memory-bound). Imagen: cutlass.

Kernel: cutlass::rmsnorm oficial (tools/util/include/cutlass/util/device_rmsnorm.h),
compilado como extension de PyTorch (MBKernels/cutlass/rmsnorm.py): en fp16 usa el kernel
vectorizado con float4 (_e8) y en bf16 el escalar (_e1); la columna 'detalle' lo indica.
Banco comun a todos los DSLs: benchmarks/banco_rmsnorm.py.

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_rmsnorm_cutlass [--rows ...] [--cols ...] [--dtypes ...]
Resultados: results/run_rmsnorm_cutlass/<fecha>_job<JOBID>/ y results/rmsnorm_metrics.csv
"""
import banco_rmsnorm
from MBKernels.cutlass.rmsnorm import rmsnorm, ruta

if __name__ == "__main__":
    args = banco_rmsnorm.parser(__doc__).parse_args()
    banco_rmsnorm.ejecutar(args, "cutlass", rmsnorm, lambda x, w: ruta(x))
