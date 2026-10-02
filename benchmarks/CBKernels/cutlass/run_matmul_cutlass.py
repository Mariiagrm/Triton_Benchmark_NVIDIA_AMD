"""Matmul de CUTLASS (CuTe DSL) frente a cuBLAS. Imagen: cutlass. Solo fp16.

Kernel: el ejemplo oficial dense_gemm.py de CUTLASS para Blackwell GeForce (Sm120GemmKernel),
sin modificar, cargado por CBKernels/cutlass/matmul_cute.py. --tile fija el bloque CTA
(M, N, K); por defecto el del ejemplo, 128 128 64. La 1a llamada incluye cute.compile.
Banco comun a todos los DSLs: benchmarks/banco_matmul.py.

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul_cutlass [--tile 128 256 64] [--sizes 4096 8192]
Resultados: results/<maquina>/run_matmul_cutlass/<fecha>_job<JOBID>/ y results/matmul_metrics.csv
"""
import banco_matmul
from CBKernels.cutlass.matmul_cute import TILE_POR_DEFECTO, preparar

if __name__ == "__main__":
    p = banco_matmul.parser(__doc__, dtypes=("fp16",))
    p.add_argument("--tile", type=int, nargs=3, default=list(TILE_POR_DEFECTO), metavar=("M", "N", "K"))
    args = p.parse_args()
    banco_matmul.ejecutar(args, "cutlass", lambda a, b: preparar(a, b, tile=args.tile),
                          lambda a, b, fn: "Sm120GemmKernel tile=" + "x".join(map(str, args.tile)))
