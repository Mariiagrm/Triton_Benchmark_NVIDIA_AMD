"""Matmul en Gluon frente a cuBLAS. Imagen: gluon.

Kernel: CBKernels/gluon/matmul.py (propio: mma_v2 + pipeline cp.async; Gluon no tiene un
matmul oficial para sm_120/121). Gluon no trae autotuning: en la 1a llamada de cada forma
se miden las configuraciones de CONFIGS (bloque, etapas, warps) y se usa la mas rapida
(columna 'detalle'; el tiempo total va en t_compilacion_s). Se exige que el PTX use tensor
cores (mma.sync). Banco comun a todos los DSLs: benchmarks/banco_matmul.py.

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul_gluon [--sizes 4096 8192] [--dtypes fp16]
Resultados: results/<maquina>/run_matmul_gluon/<fecha>_job<JOBID>/ y results/matmul_metrics.csv
"""
import banco_matmul
import comun
from CBKernels.gluon.matmul import CONFIGS, matmul

elegida = {}


def preparar(a, b):
    """Prueba cada configuracion (las que no compilan se descartan) y fija la mas rapida."""
    tiempos = {}
    for cfg in CONFIGS:
        try:
            _, kernel = matmul(a, b, cfg, devolver_kernel=True)
            comun.exigir_tensor_cores(kernel, f"matmul Gluon {cfg}")
            tiempos[cfg] = comun.medir(lambda: matmul(a, b, cfg), warmup=25, rep=100)["ms"]
        except Exception as e:  # p. ej. memoria compartida insuficiente
            print(f"   config {cfg} descartada: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
    if not tiempos:
        raise RuntimeError("ninguna configuracion de Gluon ha compilado")
    mejor = min(tiempos, key=tiempos.get)
    _, kernel = matmul(a, b, mejor, devolver_kernel=True)
    elegida[a.shape] = (mejor, ";".join(comun.instrucciones_tensor_core(kernel)))
    return lambda: matmul(a, b, mejor)


def detalle(a, b, fn):
    (BM, BN, BK, ST, WM, WN), mma = elegida[a.shape]
    return f"BM={BM} BN={BN} BK={BK} stages={ST} warps={WM}x{WN} mma={mma}"


if __name__ == "__main__":
    args = banco_matmul.parser(__doc__).parse_args()
    banco_matmul.ejecutar(args, "gluon", preparar, detalle)
