"""Baseline de rendimiento: matmul Triton con @triton.autotune vs cuBLAS (torch.matmul).

Para cada dtype y tamano se lanza el autotuning (1a llamada), se valida contra
cuBLAS y se mide la mejor configuracion con comun.medir(). El "baseline" de cada
dtype es la ejecucion Triton con mas TFLOP/s: el rival a batir por los kernels del TFM.

Ambos lados usan tensor cores: se aborta si el kernel Triton elegido no contiene MMA
en su PTX, y se registra el kernel de cuBLAS (fp32 va en TF32, ver comun.py).

Uso:
    sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp run_matmul [--sizes 4096 8192] [--dtypes fp16 bf16]
Resultados: results/run_matmul/<fecha>_job<JOBID>/{resultados.csv,meta.json}
           + tabla.md/.tex y grafica.png/.pdf (ver informe_matmul.py)
"""
import argparse
import time

import torch

import comun
import validation
import informe_matmul
from CBKernels.triton.matmul import matmul, matmul_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192], help="tamanos N (matrices N x N)")
    parser.add_argument("--dtypes", nargs="+", choices=list(comun.DTYPES), default=["fp16", "bf16"])
    args = parser.parse_args()

    comun.imprimir_contexto()
    print(f"Espacio de autotuning: {len(matmul_kernel.configs)} configuraciones\n")

    torch.manual_seed(0)
    filas = []
    baseline = {}
    for dt in args.dtypes:
        dtype = comun.DTYPES[dt]
        elem_bytes = torch.tensor([], dtype=dtype).element_size()
        filas_dt = []
        for n in args.sizes:
            M = N = K = n
            a = torch.randn((M, K), device="cuda", dtype=dtype)
            b = torch.randn((K, N), device="cuda", dtype=dtype)

            # 1a llamada: dispara el autotuning (compila y mide todas las configuraciones).
            t0 = time.perf_counter()
            c, kernel = matmul(a, b, devolver_kernel=True)
            torch.cuda.synchronize()
            t_autotune = time.perf_counter() - t0
            mma = comun.exigir_tensor_cores(kernel, f"matmul Triton {dt} {n}")
            cublas_kernels = comun.kernels_cuda(lambda: torch.matmul(a, b))

            validation.comprobar_matmul(c, a, b, f"matmul Triton {dt} {n}")

            cfg = matmul_kernel.best_config
            flops = 2 * M * N * K
            bytes_movidos = (M * K + K * N + M * N) * elem_bytes
            rt = comun.medir(lambda: matmul(a, b), flops=flops, bytes_movidos=bytes_movidos)
            rc = comun.medir(lambda: torch.matmul(a, b), flops=flops, bytes_movidos=bytes_movidos)

            print(f"== {M}x{K} @ {K}x{N} ({dt}) — autotuning {t_autotune:.1f} s")
            print(f"   Mejor config: {cfg.kwargs}, num_warps={cfg.num_warps}, num_stages={cfg.num_stages}")
            print(f"   Tensor cores Triton: {', '.join(mma)}")
            print(f"   Kernel cuBLAS: {'; '.join(cublas_kernels)}")
            print(f"   Triton : {rt['ms']:8.3f} ms  {rt['tflops']:7.2f} TFLOP/s  {rt['gbs']:8.1f} GB/s")
            print(f"   cuBLAS : {rc['ms']:8.3f} ms  {rc['tflops']:7.2f} TFLOP/s  {rc['gbs']:8.1f} GB/s")
            print(f"   Triton/cuBLAS: {rt['tflops'] / rc['tflops']:.1%}\n", flush=True)

            filas_dt.append({
                "dtype": dt, "M": M, "N": N, "K": K,
                **cfg.kwargs, "num_warps": cfg.num_warps, "num_stages": cfg.num_stages,
                "autotune_s": round(t_autotune, 1),
                **{f"triton_{k}": v for k, v in rt.items()},
                **{f"cublas_{k}": v for k, v in rc.items()},
                "triton_mma": ";".join(mma), "cublas_kernel": ";".join(cublas_kernels),
            })

        mejor = max(filas_dt, key=lambda f: f["triton_tflops"])
        baseline[dt] = mejor
        print(f"*** BASELINE {dt} (rival a batir): {mejor['triton_tflops']} TFLOP/s "
              f"en {mejor['M']}x{mejor['N']}x{mejor['K']} (cuBLAS: {mejor['cublas_tflops']} TFLOP/s) ***\n")
        filas += filas_dt

    # informe propio (grafica Triton vs cuBLAS); comun.guardar lo llama y captura errores.
    comun.guardar(filas, parametros=vars(args), resumen={"baseline": baseline},
                  informe=informe_matmul.generar)


if __name__ == "__main__":
    main()
