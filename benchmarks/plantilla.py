"""Experimento cero: GEMM minimo en tensor cores (Triton, configuracion fija) vs cuBLAS.

Sirve de PLANTILLA para el resto de experimentos: argumentos por linea de comandos,
kernels a nivel de modulo, validacion contra la referencia, comprobacion de que el
kernel usa TENSOR CORES (comun.exigir_tensor_cores), medida con comun.medir() y
guardado con comun.guardar().

Uso:
    sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp plantilla [--sizes ...] [--dtypes ...]
Resultados: results/plantilla/<fecha>_job<JOBID>/{resultados.csv,meta.json}
"""
import argparse

import torch
import triton
import triton.language as tl

import comun
import validation


# Los kernels van a nivel de modulo: @triton.jit necesita leer el fuente del fichero.
# tl.dot es lo que lleva el calculo a los tensor cores (mma.sync en sm_121).
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Matrices contiguas por filas: A (M x K), B (K x N), C (M x N).
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :],
                    mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :],
                    mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm(a, b, bm=128, bn=128, bk=32):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    kernel = gemm_kernel[(triton.cdiv(M, bm), triton.cdiv(N, bn))](a, b, c, M, N, K, BM=bm, BN=bn, BK=bk,
                                                                  num_warps=4, num_stages=3)
    return c, kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096], help="tamanos N (matrices N x N)")
    parser.add_argument("--dtypes", nargs="+", choices=list(comun.DTYPES), default=["fp16", "bf16", "tf32"])
    args = parser.parse_args()

    comun.imprimir_contexto()
    torch.manual_seed(0)

    filas = []
    for dt in args.dtypes:
        for n in args.sizes:
            a = torch.randn((n, n), device="cuda", dtype=comun.DTYPES[dt])
            b = torch.randn((n, n), device="cuda", dtype=comun.DTYPES[dt])
            c, kernel = gemm(a, b)
            mma = comun.exigir_tensor_cores(kernel, f"gemm Triton {dt}")
            validation.comprobar_matmul(c, a, b, f"gemm Triton {dt}")

            for variante, fn in {"triton": lambda: gemm(a, b), "cublas": lambda: torch.matmul(a, b)}.items():
                r = comun.medir(fn, flops=2 * n ** 3)
                print(f"{dt} {n:>5}³ {variante:<7} {r['ms']:9.4f} ms  {r['tflops']:7.2f} TFLOP/s", flush=True)
                filas.append({"variante": variante, "dtype": dt, "n": n, **r,
                              "tensor_cores": ";".join(mma) if variante == "triton" else "cuBLAS"})
            print(f"   tensor cores: {', '.join(mma)}\n", flush=True)

    mejor = max((f for f in filas if f["variante"] == "triton"), key=lambda f: f["tflops"])
    print(f"*** Mejor Triton: {mejor['tflops']} TFLOP/s ({mejor['dtype']}, n={mejor['n']}) ***")
    comun.guardar(filas, parametros=vars(args), resumen={"mejor_triton": mejor})


if __name__ == "__main__":
    main()
