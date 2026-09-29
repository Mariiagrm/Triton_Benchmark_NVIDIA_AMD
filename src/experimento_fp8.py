"""Experimento: matmul FP8 (Triton) vs FP16 baseline — ¿rompe el techo de ~100 TFLOP/s?

docs/tma.md concluye que el matmul denso fp16 esta limitado por la unidad MMA de sm_121,
no por la memoria. La via para subir ese techo es bajar la precision: los tensor cores
de Blackwell ejecutan FP8 al doble de throughput que FP16. Este experimento mide si el
kernel FP8 lo consigue y a que coste de precision.

Para cada tamano compara:
    - fp16      : baseline Triton (triton_kernels/matmul.py)
    - fp8       : kernel Triton FP8 e4m3 (triton_kernels/matmul_fp8.py)
    - fp8_cublas: cuBLASLt via torch._scaled_mm (si esta disponible en esta version)

Reporta TFLOP/s, GB/s, la estrategia de carga y MMA (del PTX) y el ERROR RELATIVO de
cada variante frente al resultado exacto en FP32 (coste de precision, clave para la memoria).

Uso:
    sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp experimento_fp8 [--sizes 4096 8192]
Resultados: results/experimento_fp8/<fecha>_job<JOBID>/{resultados.csv,meta.json} + tabla/grafica
"""
import argparse

import torch

import comun
from triton_kernels.matmul import matmul, matmul_kernel
from triton_kernels.matmul_fp8 import matmul_fp8, matmul_fp8_kernel
from triton_kernels.matmul_fp8_tma import matmul_fp8_tma, matmul_fp8_tma_kernel

FP8 = torch.float8_e4m3fn


def error_rel(c, ref):
    """Error relativo de Frobenius frente al resultado exacto (fp32)."""
    c = c.float()
    return (torch.linalg.norm(c - ref) / torch.linalg.norm(ref)).item()


def fila_triton(nombre, fn, kernel_at, a, b, ref, flops, bytes_movidos):
    """Autotunea, valida (error rel vs fp32), exige tensor cores y mide un kernel Triton."""
    M, K = a.shape
    _, N = b.shape
    c, kernel = fn(a, b, devolver_kernel=True)
    torch.cuda.synchronize()
    mma = comun.exigir_tensor_cores(kernel, f"{nombre}")
    carga = comun.copias_asincronas(kernel)
    err = error_rel(c, ref)
    cfg = kernel_at.best_config
    rt = comun.medir(lambda: fn(a, b), flops=flops, bytes_movidos=bytes_movidos)
    print(f"   [{nombre:11s}] {rt['ms']:8.3f} ms  {rt['tflops']:8.2f} TFLOP/s  {rt['gbs']:8.1f} GB/s"
          f"  | err={err:.2%}  carga={carga['carga']}")
    print(f"               config={cfg.kwargs}, warps={cfg.num_warps}, stages={cfg.num_stages}  MMA={';'.join(mma)}",
          flush=True)
    return {"variante": nombre, "M": M, "N": N, "K": K,
            **cfg.kwargs, "num_warps": cfg.num_warps, "num_stages": cfg.num_stages,
            "carga": carga["carga"], "error_rel": round(err, 5), **rt, "mma": ";".join(mma)}


def fila_cublas_fp8(a8, b8, ref, flops, bytes_movidos):
    """cuBLASLt FP8 via torch._scaled_mm (requiere b en column-major). None si no aplica."""
    M, K = a8.shape
    _, N = b8.shape
    s = torch.tensor(1.0, device="cuda")
    b_col = b8.t().contiguous().t()  # KxN en column-major, como pide cuBLASLt fp8
    try:
        op = lambda: torch._scaled_mm(a8, b_col, scale_a=s, scale_b=s, out_dtype=torch.float16)
        c = op()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"   [fp8_cublas ] no disponible: {type(e).__name__}: {e}", flush=True)
        return None
    err = error_rel(c, ref)
    rt = comun.medir(op, flops=flops, bytes_movidos=bytes_movidos)
    print(f"   [fp8_cublas ] {rt['ms']:8.3f} ms  {rt['tflops']:8.2f} TFLOP/s  {rt['gbs']:8.1f} GB/s  | err={err:.2%}",
          flush=True)
    return {"variante": "fp8_cublas", "M": M, "N": N, "K": K,
            "carga": "cublasLt", "error_rel": round(err, 5), **rt, "mma": "cublasLt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192], help="tamanos N (matrices N x N)")
    args = parser.parse_args()

    comun.imprimir_contexto()
    torch.manual_seed(0)
    filas = []
    resumen = {}
    for n in args.sizes:
        M = N = K = n
        # Referencia exacta en fp32 a partir de las MISMAS matrices logicas.
        a32 = torch.randn((M, K), device="cuda", dtype=torch.float32)
        b32 = torch.randn((K, N), device="cuda", dtype=torch.float32)
        ref = a32 @ b32

        a16, b16 = a32.half(), b32.half()
        a8, b8 = a32.to(FP8), b32.to(FP8)
        flops = 2 * M * N * K
        bytes_fp16 = (M * K + K * N + M * N) * 2
        bytes_fp8 = (M * K + K * N) * 1 + M * N * 2  # entradas fp8 (1B), salida fp16 (2B)

        print(f"== {M}x{K} @ {K}x{N}")
        f16 = fila_triton("fp16", matmul, matmul_kernel, a16, b16, ref, flops, bytes_fp16)
        f8 = fila_triton("fp8", matmul_fp8, matmul_fp8_kernel, a8, b8, ref, flops, bytes_fp8)
        f8t = fila_triton("fp8_tma", matmul_fp8_tma, matmul_fp8_tma_kernel, a8, b8, ref, flops, bytes_fp8)
        filas += [f16, f8, f8t]
        f8c = fila_cublas_fp8(a8, b8, ref, flops, bytes_fp8)
        if f8c:
            filas.append(f8c)

        base = f8["tflops"] or float("nan")
        cublas = f8c["tflops"] if f8c else None
        print(f"   -> fp8/fp16: {f8['tflops'] / f16['tflops']:.1%}  |  "
              f"fp8_tma/fp8: {f8t['tflops'] / base:.1%}"
              + (f"  |  fp8_tma vs cuBLAS: {f8t['tflops'] / cublas:.1%}" if cublas else "")
              + f"  (fp16 {f16['tflops']}, fp8 {f8['tflops']}, fp8_tma {f8t['tflops']}"
              + (f", cuBLAS {cublas}" if cublas else "") + " TFLOP/s)\n", flush=True)
        resumen[str(n)] = {
            "fp16_tflops": f16["tflops"], "fp8_tflops": f8["tflops"], "fp8_tma_tflops": f8t["tflops"],
            **({"fp8_cublas_tflops": cublas} if cublas else {}),
            "fp8_tma_carga": f8t["carga"], "fp8_error_rel": f8["error_rel"],
        }

    comun.guardar(filas, parametros=vars(args), resumen=resumen)


if __name__ == "__main__":
    main()
