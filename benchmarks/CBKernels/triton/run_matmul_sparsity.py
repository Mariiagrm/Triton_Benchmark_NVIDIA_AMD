"""Experimento: matmul con estructura dispersa 2:4 (cuSPARSELt) vs denso.

La sparsity 2:4 (de cada 4 valores, 2 son cero) es la unica via que puede duplicar el
throughput de los tensor cores en Blackwell y acercarse a la cifra de catalogo (~838
TFLOPS). Triton no soporta 2:4 en tl.dot (necesita metadatos), asi que se usa la ruta de
PyTorch: torch.sparse.to_sparse_semi_structured, que despacha a cuSPARSELt.

Para cada tamano compara (mismas matrices logicas):
    - densa_fp16   : torch.matmul denso (cuBLAS) — referencia
    - sparse24_fp16: A podada a 2:4 y comprimida, A_sp @ B (cuSPARSELt)
    - sparse24_fp8 : idem en FP8 (si cuSPARSELt/torch lo soportan en esta version)

TFLOP/s en convencion "dense-equivalent" (2*M*N*K / t): un 2x sparse aparece como ~2x.
El error se mide como sparse(A_p)@B vs denso(A_p)@B (correccion del camino sparse; la
poda 2:4 en si cambia el resultado y su impacto en exactitud es dependiente de la app).

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul_sparsity [--sizes 4096 8192]
Resultados: results/run_matmul_sparsity/<fecha>_job<JOBID>/ + tabla/grafica automaticas
"""
import argparse

import torch

import comun
from validation import error_rel

try:
    from torch.sparse import to_sparse_semi_structured
    HAY_SPARSE = True
except Exception:
    HAY_SPARSE = False

FP8 = torch.float8_e4m3fn


def podar_2_4(a):
    """Poda a patron 2:4 a lo largo de K: en cada grupo de 4, conserva los 2 de mayor |valor|."""
    M, K = a.shape
    assert K % 4 == 0, "K debe ser multiplo de 4 para 2:4"
    g = a.reshape(M, K // 4, 4)
    idx = g.abs().topk(2, dim=-1).indices
    mask = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, True)
    return (g * mask).reshape(M, K)


def elegir_kernel(kernels):
    """El GEMM entre los kernels lanzados (evita copias/casts elementwise)."""
    gemm = [k for k in kernels if any(t in k.lower() for t in ("gemm", "sparse", "cutlass", "xmma", "mma", "matmul"))]
    cand = gemm or kernels
    return max(cand, key=len) if cand else "?"


def medir_variante(nombre, op, ref, flops):
    """Mide una operacion y captura el kernel CUDA que lanza (evidencia del backend).

    op NO debe incluir casts ni construccion de tensores: solo la multiplicacion, con los
    operandos ya materializados fuera del cronometro (si no, se mide el cast, no el GEMM).
    """
    c = op()
    torch.cuda.synchronize()
    err = error_rel(c, ref)
    principal = elegir_kernel(comun.kernels_cuda(op))
    rt = comun.medir(op, flops=flops)
    print(f"   [{nombre:14s}] {rt['ms']:8.3f} ms  {rt['tflops']:8.2f} TFLOP/s  | err={err:.2%}")
    print(f"                  kernel: {principal[:90]}", flush=True)
    return {"variante": nombre, "M": ref.shape[0], "N": ref.shape[1], "K": None,
            "error_rel": round(err, 5), **rt, "kernel": principal}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192])
    args = parser.parse_args()

    comun.imprimir_contexto()
    if not HAY_SPARSE:
        print("ERROR: torch.sparse.to_sparse_semi_structured no disponible en esta version.")
        return

    torch.manual_seed(0)
    filas = []
    resumen = {}
    for n in args.sizes:
        M = N = K = n
        a32 = torch.randn((M, K), device="cuda", dtype=torch.float32)
        b32 = torch.randn((K, N), device="cuda", dtype=torch.float32)
        a_p = podar_2_4(a32)                 # A podada a 2:4 (denso, con ceros)
        flops = 2 * M * N * K

        print(f"== {M}x{K} @ {K}x{N}")
        # Casts materializados FUERA del cronometro (si no, se mide el cast, no el GEMM).
        a16, a_p16, b16 = a32.half(), a_p.half(), b32.half()
        ref_densa = a16 @ b16                 # referencia denso completo
        ref_podada = a_p16 @ b16             # referencia denso de la matriz podada

        filas.append(medir_variante("densa_fp16", lambda: a16 @ b16, ref_densa, flops))

        for nombre, dtype in [("sparse24_fp16", torch.float16), ("sparse24_fp8", FP8)]:
            try:
                a_sp = to_sparse_semi_structured(a_p16 if dtype == torch.float16 else a_p.to(dtype))
                b = b16 if dtype == torch.float16 else b32.to(dtype)
                fila = medir_variante(nombre, lambda: a_sp @ b, ref_podada, flops)
                filas.append(fila)
            except Exception as e:
                print(f"   [{nombre:14s}] no disponible: {type(e).__name__}: {e}", flush=True)

        densa = next(f for f in filas if f["variante"] == "densa_fp16" and f["M"] == M)
        sp16 = next((f for f in filas if f["variante"] == "sparse24_fp16" and f["M"] == M), None)
        if sp16:
            sp = sp16["tflops"] / densa["tflops"] if densa["tflops"] else float("nan")
            print(f"   -> sparse24_fp16/densa: {sp:.1%}  ({sp16['tflops']} vs {densa['tflops']} TFLOP/s)\n", flush=True)
            resumen[str(n)] = {"densa_tflops": densa["tflops"], "sparse24_fp16_tflops": sp16["tflops"],
                               "speedup": round(sp, 4)}

    comun.guardar(filas, parametros=vars(args), resumen=resumen)


if __name__ == "__main__":
    main()
