"""Experimento: matmul baseline vs block-pointers vs descriptores TMA.

Compara tres kernels Triton sobre la misma metodologia para aislar el efecto de la
estrategia de carga (evidencia para la memoria del TFM):
    - baseline : aritmetica de punteros manual (CBKernels/triton/matmul.py)
    - blockptr : tl.make_block_ptr (INTENTO de TMA; en sm_121 se queda en cp.async)
    - tma      : tl.make_tensor_descriptor (TMA real: cp.async.bulk.tensor)

Para cada dtype y tamano ejecuta las tres variantes, valida contra cuBLAS
(torch.matmul), exige tensor cores y reporta:
    - TFLOP/s y GB/s de cada una (comun.medir)
    - la estrategia de carga leida del PTX (TMA / cp.async / sincrona)
    - el speedup de cada variante frente al baseline

Uso:
    sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp run_matmul_tma [--sizes 4096 8192] [--dtypes fp16 bf16]
Resultados: results/run_matmul_tma/<fecha>_job<JOBID>/{resultados.csv,meta.json}
"""
import argparse
import time

import torch

import comun
import validation
from CBKernels.triton.matmul import matmul, matmul_kernel
from CBKernels.triton.matmul_blockptr import matmul_blockptr, matmul_blockptr_kernel
from CBKernels.triton.matmul_tma import matmul_tma, matmul_tma_kernel

# Los kernels a comparar: etiqueta -> (funcion, kernel autotuneable). Orden = orden de salida.
VARIANTES = {
    "baseline": (matmul, matmul_kernel),
    "blockptr": (matmul_blockptr, matmul_blockptr_kernel),
    "tma": (matmul_tma, matmul_tma_kernel),
}


def evaluar(nombre, fn, kernel_at, a, b, dt, n, flops, bytes_movidos):
    """Autotunea, valida y mide una variante. Devuelve una fila de resultados."""
    M, K = a.shape
    _, N = b.shape

    t0 = time.perf_counter()
    c, kernel = fn(a, b, devolver_kernel=True)
    torch.cuda.synchronize()
    t_autotune = time.perf_counter() - t0

    mma = comun.exigir_tensor_cores(kernel, f"{nombre} {dt} {n}")
    carga = comun.copias_asincronas(kernel)
    validation.comprobar_matmul(c, a, b, f"{nombre} {dt} {n}")

    cfg = kernel_at.best_config
    rt = comun.medir(lambda: fn(a, b), flops=flops, bytes_movidos=bytes_movidos)

    print(f"   [{nombre:8s}] {rt['ms']:8.3f} ms  {rt['tflops']:7.2f} TFLOP/s  {rt['gbs']:8.1f} GB/s"
          f"  | carga={carga['carga']}  | autotune {t_autotune:.1f} s")
    print(f"              config={cfg.kwargs}, num_warps={cfg.num_warps}, num_stages={cfg.num_stages}")
    print(f"              MMA={', '.join(mma)}", flush=True)

    return {
        "variante": nombre, "dtype": dt, "M": M, "N": N, "K": K,
        **cfg.kwargs, "num_warps": cfg.num_warps, "num_stages": cfg.num_stages,
        "carga": carga["carga"], "usa_tma": bool(carga["tma"]),
        "autotune_s": round(t_autotune, 1),
        **rt,
        "mma": ";".join(mma),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192], help="tamanos N (matrices N x N)")
    parser.add_argument("--dtypes", nargs="+", choices=list(comun.DTYPES), default=["fp16", "bf16"])
    args = parser.parse_args()

    comun.imprimir_contexto()
    print("Autotuning: " + " | ".join(f"{n} {len(k.configs)} configs"
                                      for n, (_, k) in VARIANTES.items()) + "\n")

    torch.manual_seed(0)
    filas = []
    resumen = {}
    for dt in args.dtypes:
        dtype = comun.DTYPES[dt]
        elem_bytes = torch.tensor([], dtype=dtype).element_size()
        for n in args.sizes:
            M = N = K = n
            a = torch.randn((M, K), device="cuda", dtype=dtype)
            b = torch.randn((K, N), device="cuda", dtype=dtype)
            flops = 2 * M * N * K
            bytes_movidos = (M * K + K * N + M * N) * elem_bytes

            print(f"== {M}x{K} @ {K}x{N} ({dt})")
            fila_por_variante = {}
            for nombre, (fn, kernel_at) in VARIANTES.items():
                fila = evaluar(nombre, fn, kernel_at, a, b, dt, n, flops, bytes_movidos)
                fila_por_variante[nombre] = fila
                filas.append(fila)

            tf_base = fila_por_variante["baseline"]["tflops"]
            speedups = {}
            for nombre, fila in fila_por_variante.items():
                if nombre == "baseline":
                    continue
                sp = fila["tflops"] / tf_base if tf_base else float("nan")
                speedups[nombre] = round(sp, 4)
                print(f"   -> {nombre:8s} vs baseline: {sp:6.1%}  "
                      f"({fila['tflops']} vs {tf_base} TFLOP/s)  carga={fila['carga']}")
            print(flush=True)
            resumen[f"{dt}_{n}"] = {
                "baseline_tflops": tf_base,
                **{f"{n2}_tflops": f["tflops"] for n2, f in fila_por_variante.items() if n2 != "baseline"},
                **{f"{n2}_carga": f["carga"] for n2, f in fila_por_variante.items() if n2 != "baseline"},
                "speedups": speedups,
            }

    comun.guardar(filas, parametros=vars(args), resumen=resumen)


if __name__ == "__main__":
    main()
