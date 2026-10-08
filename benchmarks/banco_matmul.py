"""Banco COMUN de matmul: mismas formas, validacion, medida y guardado para todos los DSLs.

Cada benchmarks/CBKernels/<dsl>/run_matmul_<dsl>.py solo aporta como preparar su kernel;
asi la comparacion entre DSLs es justa (misma metodologia que el resto del TFM) y todas
las ejecuciones acaban en results/matmul_metrics.csv. Por cada tamano y dtype se mide el
kernel del DSL y, como referencia, torch.matmul (cuBLAS), en TFLOP/s.

    preparar(a, b) -> fn      compila/autotunea (se cronometra aparte: t_compilacion_s);
                              fn() lanza solo el kernel y devuelve C = A @ B
"""
import argparse
import time

import torch

import comun
import validation


def parser(descripcion, dtypes=("fp16",)):
    p = argparse.ArgumentParser(description=descripcion, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192], help="M = N = K")
    p.add_argument("--dtypes", nargs="+", choices=list(dtypes), default=list(dtypes))
    return p


def ejecutar(args, dsl, preparar, detalle=None):
    """Mide el kernel de `preparar` frente a cuBLAS y guarda los resultados.

    detalle(a, b, fn): texto opcional por forma (configuracion elegida, etc.), que se
    guarda en la columna 'detalle'.
    """
    comun.imprimir_contexto()
    torch.manual_seed(0)

    filas = []
    resumen = {}
    for dt in args.dtypes:
        dtype = comun.DTYPES[dt]
        for n in args.sizes:
            M = N = K = n
            a = torch.randn((M, K), device="cuda", dtype=dtype)
            b = torch.randn((K, N), device="cuda", dtype=dtype)
            flops = 2 * M * N * K
            bytes_movidos = (M * K + K * N + M * N) * a.element_size()

            # Compilacion / autotuning + 1a ejecucion: fuera de la medida.
            t0 = time.perf_counter()
            fn = preparar(a, b)
            c = fn()
            torch.cuda.synchronize()
            t_compilacion = round(time.perf_counter() - t0, 2)
            validation.comprobar_matmul(c, a, b, f"matmul {dsl} {dt} {n}")
            info = detalle(a, b, fn) if detalle else ""
            cublas = comun.kernels_cuda(lambda: torch.matmul(a, b))

            print(f"== {M}x{N}x{K} ({dt}): 1a llamada {dsl} {t_compilacion} s {info}")
            tflops = {}
            for nombre, f in {dsl: fn, "cublas": lambda: torch.matmul(a, b)}.items():
                r = comun.medir(f, flops=flops, bytes_movidos=bytes_movidos)
                tflops[nombre] = r["tflops"]
                print(f"   [{nombre:7s}] {r['ms']:8.3f} ms  {r['tflops']:7.2f} TFLOP/s", flush=True)
                filas.append({"variante": nombre, "dtype": dt, "M": M, "N": N, "K": K, **r,
                              "t_compilacion_s": t_compilacion if nombre == dsl else "",
                              "detalle": info if nombre == dsl else ";".join(cublas)})
            sp = tflops[dsl] / tflops["cublas"]
            print(f"   -> {dsl}/cuBLAS: {sp:.1%}\n", flush=True)
            resumen[f"{dt}_{n}"] = {**{f"{k}_tflops": v for k, v in tflops.items()}, f"{dsl}_vs_cublas": round(sp, 4)}

    comun.guardar(filas, parametros={**vars(args), "dsl": dsl}, resumen=resumen)
