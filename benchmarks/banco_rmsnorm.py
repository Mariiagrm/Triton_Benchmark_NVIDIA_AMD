"""Banco COMUN de RMSNorm: mismas formas, validacion, medida y guardado para todos los DSLs.

Cada benchmarks/MBKernels/<dsl>/run_rmsnorm_<dsl>.py solo aporta su kernel; asi la
comparacion entre DSLs es justa (misma metodologia) y todas las ejecuciones acaban en
results/rmsnorm_metrics.csv. Por cada forma y dtype se mide el kernel del DSL y, como
referencia, torch.nn.functional.rms_norm, en GB/s efectivos y % del pico.

RMSNorm lee x (M x N) y w (N) y escribe y (M x N): es memory-bound. No hay producto
matricial, asi que no se exigen tensor cores. Las formas por defecto son de LLM
(M = tokens, N = dimension oculta) y superan la L2, para medir la memoria principal.
"""
import argparse
import time

import torch
import torch.nn.functional as F

import comun
import validation


def parser(descripcion):
    p = argparse.ArgumentParser(description=descripcion, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, nargs="+", default=[4096, 16384], help="M: filas (tokens)")
    p.add_argument("--cols", type=int, nargs="+", default=[2048, 4096, 8192], help="N: dimension oculta")
    p.add_argument("--dtypes", nargs="+", choices=["fp16", "bf16"], default=["fp16", "bf16"])
    p.add_argument("--eps", type=float, default=1e-5)
    # GB10: LPDDR5X de 256 bits a 8533 MT/s -> ~273 GB/s nominales. Cambialo para otra GPU.
    p.add_argument("--pico-gbs", type=float, default=273.0, help="ancho de banda pico de la GPU (GB/s)")
    return p


def ejecutar(args, dsl, kernel, detalle=None):
    """Mide kernel(x, w, eps) -> y frente a PyTorch y guarda los resultados.

    detalle(x, w): texto opcional por forma (p. ej. el kernel CUTLASS elegido), que se
    guarda en la columna 'detalle'.
    """
    comun.imprimir_contexto()
    torch.manual_seed(0)

    filas = []
    resumen = {}
    for dt in args.dtypes:
        dtype = comun.DTYPES[dt]
        for M in args.rows:
            for N in args.cols:
                x = torch.randn((M, N), device="cuda", dtype=dtype)
                w = torch.randn((N,), device="cuda", dtype=dtype)
                bytes_movidos = (2 * M * N + N) * x.element_size()  # leer x y w, escribir y

                # 1a llamada: compila (y en Helion autotunea); se mide aparte, fuera del banco.
                t0 = time.perf_counter()
                y = kernel(x, w, args.eps)
                torch.cuda.synchronize()
                t_compilacion = round(time.perf_counter() - t0, 2)
                validation.comprobar_rmsnorm(y, x, w, args.eps, f"rmsnorm {dsl} {dt} {M}x{N}")
                info = detalle(x, w) if detalle else ""

                print(f"== {M}x{N} ({dt}), {bytes_movidos / 1e6:.0f} MB movidos, "
                      f"1a llamada {dsl} {t_compilacion} s {info}")
                gbs = {}
                for nombre, fn in {dsl: lambda: kernel(x, w, args.eps),
                                   "torch": lambda: F.rms_norm(x, (N,), w, args.eps)}.items():
                    r = comun.medir(fn, bytes_movidos=bytes_movidos)
                    pico = r["gbs"] / args.pico_gbs
                    gbs[nombre] = r["gbs"]
                    print(f"   [{nombre:7s}] {r['ms']:8.4f} ms  {r['gbs']:7.1f} GB/s  ({pico:.0%} del pico)", flush=True)
                    filas.append({"variante": nombre, "dtype": dt, "M": M, "N": N, **r,
                                  "pct_pico": round(100 * pico, 1),
                                  "t_compilacion_s": t_compilacion if nombre == dsl else "",
                                  "detalle": info if nombre == dsl else ""})
                sp = gbs[dsl] / gbs["torch"]
                print(f"   -> {dsl}/torch: {sp:.1%}\n", flush=True)
                resumen[f"{dt}_{M}x{N}"] = {**{f"{k}_gbs": v for k, v in gbs.items()},
                                            f"{dsl}_vs_torch": round(sp, 4)}

    comun.guardar(filas, parametros={**vars(args), "dsl": dsl}, resumen=resumen)
