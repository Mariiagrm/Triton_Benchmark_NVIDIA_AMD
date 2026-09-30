"""Experimento: RMSNorm en Triton frente a PyTorch (operador memory-bound).

RMSNorm lee x (M x N) y w (N), y escribe y (M x N): su rendimiento es ancho de
banda de memoria. Se mide en GB/s efectivos y en % del pico de la GPU.

Variantes:
    - triton : MBKernels/triton/rmsnorm_baseline.py (una fila por programa)
    - torch  : torch.nn.functional.rms_norm (kernel de PyTorch)

No hay producto matricial, asi que no se exige tensor cores (ver el kernel).
Las formas por defecto son de LLM (M = tokens, N = dimension oculta) y superan la
L2, para medir la memoria principal y no la cache.

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_rmsnorm [--rows ...] [--cols ...] [--dtypes ...]
Resultados: results/run_rmsnorm/<fecha>_job<JOBID>/{resultados.csv,meta.json}
"""
import argparse

import torch
import torch.nn.functional as F

import comun
import validation
from MBKernels.triton.rmsnorm_baseline import rmsnorm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[4096, 16384], help="M: filas (tokens)")
    parser.add_argument("--cols", type=int, nargs="+", default=[2048, 4096, 8192], help="N: dimension oculta")
    parser.add_argument("--dtypes", nargs="+", choices=["fp16", "bf16"], default=["fp16", "bf16"])
    parser.add_argument("--eps", type=float, default=1e-5)
    # GB10: LPDDR5X de 256 bits a 8533 MT/s -> ~273 GB/s nominales. Cambialo para otra GPU.
    parser.add_argument("--pico-gbs", type=float, default=273.0, help="ancho de banda pico de la GPU (GB/s)")
    args = parser.parse_args()

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

                y, kernel = rmsnorm(x, w, args.eps, devolver_kernel=True)
                validation.comprobar_rmsnorm(y, x, w, args.eps, f"rmsnorm Triton {dt} {M}x{N}")

                variantes = {
                    "triton": lambda: rmsnorm(x, w, args.eps),
                    "torch": lambda: F.rms_norm(x, (N,), w, args.eps),
                }
                print(f"== {M}x{N} ({dt}), {bytes_movidos / 1e6:.0f} MB movidos, "
                      f"num_warps Triton = {kernel.metadata.num_warps}")
                gbs = {}
                for nombre, fn in variantes.items():
                    r = comun.medir(fn, bytes_movidos=bytes_movidos)
                    pico = r["gbs"] / args.pico_gbs
                    gbs[nombre] = r["gbs"]
                    print(f"   [{nombre:6s}] {r['ms']:8.4f} ms  {r['gbs']:7.1f} GB/s  ({pico:.0%} del pico)", flush=True)
                    filas.append({"variante": nombre, "dtype": dt, "M": M, "N": N, **r,
                                  "pct_pico": round(100 * pico, 1)})
                sp = gbs["triton"] / gbs["torch"]
                print(f"   -> Triton/torch: {sp:.1%}\n", flush=True)
                resumen[f"{dt}_{M}x{N}"] = {**{f"{k}_gbs": v for k, v in gbs.items()},
                                            "triton_vs_torch": round(sp, 4)}

    comun.guardar(filas, parametros=vars(args), resumen=resumen)


if __name__ == "__main__":
    main()
