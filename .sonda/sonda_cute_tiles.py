"""Sonda: el dense_gemm de CuTe rinde ~24 % de cuBLAS en GB10 (job 20841) con su tile por defecto,
y no por max_active_clusters (sonda job 20844). Barre tiles CTA del ejemplo a 8192^3 fp16.
Uso: DSL=cutlass VALIDAR=0 bash ejecutar.sh encolar exp .sonda/sonda_cute_tiles.py
"""
import torch

import comun
from CBKernels.cutlass import matmul_cute

comun.imprimir_contexto()
n = 8192
a = torch.randn((n, n), device="cuda", dtype=torch.float16)
b = torch.randn((n, n), device="cuda", dtype=torch.float16)
r = comun.medir(lambda: torch.matmul(a, b), flops=2 * n ** 3)
print(f"cuBLAS: {r['tflops']} TFLOP/s ({r['reloj_mhz']} MHz)", flush=True)
for tile in [(128, 128, 64), (128, 128, 32), (128, 256, 64), (256, 128, 64), (64, 128, 64), (128, 64, 64), (64, 64, 64)]:
    try:
        fn = matmul_cute.preparar(a, b, tile=tile)
        fn()
        torch.cuda.synchronize()
        r = comun.medir(fn, flops=2 * n ** 3)
        print(f"tile {tile}: {r['tflops']:7.2f} TFLOP/s ({r['reloj_mhz']} MHz)", flush=True)
    except Exception as e:
        print(f"tile {tile}: {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True)
