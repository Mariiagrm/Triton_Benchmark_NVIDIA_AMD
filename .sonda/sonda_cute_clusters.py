"""Sonda: por que el dense_gemm de CuTe rinde ~24 % de cuBLAS en GB10 (job 20841) y no en la RTX 5090.

Hipotesis: el kernel es persistente y lanza max_active_clusters clusters, que se pide a
cutlass.utils.HardwareInfo().get_max_active_clusters(1); si en GB10 devuelve menos que los SMs
reales, solo trabaja una parte de la GPU. Se imprime ese valor y se mide el GEMM forzandolo.
Uso: DSL=cutlass VALIDAR=0 bash ejecutar.sh encolar exp .sonda/sonda_cute_clusters.py
"""
import torch

import comun
from CBKernels.cutlass import matmul_cute


def main():
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch

    comun.imprimir_contexto()
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    hw = cutlass.utils.HardwareInfo()
    print(f"SMs (torch): {n_sm}")
    for nombre in ("get_max_active_clusters", "get_device_multiprocessor_count", "get_l2_cache_size_in_bytes"):
        f = getattr(hw, nombre, None)
        try:
            print(f"HardwareInfo.{nombre}: {f(1) if nombre == 'get_max_active_clusters' else f()}")
        except Exception as e:
            print(f"HardwareInfo.{nombre}: {type(e).__name__}: {e}")

    n = 8192
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    c = torch.empty((n, n), device="cuda", dtype=torch.float16)
    ta = matmul_cute._tensor_cute(a.unsqueeze(0).permute(1, 2, 0))
    tb = matmul_cute._tensor_cute(b.unsqueeze(0).permute(2, 1, 0))
    tc = matmul_cute._tensor_cute(c.unsqueeze(0).permute(1, 2, 0))
    stream = cutlass_torch.default_stream()
    ref = torch.matmul(a, b)
    rc = comun.medir(lambda: torch.matmul(a, b), flops=2 * n ** 3)
    print(f"cuBLAS: {rc['tflops']} TFLOP/s ({rc['reloj_mhz']} MHz, {rc['potencia_w']} W)")
    for clusters in sorted({hw.get_max_active_clusters(1), n_sm, 2 * n_sm, 4 * n_sm}):
        gemm = matmul_cute.ejemplo().Sm120GemmKernel(cutlass.Float32, tuple(matmul_cute.TILE_POR_DEFECTO), 4)
        compilado = cute.compile(gemm, ta, tb, tc, clusters, stream)
        compilado(ta, tb, tc, stream)
        torch.cuda.synchronize()
        err = ((c.float() - ref.float()).norm() / ref.float().norm()).item()
        r = comun.medir(lambda: compilado(ta, tb, tc, stream), flops=2 * n ** 3)
        print(f"max_active_clusters={clusters:>4}: {r['tflops']:7.2f} TFLOP/s  ({r['reloj_mhz']} MHz, "
              f"{r['potencia_w']} W)  error_rel={err:.2e}", flush=True)


if __name__ == "__main__":
    main()
