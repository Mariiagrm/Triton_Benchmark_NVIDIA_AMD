# run_matmul_fp8

Generado automáticamente desde `results/run_matmul_fp8/20260930-141501_job20423/`.

![run_matmul_fp8](figuras/run_matmul_fp8.png)

| variante | M | N | K | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages | carga | error_rel | ms | ms_p20 | ms_p80 | tflops | gbs | mma |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 4 | cp.async | 0.00021 | 1.4033 | 1.3978 | 1.5971 | 97.94 | 71.7 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| fp8 | 4096 | 4096 | 4096 | 256 | 128 | 128 | 8 | 8 | 3 | cp.async | 0.03752 | 0.9103 | 0.9073 | 0.9185 | 150.98 | 73.7 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_tma | 4096 | 4096 | 4096 | 128 | 128 | 64 | 8 | 4 | 4 | TMA | 0.03752 | 0.9093 | 0.9073 | 0.9155 | 151.15 | 73.8 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_cublas | 4096 | 4096 | 4096 |  |  |  |  |  |  | cublasLt | 0.03752 | 0.7854 | 0.7803 | 0.7967 | 175 | 85.4 | cublasLt |
| fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | 0.00021 | 11.8026 | 11.7821 | 12.3607 | 93.16 | 34.1 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| fp8 | 8192 | 8192 | 8192 | 256 | 128 | 128 | 8 | 8 | 3 | cp.async | 0.03755 | 7.6595 | 7.6275 | 7.9587 | 143.55 | 35 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_tma | 8192 | 8192 | 8192 | 128 | 256 | 128 | 8 | 8 | 3 | TMA | 0.03755 | 7.2253 | 7.2083 | 7.6075 | 152.18 | 37.2 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_cublas | 8192 | 8192 | 8192 |  |  |  |  |  |  | cublasLt | 0.03755 | 5.8716 | 5.8011 | 5.9836 | 187.26 | 45.7 | cublasLt |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20423, 2026-09-30T14:15:01. Mediana de triton.testing.do_bench.*
