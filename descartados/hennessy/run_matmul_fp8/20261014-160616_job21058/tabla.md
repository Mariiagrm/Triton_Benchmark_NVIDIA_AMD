| variante | M | N | K | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages | carga | error_rel | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | gbs | mma |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| fp16 | 4096 | 4096 | 4096 | 128 | 128 | 64 | 8 | 8 | 4 | cp.async | 0.00021 | 4.1764 | 4.0776 | 4.3481 | 2476 | 46.5 | 32.91 | 24.1 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| fp8 | 4096 | 4096 | 4096 | 256 | 128 | 128 | 8 | 8 | 3 | cp.async | 0.03752 | 1.0273 | 0.9196 | 3.4038 | 2470 | 61.8 | 133.78 | 65.3 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_tma | 4096 | 4096 | 4096 | 128 | 256 | 128 | 8 | 8 | 3 | TMA | 0.03752 | 0.9932 | 0.9216 | 3.3921 | 2463 | 50.9 | 138.39 | 67.6 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_cublas | 4096 | 4096 | 4096 |  |  |  |  |  |  | cublasLt | 0.03752 | 0.7957 | 0.7895 | 3.2442 | 2470 | 58.5 | 172.72 | 84.3 | cublasLt |
| fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | 0.00021 | 26.2922 | 23.9856 | 26.6363 | 2450 | 70.6 | 41.82 | 15.3 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| fp8 | 8192 | 8192 | 8192 | 128 | 128 | 64 | 8 | 4 | 3 | cp.async | 0.03755 | 17.5207 | 17.128 | 17.5968 | 2418 | 71.2 | 62.75 | 15.3 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_tma | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | TMA | 0.03755 | 17.7377 | 17.551 | 17.8241 | 2470 | 69.7 | 61.99 | 15.1 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_cublas | 8192 | 8192 | 8192 |  |  |  |  |  |  | cublasLt | 0.03755 | 13.3475 | 13.1248 | 13.5377 | 2450 | 71.5 | 82.38 | 20.1 | cublasLt |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 21058, 2026-10-14T16:06:16. Mediana de triton.testing.do_bench.*
