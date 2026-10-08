| variante | M | N | K | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages | carga | error_rel | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | gbs | reloj_sost_mhz | potencia_sost_w | potencia_reposo_w | mj_llamada | tflops_sost | gflop_j | gflop_j_din | mma |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | 0.00021 | 12.1361 | 12.065 | 12.5801 | 2476 | 88.3 | 90.6 | 33.2 | 2177 | 89.3 | 16.1 | 1086.35 | 90.38 | 1012.1 | 1234.7 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| fp8 | 8192 | 8192 | 8192 | 256 | 128 | 128 | 8 | 8 | 3 | cp.async | 0.03753 | 7.5535 | 7.5305 | 7.5803 | 2457 | 91.8 | 145.56 | 35.5 | 2138 | 90.1 | 16.1 | 693.35 | 142.88 | 1585.8 | 1930.8 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_tma | 8192 | 8192 | 8192 | 128 | 256 | 128 | 8 | 8 | 3 | TMA | 0.03753 | 7.3144 | 7.2954 | 7.6698 | 2483 | 39.3 | 150.32 | 36.7 | 2158 | 90.4 | 16.1 | 674.888 | 147.28 | 1629.2 | 1982.2 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_cublas | 8192 | 8192 | 8192 |  |  |  |  |  |  | cublasLt | 0.03753 | 5.9822 | 5.9336 | 6.126 | 2158 | 90.8 | 183.8 | 44.9 | 2112 | 89.5 | 16.1 | 539.891 | 182.27 | 2036.5 | 2483.2 | cublasLt |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20849, 2026-10-08T23:30:14. Mediana de triton.testing.do_bench.*
