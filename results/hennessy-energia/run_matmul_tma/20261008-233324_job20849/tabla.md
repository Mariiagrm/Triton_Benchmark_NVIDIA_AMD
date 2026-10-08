| variante | dtype | M | N | K | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages | carga | usa_tma | autotune_s | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | gbs | reloj_sost_mhz | potencia_sost_w | potencia_reposo_w | mj_llamada | tflops_sost | gflop_j | gflop_j_din | mma |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| baseline | fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | False | 13.8 | 12.6313 | 12.1885 | 12.716 | 2255 | 92.1 | 87.05 | 31.9 | 2158 | 88.3 | 17 | 1078.93 | 89.98 | 1019.1 | 1262 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| blockptr | fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | False | 35 | 11.9829 | 11.9518 | 12.5148 | 2268 | 87.7 | 91.76 | 33.6 | 2125 | 87.9 | 17 | 1061.28 | 91.07 | 1036 | 1284.4 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 8192 | 8192 | 8192 | 128 | 128 | 32 | 8 | 4 | 4 | TMA | True | 64.5 | 11.5865 | 11.5614 | 11.6543 | 2242 | 86 | 94.9 | 34.8 | 2047 | 88 | 17 | 1033.32 | 93.64 | 1064.1 | 1318.8 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20849, 2026-10-08T23:33:24. Mediana de triton.testing.do_bench.*
