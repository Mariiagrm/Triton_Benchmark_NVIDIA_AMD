| variante | dtype | M | N | K | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages | carga | usa_tma | autotune_s | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | pct_pico | gbs | mma |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| baseline | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 4 | cp.async | False | 7.8 | 1.4337 | 1.4304 | 1.4428 | 2411 | 58.4 | 95.87 | 78.1 | 70.2 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| blockptr | fp16 | 4096 | 4096 | 4096 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | False | 15.7 | 1.5053 | 1.496 | 1.5356 | 2470 | 84.7 | 91.3 | 74.3 | 66.9 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 8 | 4 | TMA | True | 19.3 | 1.4428 | 1.4388 | 1.45 | 2444 | 93.7 | 95.26 | 77.6 | 69.8 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| baseline | fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | False | 13.9 | 12.3768 | 12.1353 | 12.6669 | 2262 | 91.6 | 88.84 | 72.3 | 32.5 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| blockptr | fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | False | 34.9 | 11.989 | 11.9565 | 12.4801 | 2255 | 87.1 | 91.71 | 74.7 | 33.6 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 8192 | 8192 | 8192 | 128 | 128 | 32 | 8 | 4 | 4 | TMA | True | 64.5 | 11.6204 | 11.5839 | 12.1297 | 2470 | 89.2 | 94.62 | 77 | 34.7 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 21058, 2026-10-15T08:11:35. Mediana de triton.testing.do_bench.*
