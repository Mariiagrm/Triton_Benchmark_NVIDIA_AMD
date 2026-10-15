| variante | dtype | M | N | K | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages | carga | usa_tma | autotune_s | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | pct_pico | gbs | mma |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| baseline | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 4 | cp.async | False | 7.8 | 1.4275 | 1.4232 | 1.4833 | 2437 | 79.6 | 96.28 | 78.4 | 70.5 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| blockptr | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 4 | cp.async | False | 15.7 | 1.4735 | 1.4684 | 1.5228 | 2444 | 78.7 | 93.27 | 75.9 | 68.3 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 8 | 4 | TMA | True | 19.2 | 1.4338 | 1.4304 | 1.5143 | 2366 | 80.4 | 95.86 | 78 | 70.2 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| baseline | fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | False | 13.7 | 12.1445 | 12.0966 | 12.5997 | 2437 | 93.8 | 90.54 | 73.7 | 33.2 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| blockptr | fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | False | 34.5 | 11.8999 | 11.8747 | 12.3753 | 2268 | 87.8 | 92.4 | 75.2 | 33.8 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 8192 | 8192 | 8192 | 128 | 128 | 32 | 8 | 4 | 4 | TMA | True | 64.3 | 11.4975 | 11.4678 | 12.0963 | 2229 | 89.9 | 95.63 | 77.9 | 35 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 21058, 2026-10-14T15:47:03. Mediana de triton.testing.do_bench.*
