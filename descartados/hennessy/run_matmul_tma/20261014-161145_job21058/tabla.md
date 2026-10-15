| variante | dtype | M | N | K | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages | carga | usa_tma | autotune_s | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | gbs | mma |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| baseline | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 3 | cp.async | False | 9.8 | 3.9132 | 2.5558 | 4.047 | 2483 | 39.3 | 35.12 | 25.7 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| blockptr | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 3 | cp.async | False | 20.4 | 3.9608 | 1.5085 | 4.1081 | 2483 | 55.5 | 34.7 | 25.4 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 3 | TMA | True | 28.4 | 3.917 | 3.8543 | 4.203 | 2483 | 55.1 | 35.09 | 25.7 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| baseline | fp16 | 8192 | 8192 | 8192 | 128 | 128 | 32 | 8 | 4 | 3 | cp.async | False | 24.3 | 26.6454 | 26.4005 | 27.101 | 2463 | 72.6 | 41.26 | 15.1 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| blockptr | fp16 | 8192 | 8192 | 8192 | 128 | 256 | 32 | 8 | 8 | 5 | cp.async | False | 65.7 | 26.8785 | 26.7438 | 27.2162 | 2444 | 75.2 | 40.91 | 15 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 8192 | 8192 | 8192 | 128 | 128 | 32 | 8 | 4 | 4 | TMA | True | 129.1 | 25.9225 | 23.6312 | 26.371 | 2463 | 68.5 | 42.42 | 15.5 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 21058, 2026-10-14T16:11:45. Mediana de triton.testing.do_bench.*
