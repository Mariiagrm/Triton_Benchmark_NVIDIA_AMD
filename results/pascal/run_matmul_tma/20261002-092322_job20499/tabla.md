| variante | dtype | M | N | K | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages | carga | usa_tma | autotune_s | ms | ms_p20 | ms_p80 | tflops | gbs | mma |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| baseline | fp16 | 4096 | 4096 | 4096 | 128 | 64 | 32 | 8 | 4 | 3 | cp.async | False | 7.3 | 0.6639 | 0.6625 | 0.6688 | 207.01 | 151.6 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| blockptr | fp16 | 4096 | 4096 | 4096 | 64 | 128 | 32 | 8 | 4 | 3 | cp.async | False | 57.4 | 0.6539 | 0.6513 | 0.6591 | 210.2 | 154 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 4096 | 4096 | 4096 | 64 | 128 | 64 | 8 | 4 | 2 | TMA | True | 47.3 | 0.6431 | 0.639 | 0.6461 | 213.72 | 156.5 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| baseline | fp16 | 8192 | 8192 | 8192 | 256 | 128 | 64 | 8 | 8 | 3 | cp.async | False | 8.9 | 5.036 | 5.034 | 5.0381 | 218.33 | 80 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| blockptr | fp16 | 8192 | 8192 | 8192 | 128 | 128 | 64 | 8 | 4 | 2 | cp.async | False | 18 | 5.0529 | 5.0504 | 5.0545 | 217.6 | 79.7 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 8192 | 8192 | 8192 | 128 | 128 | 64 | 8 | 4 | 2 | TMA | True | 23.1 | 4.9172 | 4.9121 | 4.9213 | 223.6 | 81.9 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |

*NVIDIA GeForce RTX 5090 (sm_120), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20499, 2026-10-02T09:23:22. Mediana de triton.testing.do_bench.*
