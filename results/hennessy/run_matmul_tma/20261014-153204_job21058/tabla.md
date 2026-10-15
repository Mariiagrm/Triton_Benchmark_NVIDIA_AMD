| variante | dtype | M | N | K | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages | carga | usa_tma | autotune_s | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | gbs | mma |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| baseline | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 4 | cp.async | False | 7.9 | 1.4326 | 1.4285 | 1.4746 | 2411 | 85.8 | 95.94 | 70.3 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| blockptr | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 3 | cp.async | False | 15.7 | 1.4771 | 1.4692 | 1.5197 | 2463 | 81.2 | 93.05 | 68.1 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 8 | 4 | TMA | True | 19.3 | 1.4418 | 1.4377 | 1.5174 | 2457 | 85.1 | 95.33 | 69.8 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| baseline | fp16 | 8192 | 8192 | 8192 | 256 | 128 | 64 | 8 | 8 | 3 | cp.async | False | 13.7 | 12.2281 | 12.1865 | 12.7417 | 2242 | 87.4 | 89.92 | 32.9 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| blockptr | fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | False | 34.7 | 11.948 | 11.8925 | 12.4259 | 2424 | 88.4 | 92.02 | 33.7 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 8192 | 8192 | 8192 | 128 | 128 | 32 | 8 | 4 | 4 | TMA | True | 64.5 | 11.4693 | 11.4385 | 12.0705 | 2275 | 87 | 95.87 | 35.1 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 21058, 2026-10-14T15:32:04. Mediana de triton.testing.do_bench.*
