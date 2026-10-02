| variante | dtype | M | N | K | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages | carga | usa_tma | autotune_s | ms | ms_p20 | ms_p80 | tflops | gbs | mma |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| baseline | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 4 | cp.async | False | 7.8 | 1.3998 | 1.3964 | 1.5618 | 98.18 | 71.9 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 4 | cp.async | False | 58.8 | 1.4448 | 1.4377 | 1.5731 | 95.12 | 69.7 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| baseline | fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | False | 14 | 12.0494 | 11.9899 | 12.5665 | 91.25 | 33.4 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| tma | fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | False | 35.2 | 12.1661 | 11.9353 | 12.5131 | 90.37 | 33.1 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| baseline | bf16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 4 | cp.async | False | 7.6 | 1.4059 | 1.4019 | 1.5525 | 97.76 | 71.6 | mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 |
| tma | bf16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 4 | cp.async | False | 60.4 | 1.4377 | 1.4316 | 1.546 | 95.6 | 70 | mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 |
| baseline | bf16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | False | 13.4 | 11.8363 | 11.8069 | 12.39 | 92.89 | 34 | mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 |
| tma | bf16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | False | 34.6 | 11.7289 | 11.698 | 12.2685 | 93.74 | 34.3 | mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20397, 2026-09-29T20:45:34. Mediana de triton.testing.do_bench.*
