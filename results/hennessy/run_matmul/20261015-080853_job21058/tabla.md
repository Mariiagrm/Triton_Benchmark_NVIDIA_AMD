| dtype | M×N×K | Config. Triton (BM×BN×BK, warps, stages) | Triton (ms) | Triton (TFLOP/s) | cuBLAS (ms) | cuBLAS (TFLOP/s) | Triton/cuBLAS |
|:---|:---|:---|---:|---:|---:|---:|---:|
| fp16 | 4096×4096×4096 | 128×128×32, w4, s4 | 1.427 | 96.3 | 1.728 | 79.5 | 121.2% |
| fp16 | 8192×8192×8192 | 256×128×64, w8, s3 | 12.232 | 89.9 | 11.828 | 93.0 | 96.7% |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 21058, 2026-10-15T08:08:53. Mediana de triton.testing.do_bench.*

Tensor cores (PTX Triton): `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
