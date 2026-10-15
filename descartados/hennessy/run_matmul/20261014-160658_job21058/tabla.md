| dtype | M×N×K | Config. Triton (BM×BN×BK, warps, stages) | Triton (ms) | Triton (TFLOP/s) | cuBLAS (ms) | cuBLAS (TFLOP/s) | Triton/cuBLAS |
|:---|:---|:---|---:|---:|---:|---:|---:|
| fp16 | 4096×4096×4096 | 128×128×32, w4, s3 | 3.992 | 34.4 | 4.316 | 31.8 | 108.1% |
| fp16 | 8192×8192×8192 | 128×256×64, w8, s3 | 26.545 | 41.4 | 26.649 | 41.3 | 100.4% |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 21058, 2026-10-14T16:06:58. Mediana de triton.testing.do_bench.*

Tensor cores (PTX Triton): `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
