| dtype | M×N×K | Config. Triton (BM×BN×BK, warps, stages) | Triton (ms) | Triton (TFLOP/s) | cuBLAS (ms) | cuBLAS (TFLOP/s) | Triton/cuBLAS |
|:---|:---|:---|---:|---:|---:|---:|---:|
| fp16 | 4096×4096×4096 | 128×128×32, w4, s4 | 1.396 | 98.5 | 1.732 | 79.4 | 124.1% |
| fp16 | 8192×8192×8192 | 128×256×64, w8, s3 | 11.724 | 93.8 | 11.715 | 93.9 | 99.9% |
| fp16 | 16384×16384×16384 | 256×128×64, w8, s3 | 95.192 | 92.4 | 92.074 | 95.5 | 96.7% |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20388, 2026-09-29T09:35:29. Mediana de triton.testing.do_bench.*
