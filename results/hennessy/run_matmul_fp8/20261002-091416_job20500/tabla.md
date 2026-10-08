| variante | M | N | K | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages | carga | error_rel | ms | ms_p20 | ms_p80 | tflops | gbs | mma |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 4 | cp.async | 0.00021 | 1.4743 | 1.4684 | 1.4799 | 93.22 | 68.3 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| fp8 | 4096 | 4096 | 4096 | 256 | 128 | 128 | 8 | 8 | 3 | cp.async | 0.03752 | 0.9226 | 0.9195 | 0.9247 | 148.97 | 72.7 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_tma | 4096 | 4096 | 4096 | 128 | 256 | 128 | 8 | 8 | 3 | TMA | 0.03752 | 0.9063 | 0.9042 | 0.9097 | 151.65 | 74 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_cublas | 4096 | 4096 | 4096 |  |  |  |  |  |  | cublasLt | 0.03752 | 0.778 | 0.7731 | 0.7813 | 176.65 | 86.3 | cublasLt |
| fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | 0.00021 | 11.8532 | 11.829 | 11.8732 | 92.76 | 34 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| fp8 | 8192 | 8192 | 8192 | 256 | 128 | 128 | 8 | 8 | 3 | cp.async | 0.03755 | 7.4235 | 7.4101 | 7.4414 | 148.11 | 36.2 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_tma | 8192 | 8192 | 8192 | 128 | 256 | 128 | 8 | 8 | 3 | TMA | 0.03755 | 7.122 | 7.1127 | 7.1321 | 154.38 | 37.7 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_cublas | 8192 | 8192 | 8192 |  |  |  |  |  |  | cublasLt | 0.03755 | 5.8532 | 5.8219 | 5.8783 | 187.85 | 45.9 | cublasLt |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20500, 2026-10-02T09:14:16. Mediana de triton.testing.do_bench.*
