| variante | M | N | K | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | num_warps | num_stages | carga | error_rel | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | gbs | mma |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| fp16 | 4096 | 4096 | 4096 | 128 | 128 | 32 | 8 | 4 | 4 | cp.async | 0.00021 | 1.4244 | 1.4193 | 1.5068 | 2476 | 76.3 | 96.49 | 70.7 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| fp8 | 4096 | 4096 | 4096 | 256 | 128 | 128 | 8 | 8 | 3 | cp.async | 0.03752 | 0.936 | 0.9328 | 0.9433 | 2457 | 87.5 | 146.84 | 71.7 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_tma | 4096 | 4096 | 4096 | 128 | 256 | 128 | 8 | 8 | 3 | TMA | 0.03752 | 0.9236 | 0.9218 | 0.939 | 2450 | 86.4 | 148.8 | 72.7 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_cublas | 4096 | 4096 | 4096 |  |  |  |  |  |  | cublasLt | 0.03752 | 0.7905 | 0.7855 | 0.8664 | 2450 | 57.8 | 173.86 | 84.9 | cublasLt |
| fp16 | 8192 | 8192 | 8192 | 128 | 256 | 64 | 8 | 8 | 3 | cp.async | 0.00021 | 12.1891 | 12.0996 | 12.58 | 2255 | 89.5 | 90.2 | 33 | mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| fp8 | 8192 | 8192 | 8192 | 256 | 128 | 128 | 8 | 8 | 3 | cp.async | 0.03755 | 7.5935 | 7.5694 | 7.9944 | 2450 | 96.3 | 144.8 | 35.4 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_tma | 8192 | 8192 | 8192 | 256 | 128 | 128 | 8 | 8 | 3 | TMA | 0.03755 | 7.5059 | 7.4595 | 7.8738 | 2463 | 110.2 | 146.49 | 35.8 | mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 |
| fp8_cublas | 8192 | 8192 | 8192 |  |  |  |  |  |  | cublasLt | 0.03755 | 5.8737 | 5.8332 | 6.0909 | 2288 | 90.1 | 187.19 | 45.7 | cublasLt |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 21058, 2026-10-14T15:18:50. Mediana de triton.testing.do_bench.*
