| variante | dtype | M | N | K | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | pct_pico | gbs | t_compilacion_s | detalle |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| gluon | fp16 | 4096 | 4096 | 4096 | 1.6876 | 1.6825 | 1.7869 | 2411 | 14.6 | 81.44 | 66.3 | 59.7 | 1.28 | BM=128 BN=128 BK=32 stages=3 warps=2x2 mma=mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| cublas | fp16 | 4096 | 4096 | 4096 | 1.7315 | 1.699 | 1.9188 | 2437 | 82.9 | 79.38 | 64.6 | 58.1 |  | void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_nn_align8>(cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_nn_align8::Params) |
| gluon | fp16 | 8192 | 8192 | 8192 | 14.1972 | 14.1724 | 14.2391 | 2203 | 89.2 | 77.45 | 63.1 | 28.4 | 1.77 | BM=128 BN=128 BK=32 stages=3 warps=2x2 mma=mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| cublas | fp16 | 8192 | 8192 | 8192 | 11.5795 | 11.545 | 11.7334 | 2177 | 89.1 | 94.95 | 77.3 | 34.8 |  | nvjet_sm121_hsh_mma_128x208x64_2_32x104x64_tmaAB_bz_NNNN |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 21058, 2026-10-15T08:12:07. Mediana de triton.testing.do_bench.*
