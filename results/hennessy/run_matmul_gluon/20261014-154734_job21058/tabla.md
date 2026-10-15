| variante | dtype | M | N | K | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | pct_pico | gbs | t_compilacion_s | detalle |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| gluon | fp16 | 4096 | 4096 | 4096 | 1.6691 | 1.6624 | 1.7327 | 2411 | 14.3 | 82.34 | 67 | 60.3 | 1.29 | BM=128 BN=128 BK=32 stages=3 warps=2x2 mma=mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| cublas | fp16 | 4096 | 4096 | 4096 | 1.7665 | 1.7279 | 1.8851 | 2457 | 76 | 77.8 | 63.3 | 57 |  | void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_nn_align8>(cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_nn_align8::Params) |
| gluon | fp16 | 8192 | 8192 | 8192 | 14.1251 | 14.0708 | 14.541 | 2229 | 90.6 | 77.84 | 63.4 | 28.5 | 1.81 | BM=128 BN=128 BK=32 stages=3 warps=2x2 mma=mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| cublas | fp16 | 8192 | 8192 | 8192 | 11.5314 | 11.5069 | 12.1859 | 2262 | 95.2 | 95.35 | 77.6 | 34.9 |  | nvjet_sm121_hsh_mma_128x208x64_2_32x104x64_tmaAB_bz_NNNN |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 21058, 2026-10-14T15:47:34. Mediana de triton.testing.do_bench.*
