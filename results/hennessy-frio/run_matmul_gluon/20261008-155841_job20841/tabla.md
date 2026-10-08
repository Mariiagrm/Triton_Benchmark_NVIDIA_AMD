| variante | dtype | M | N | K | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | gbs | t_compilacion_s | detalle |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| gluon | fp16 | 8192 | 8192 | 8192 | 13.8722 | 13.8257 | 14.2421 | 2411 | 39.3 | 79.26 | 29 | 3.88 | BM=128 BN=128 BK=32 stages=3 warps=2x2 mma=mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| cublas | fp16 | 8192 | 8192 | 8192 | 11.3051 | 11.2743 | 11.5016 | 2320 | 62.4 | 97.26 | 35.6 |  | nvjet_sm121_hsh_mma_128x208x64_2_32x104x64_tmaAB_bz_NNNN |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20841, 2026-10-08T15:58:41. Mediana de triton.testing.do_bench.*
