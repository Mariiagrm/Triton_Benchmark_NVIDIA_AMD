| variante | dtype | M | N | K | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | gbs | t_compilacion_s | detalle |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| cutlass | fp16 | 8192 | 8192 | 8192 | 46.6149 | 46.1304 | 46.7775 | 2424 | 14.8 | 23.59 | 8.6 | 1.04 | Sm120GemmKernel tile=128x128x64 |
| cublas | fp16 | 8192 | 8192 | 8192 | 11.391 | 11.3547 | 11.9424 | 2437 | 32.7 | 96.52 | 35.3 |  | nvjet_sm121_hsh_mma_128x208x64_2_32x104x64_tmaAB_bz_NNNN |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20841, 2026-10-08T15:58:28. Mediana de triton.testing.do_bench.*
