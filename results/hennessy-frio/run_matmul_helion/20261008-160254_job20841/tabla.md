| variante | dtype | M | N | K | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | gbs | t_compilacion_s | detalle |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| helion | fp16 | 8192 | 8192 | 8192 | 17.751 | 17.6759 | 18.1758 | 2418 | 20.8 | 61.94 | 22.7 | 239.55 | autotune=quick block_sizes=[64,512,64] num_warps=16 num_stages=2 pid_type=persistent_interleaved indexing=['pointer','pointer','pointer'] |
| cublas | fp16 | 8192 | 8192 | 8192 | 11.6762 | 11.6242 | 12.1746 | 2405 | 61.9 | 94.17 | 34.5 |  | nvjet_sm121_hsh_mma_128x208x64_2_32x104x64_tmaAB_bz_NNNN |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20841, 2026-10-08T16:02:54. Mediana de triton.testing.do_bench.*
