| variante | dtype | M | N | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | gbs | pct_pico | t_compilacion_s | detalle |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| cutlass | fp16 | 16384 | 8192 | 2.174 | 2.1688 | 2.183 | 2411 | 13.5 | 247 | 90.5 | 55.52 | rmsnorm_twoPassAlgo_e8 |
| torch | fp16 | 16384 | 8192 | 2.496 | 2.4842 | 2.5056 | 2450 | 30 | 215.1 | 78.8 |  |  |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20843, 2026-10-08T16:13:24. Mediana de triton.testing.do_bench.*
