| variante | dtype | M | N | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | gbs | pct_pico | t_compilacion_s | detalle |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| triton | fp16 | 16384 | 8192 | 2.2159 | 2.2118 | 2.2236 | 2418 | 15.5 | 242.3 | 88.8 | 0.61 | num_warps=16 |
| torch | fp16 | 16384 | 8192 | 2.4944 | 2.4862 | 2.5076 | 2502 | 27.2 | 215.2 | 78.8 |  |  |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20843, 2026-10-08T16:15:13. Mediana de triton.testing.do_bench.*
