| variante | dtype | M | N | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | gbs | pct_pico | t_compilacion_s | detalle |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| gluon | fp16 | 16384 | 8192 | 2.218 | 2.2129 | 2.2252 | 2489 | 15.8 | 242.1 | 88.7 | 0.6 | num_warps=16 vec=8 |
| torch | fp16 | 16384 | 8192 | 2.4852 | 2.4772 | 2.4981 | 2522 | 26.5 | 216 | 79.1 |  |  |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20843, 2026-10-08T16:13:32. Mediana de triton.testing.do_bench.*
