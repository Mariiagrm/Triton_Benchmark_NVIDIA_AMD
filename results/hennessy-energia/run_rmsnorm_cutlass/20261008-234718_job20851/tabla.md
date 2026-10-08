| variante | dtype | M | N | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | gbs | reloj_sost_mhz | potencia_sost_w | potencia_reposo_w | mj_llamada | gbs_sost | gb_j | gb_j_din | pct_pico | t_compilacion_s | detalle |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| cutlass | fp16 | 16384 | 8192 | 2.1749 | 2.1699 | 2.1821 | 2411 | 22.6 | 246.9 | 2502 | 41.4 | 13.7 | 89.413 | 248.6 | 6 | 8.97 | 90.4 | 55.3 | rmsnorm_twoPassAlgo_e8 |
| torch | fp16 | 16384 | 8192 | 2.4848 | 2.475 | 2.4934 | 2502 | 41.6 | 216.1 | 2437 | 34.4 | 13.7 | 85.2705 | 216.6 | 6.3 | 10.46 | 79.2 |  |  |
| cutlass | bf16 | 16384 | 8192 | 2.3655 | 2.3589 | 2.3726 | 2431 | 34.5 | 227 | 2502 | 49 | 13.7 | 114.723 | 229.3 | 4.68 | 6.5 | 83.2 | 0 | rmsnorm_twoPassAlgo_e1 |
| torch | bf16 | 16384 | 8192 | 2.4812 | 2.471 | 2.49 | 2502 | 34 | 216.4 | 2437 | 34.6 | 13.7 | 85.3064 | 217.8 | 6.29 | 10.42 | 79.3 |  |  |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20851, 2026-10-08T23:47:18. Mediana de triton.testing.do_bench.*
