| variante | dtype | M | N | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | gbs | pct_pico | t_compilacion_s | detalle |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| helion | fp16 | 16384 | 8192 | 2.1719 | 2.1678 | 2.1806 | 2418 | 18 | 247.2 | 90.5 | 77.33 | autotune=quick block_sizes=[1] num_warps=32 num_stages=1 pid_type=flat indexing=['tensor_descriptor','pointer','pointer','pointer','tensor_descriptor','tensor_descriptor','pointer'] |
| torch | fp16 | 16384 | 8192 | 2.6236 | 2.6122 | 2.6368 | 2502 | 28.5 | 204.6 | 74.9 |  |  |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20843, 2026-10-08T16:15:02. Mediana de triton.testing.do_bench.*
