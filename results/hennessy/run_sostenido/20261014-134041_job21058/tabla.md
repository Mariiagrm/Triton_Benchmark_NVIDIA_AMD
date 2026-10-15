| kernel | precision | tflops | pct_pico | reloj_mhz | potencia_w | pico_mma_a_ese_reloj | pct_pico_mismo_reloj |
|:---|:---|:---|:---|:---|:---|:---|:---|
| mma.sync f16·f32acc | fp16 | 119.45 | 97.2 | 2444 | 99.2 | 120.1 | 99.4 |
| mma.sync e4m3·f32acc | fp8 | 236.66 | 192.7 | 2418 | 99 | 237.7 | 99.6 |
| cuBLAS fp16 | fp16 | 94.03 | 76.6 | 2125 | 89.8 | 104.4 | 90 |
| Triton fp16 | fp16 | 89.8 | 73.1 | 2177 | 89.5 | 107 | 83.9 |
| Triton + TMA fp16 | fp16 | 94.09 | 76.6 | 2086 | 89.8 | 102.5 | 91.8 |
| cuBLASLt FP8 | fp8 | 184.62 | 150.3 | 2151 | 91.5 | 211.5 | 87.3 |
| Triton FP8 | fp8 | 144.82 | 117.9 | 2158 | 91.4 | 212.1 | 68.3 |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 21058, 2026-10-14T13:40:41. Mediana de triton.testing.do_bench.*
