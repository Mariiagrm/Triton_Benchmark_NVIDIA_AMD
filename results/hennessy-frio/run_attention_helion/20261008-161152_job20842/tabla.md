| variante | dtype | B | H | N_CTX | HEAD_DIM | causal | ms | ms_p20 | ms_p80 | reloj_mhz | potencia_w | tflops | gbs | error_rel | t_compilacion_s | detalle |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| helion | fp16 | 4 | 32 | 4096 | 128 | False | 12.9725 | 12.9516 | 13.484 | 2418 | 20.8 | 84.76 | 41.4 | 0.000439 | 127.62 | autotune=quick block_sizes=[1,128,32] num_warps=4 num_stages=2 pid_type=flat indexing=['tensor_descriptor','pointer','tensor_descriptor','pointer'] mma=mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 |
| sdpa | fp16 | 4 | 32 | 4096 | 128 | False | 13.4932 | 13.451 | 13.9749 | 2210 | 67.6 | 81.49 | 39.8 |  |  | void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::half_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::half_t> >, false, false, false, false, true, true, false, false>(pytorch_flash::Flash_fwd_params) |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20842, 2026-10-08T16:11:52. Mediana de triton.testing.do_bench.*
