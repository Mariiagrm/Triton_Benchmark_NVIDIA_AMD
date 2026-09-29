| variante | M | N | K | error_rel | ms | ms_p20 | ms_p80 | tflops | kernel |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| densa_fp16 | 4096 | 4096 |  | 0 | 2.0489 | 2.0243 | 2.1009 | 67.08 | void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) |
| sparse24_fp16 | 4096 | 4096 |  | 4e-05 | 2.7698 | 2.6865 | 2.8506 | 49.62 | sm80_xmma_sparse_gemm_f16f16_f16f32_f32_tt_t_tilesize128x128x64_stage3_warpsize2x2x1_sptensor16x8x32_execute_kernel_5x_cusparselt |
| densa_fp16 | 8192 | 8192 |  | 0 | 13.5557 | 13.0844 | 13.8432 | 81.11 | void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) |
| sparse24_fp16 | 8192 | 8192 |  | 6e-05 | 32.3276 | 32.2441 | 32.7116 | 34.01 | sm80_xmma_sparse_gemm_f16f16_f16f32_f32_tt_t_tilesize128x128x64_stage3_warpsize2x2x1_sptensor16x8x32_execute_kernel_5x_cusparselt |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20401, 2026-09-29T21:39:34. Mediana de triton.testing.do_bench.*
