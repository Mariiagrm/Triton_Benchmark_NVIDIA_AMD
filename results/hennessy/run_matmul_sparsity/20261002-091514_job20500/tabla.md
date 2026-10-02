| variante | M | N | K | error_rel | ms | ms_p20 | ms_p80 | tflops | kernel |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| densa_fp16 | 4096 | 4096 |  | 0 | 1.7992 | 1.749 | 1.8408 | 76.39 | void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_nn_align8>(cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_nn_align8::Params) |
| sparse24_fp16 | 4096 | 4096 |  | 4e-05 | 1.0998 | 1.0937 | 1.108 | 124.96 | sm80_xmma_sparse_gemm_f16f16_f16f32_f32_tt_t_tilesize128x128x64_stage3_warpsize2x2x1_sptensor16x8x32_execute_kernel_5x_cusparselt |
| densa_fp16 | 8192 | 8192 |  | 0 | 11.7422 | 11.7161 | 11.7791 | 93.64 | nvjet_sm121_hsh_mma_128x208x64_2_32x104x64_tmaAB_bz_NNNN |
| sparse24_fp16 | 8192 | 8192 |  | 6e-05 | 34.2184 | 33.8592 | 34.4101 | 32.13 | sm80_xmma_sparse_gemm_f16f16_f16f32_f32_tt_t_tilesize128x128x64_stage3_warpsize2x2x1_sptensor16x8x32_execute_kernel_5x_cusparselt |

*NVIDIA GB10 (sm_121), torch 2.9.0a0+145a3a7bda.nv25.10, triton 3.8.0, CUDA 13.0; job 20500, 2026-10-02T09:15:14. Mediana de triton.testing.do_bench.*
