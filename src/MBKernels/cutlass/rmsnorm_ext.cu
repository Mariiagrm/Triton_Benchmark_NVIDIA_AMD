// RMSNorm de CUTLASS (tools/util/include/cutlass/util/device_rmsnorm.h) como extension
// de PyTorch, para medirlo con el mismo banco que los kernels Python.
//
// Se usa la funcion oficial cutlass::rmsnorm sin modificarla. Un bloque por fila, dos
// pasadas (suma de cuadrados + normalizacion). En fp16 con N % 8 == 0 usa el kernel
// vectorizado rmsnorm_twoPassAlgo_e8 (float4 = 8 half por carga); en cualquier otro
// caso (p. ej. bf16) el escalar rmsnorm_twoPassAlgo_e1.
//
// Sufijo _ext.cu: CMakeLists.txt no lo compila como ejecutable; lo compila en tiempo de
// ejecucion MBKernels/cutlass/rmsnorm.py (torch.utils.cpp_extension).
#include <iostream>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "cutlass/util/device_rmsnorm.h"

template <typename T>
static void lanzar(torch::Tensor y, torch::Tensor x, torch::Tensor w, double eps) {
    const int m = x.size(0);
    const int n = x.size(1);
    using Ref = cutlass::TensorRef<T, cutlass::layout::RowMajor>;
    Ref ref_y(reinterpret_cast<T*>(y.data_ptr()), cutlass::layout::RowMajor(n));
    Ref ref_x(reinterpret_cast<T*>(x.data_ptr()), cutlass::layout::RowMajor(n));
    Ref ref_w(reinterpret_cast<T*>(w.data_ptr()), cutlass::layout::RowMajor(n));
    cutlass::rmsnorm<T>(cutlass::MatrixCoord(m, n), ref_y, ref_x, ref_w,
                        at::cuda::getCurrentCUDAStream().stream(), static_cast<float>(eps));
}

// Kernel que usa cutlass::rmsnorm para estos datos (evidencia para la memoria).
std::string ruta(torch::Tensor x) {
    const bool e8 = x.scalar_type() == at::kHalf && x.size(1) % 8 == 0;
    return e8 ? "rmsnorm_twoPassAlgo_e8" : "rmsnorm_twoPassAlgo_e1";
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, double eps) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x y w deben estar en la GPU");
    TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x debe ser (M, N) contiguo");
    TORCH_CHECK(w.dim() == 1 && w.is_contiguous() && w.size(0) == x.size(1), "w debe ser (N,) contiguo");
    TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x y w deben tener el mismo dtype");
    auto y = torch::empty_like(x);
    switch (x.scalar_type()) {
        case at::kHalf:     lanzar<cutlass::half_t>(y, x, w, eps); break;
        case at::kBFloat16: lanzar<cutlass::bfloat16_t>(y, x, w, eps); break;
        default: TORCH_CHECK(false, "cutlass::rmsnorm: solo fp16 y bf16");
    }
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm, "RMSNorm (cutlass::rmsnorm)");
    m.def("ruta", &ruta, "kernel de CUTLASS que se usaria");
}
