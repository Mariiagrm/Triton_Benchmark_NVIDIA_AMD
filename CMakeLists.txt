# Compilacion de los kernels C++/CUDA (Cutlass/CuTe) del TFM.
#
# Cada fichero .cu de src/CBKernels/cutlass/ y src/MBKernels/cutlass/ se compila como un
# ejecutable independiente (<familia>_<nombre>, p. ej. CB_matmul). Uso, dentro del contenedor:
#   cmake -S . -B build        (en la imagen cutlass CUTLASS_DIR ya esta definida)
#   cmake --build build -j
#
# GB10 es sm_121; la variante 'a' habilita las instrucciones especificas de la arquitectura.
cmake_minimum_required(VERSION 3.24)
project(tfm_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 121a)
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUTLASS es header-only: basta con sus include/ y tools/util/include/.
set(CUTLASS_DIR "$ENV{CUTLASS_DIR}" CACHE PATH "Raiz del repositorio de CUTLASS")
if(NOT CUTLASS_DIR OR NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "No encuentro CUTLASS: pasa -DCUTLASS_DIR=/ruta/a/cutlass "
                        "(git clone https://github.com/NVIDIA/cutlass.git)")
endif()

find_package(CUDAToolkit REQUIRED)

set(TFM_KERNELS_CUDA 0)
foreach(familia CB MB)
    file(GLOB fuentes CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/${familia}Kernels/cutlass/*.cu")
    foreach(fuente ${fuentes})
        get_filename_component(nombre "${fuente}" NAME_WE)
        set(objetivo "${familia}_${nombre}")
        add_executable(${objetivo} "${fuente}")
        target_include_directories(${objetivo} PRIVATE
            "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/tools/util/include")
        target_compile_options(${objetivo} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
        target_link_libraries(${objetivo} PRIVATE CUDA::cudart CUDA::cublas)
        math(EXPR TFM_KERNELS_CUDA "${TFM_KERNELS_CUDA} + 1")
    endforeach()
endforeach()

if(TFM_KERNELS_CUDA EQUAL 0)
    message(STATUS "No hay kernels .cu en src/*Kernels/cutlass/ todavia: nada que compilar.")
endif()
