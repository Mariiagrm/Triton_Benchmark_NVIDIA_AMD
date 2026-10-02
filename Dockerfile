# Una imagen por DSL, todas desde este Dockerfile multi-stage (se elige con --target):
#   docker build --target triton-tlx -t tfm-triton-tlx:ngc-arm64 .   # Triton estandar + TLX
#   docker build --target gluon      -t tfm-gluon:ngc-arm64 .        # Gluon
#   docker build --target helion     -t tfm-helion:ngc-arm64 .       # Helion
#   docker build --target cutlass    -t tfm-cutlass:ngc-arm64 .      # Cutlass/CuTe (C++/CUDA)
# ejecutar.sh lo hace solo:  ejecutar.sh imagen [dsl|todas]
#
# Base comun: NVIDIA NGC PyTorch (multi-arch: amd64 y arm64/aarch64). hennessy es aarch64:
# pytorch/pytorch:*-cuda* solo existe para amd64 y da "exec format error". La imagen NGC
# trae ya torch, triton, CUDA, cuDNN... compilados para la arquitectura del nodo.
# 25.10 -> PyTorch 2.9 (Helion requiere PyTorch >= 2.9). Todas las imagenes necesitan
# torch: la referencia de validacion (benchmarks/validation.py) y cuBLAS son PyTorch.
FROM nvcr.io/nvidia/pytorch:25.10-py3 AS base

WORKDIR /workspace/tfm

RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# NO se instala torch ni triton: vienen en la imagen NGC.
RUN python -m pip install --upgrade pip
RUN pip install pandas matplotlib jupyterlab pytest ninja

RUN pip install huggingface_hub

# Mismo Triton en TODAS las imagenes: triton-utlx exige triton~=3.8.0 y, al instalarlo,
# pip sustituye el Triton de NGC por el de PyPI. Sin fijarlo aqui, triton-tlx compilaria
# con Triton 3.8.0 y gluon/helion/cutlass con el de NGC (otra version): la comparacion
# entre DSLs no seria justa. 3.8.0 es la version con la que se midieron los resultados.
RUN pip install "triton~=3.8.0"


# --- Triton estandar + TLX (Triton Language Extensions) -------------------------------
FROM base AS triton-tlx
RUN pip install triton-utlx
# Enlace a ruta fija: site-packages cambia segun la imagen.
RUN ln -s "$(python -c 'import os, utlx_plugin; print(os.path.dirname(utlx_plugin.__file__))')/libutlx.so" /opt/libutlx.so
ENV TRITON_PLUGIN_PATHS="/opt/libutlx.so"
ENV TFM_DSL=triton-tlx


# --- Gluon (triton.experimental.gluon: viene con el Triton de NGC, sin plugin TLX) -----
FROM base AS gluon
ENV TFM_DSL=gluon


# --- Helion (DSL de PyTorch) ------------------------------------------------------------
FROM base AS helion
RUN git clone https://github.com/pytorch/helion.git /workspace/helion \
    && pip install -e /workspace/helion
ENV TFM_DSL=helion


# --- Cutlass/CuTe (C++/CUDA, header-only; se compila con el CMakeLists.txt del repo) ----
FROM base AS cutlass
ARG CUTLASS_VERSION=v4.8.0
RUN git clone --depth 1 --branch ${CUTLASS_VERSION} https://github.com/NVIDIA/cutlass.git /opt/cutlass
RUN pip install "cmake>=3.24"
# CuTe DSL (Python) en la MISMA version que el CUTLASS clonado, con las librerias de CUDA 13
# (la de NGC 25.10). No depende de torch: no toca el PyTorch de NGC. Lo usa el matmul
# CBKernels/cutlass/matmul_cute.py (ejemplo oficial blackwell_geforce/dense_gemm.py).
RUN pip install "nvidia-cutlass-dsl[cu13]==${CUTLASS_VERSION#v}"
ENV CUTLASS_DIR=/opt/cutlass
ENV TFM_DSL=cutlass
