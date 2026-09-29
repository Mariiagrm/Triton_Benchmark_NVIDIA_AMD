# Imagen base NVIDIA NGC de PyTorch (multi-arch: amd64 y arm64/aarch64).
# hennessy es aarch64: pytorch/pytorch:*-cuda* solo existe para amd64 y da
# "exec format error". La imagen NGC trae ya torch, triton, CUDA, cuDNN,
# jupyterlab, pytest, ninja... compilados para la arquitectura del nodo.
# 25.10 -> PyTorch 2.9 (Helion requiere PyTorch >= 2.9).
FROM nvcr.io/nvidia/pytorch:25.10-py3

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
RUN pip install triton-utlx

RUN git clone https://github.com/pytorch/helion.git /workspace/helion \
    && pip install -e /workspace/helion

# Enlace a ruta fija: site-packages cambia segun la imagen.
RUN ln -s "$(python -c 'import os, utlx_plugin; print(os.path.dirname(utlx_plugin.__file__))')/libutlx.so" /opt/libutlx.so
ENV TRITON_PLUGIN_PATHS="/opt/libutlx.so"
