"""Matmul en Helion: C = A @ B, acumulacion fp32.

Es el kernel `matmul` del ejemplo oficial pytorch/helion (examples/matmul.py) sin el
parametro `epilogue` (identidad aqui): mismo decorador (static_shapes=True y los
autotune_config_overrides del ejemplo) y mismo cuerpo. hl.tile([m, n]) reparte C en
bloques, el bucle en K acumula con torch.addmm en fp32 y Helion genera y autotunea el
kernel Triton (tl.dot -> tensor cores).

El esfuerzo de autotuning lo fija HELION_AUTOTUNE_EFFORT (none | quick | full), que Helion
lee al definir el kernel: hay que fijarlo ANTES de importar este modulo.
"""
import torch
from torch import Tensor

import helion
from helion._compat import use_tileir_tunables
import helion.language as hl


@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    }
    if not use_tileir_tunables()
    else {},
)
def matmul(x: Tensor, y: Tensor) -> Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out
