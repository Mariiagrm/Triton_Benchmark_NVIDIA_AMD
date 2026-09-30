"""RMSNorm en Helion: y = x / sqrt(mean(x^2) + eps) * w.

Basado en el ejemplo oficial pytorch/helion (examples/rms_norm.py, rms_norm_fwd), sin
la salida auxiliar inv_rms (solo hace falta para el backward): asi mueve los mismos
bytes que las demas implementaciones.

En Helion el kernel se escribe con operaciones de PyTorch sobre tiles; hl.tile(m)
reparte las filas en bloques y Helion genera (y autotunea) el kernel Triton. El nivel
de autotuning lo fija HELION_AUTOTUNE_EFFORT (none | quick | full), que Helion lee al
definir el kernel: hay que fijarlo ANTES de importar este modulo.
"""
import torch

import helion
import helion.language as hl


@helion.kernel
def rmsnorm_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(x_tile * x_tile, dim=-1) + eps)
        out[tile_m, :] = (x_tile * inv_rms[:, None] * weight[:].to(torch.float32)).to(out.dtype)
    return out


def rmsnorm(x, w, eps=1e-5):
    """RMSNorm sobre la ultima dimension de x (M x N) con pesos w (N,)."""
    assert x.ndim == 2 and w.shape == (x.shape[1],), "x debe ser (M, N) y w (N,)"
    return rmsnorm_kernel(x, w, eps)
