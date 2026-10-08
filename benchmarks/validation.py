"""Verificacion de correccion: la salida de cada kernel se compara con PyTorch.

Todo benchmark valida ANTES de medir; un kernel rapido pero incorrecto no cuenta.
    - comprobar_matmul / comprobar_rmsnorm: abortan (AssertionError) si la salida no
      coincide con la referencia de PyTorch dentro de la tolerancia del dtype.
    - error_rel: error relativo de Frobenius, para precisiones reducidas (FP8, 2:4)
      en las que no se exige tolerancia fija sino que se informa del error.
"""
import torch
import torch.nn.functional as F

# fp16/bf16/tf32 con K grande: la acumulacion en distinto orden que cuBLAS da diferencias ~1e-1 absolutas.
TOL_MATMUL = {"atol": 1e-1, "rtol": 1e-2}
# RMSNorm fp16/bf16 frente a la referencia calculada en fp32.
TOL_RMSNORM = {"atol": 2e-2, "rtol": 2e-2}


def comprobar(salida, referencia, atol, rtol, nombre="kernel"):
    """assert_close con el nombre del kernel en el mensaje de error."""
    try:
        torch.testing.assert_close(salida, referencia, atol=atol, rtol=rtol)
    except AssertionError as e:
        raise AssertionError(f"{nombre}: la salida no coincide con PyTorch.\n{e}") from None


def comprobar_matmul(c, a, b, nombre="matmul"):
    """c debe ser a @ b (referencia: torch.matmul, es decir, cuBLAS)."""
    comprobar(c, torch.matmul(a, b), nombre=nombre, **TOL_MATMUL)


def comprobar_rmsnorm(y, x, w, eps, nombre="rmsnorm"):
    """y debe ser RMSNorm(x) * w (referencia: F.rms_norm calculado en fp32)."""
    ref = F.rms_norm(x.float(), (x.shape[-1],), w.float(), eps).to(y.dtype)
    comprobar(y, ref, nombre=nombre, **TOL_RMSNORM)


def error_rel(c, ref):
    """Error relativo de Frobenius ||c - ref|| / ||ref||, calculado en fp32."""
    c, ref = c.float(), ref.float()
    return (torch.linalg.norm(c - ref) / torch.linalg.norm(ref)).item()
