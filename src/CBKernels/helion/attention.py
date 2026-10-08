"""FlashAttention (forward) de Helion: el ejemplo oficial SIN MODIFICAR.

Carga /workspace/helion/examples/attention.py (el repositorio de Helion instalado en la
imagen helion) y usa sus kernels que devuelven solo la salida:
    attention_output(q, k, v)          no causal
    causal_attention_output(q, k, v)   causal
Ambos: Q, K, V de forma (..., N, D), escala 1/sqrt(D) fija, softmax "online" en base 2 y
acumulacion en fp32 (algoritmo de FlashAttention). El kernel causal recorre TODOS los bloques
de K/V y enmascara con -inf (no salta el triangulo superior): hace el trabajo del no causal.

Requiere la imagen helion. HELION_AUTOTUNE_EFFORT debe fijarse antes de importar este modulo.
"""
import importlib.util
import os
import sys

_RUTA = os.path.join(os.environ.get("HELION_DIR", "/workspace/helion"), "examples", "attention.py")
_modulo = None


def ejemplo():
    """Modulo examples/attention.py oficial (cargado una vez)."""
    global _modulo
    if _modulo is None:
        if not os.path.isfile(_RUTA):
            raise RuntimeError(f"No encuentro el ejemplo de Helion: {_RUTA} (usa la imagen helion)")
        spec = importlib.util.spec_from_file_location("helion_ejemplo_attention", _RUTA)
        _modulo = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _modulo
        spec.loader.exec_module(_modulo)
    return _modulo


def kernel(causal):
    """El kernel de Helion (objeto @helion.kernel) para el caso causal o no causal."""
    ej = ejemplo()
    return ej.causal_attention_output if causal else ej.attention_output
