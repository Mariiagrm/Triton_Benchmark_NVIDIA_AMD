"""FlashAttention-2 (forward) de CUTLASS (CuTe DSL): el ejemplo oficial SIN MODIFICAR.

Usa SIN MODIFICAR el ejemplo oficial de NVIDIA
    $CUTLASS_DIR/examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py
(clase FlashAttentionForwardAmpere: MMA warp.MmaF16BF16Op 16x8x16 -> mma.sync, copias
global->compartida con cp.async y ldmatrix; acumulacion en fp32). Es el unico FMHA de CuTe DSL
que corre en Blackwell GeForce (sm_120/121): los de hopper/ y blackwell/ usan wgmma y tcgen05,
que esta arquitectura no tiene. Este modulo solo lo carga y le pasa los tensores como su
funcion run(): from_dlpack(...).mark_layout_dynamic(leading_dim=3)
.mark_compact_shape_dynamic(mode=3, ..., divisibility=128 // 16) y cute.compile(...).

El ejemplo trabaja con (B, N, H, D) contiguo; el banco usa (B, H, N, D) como el tutorial de
Triton, asi que se hace una copia transpuesta de Q, K, V una vez, FUERA de la medida, y la
salida se devuelve como vista (B, H, N, D).

CUTE_DSL_KEEP=ptx (base_dsl/env_manager.py de CuTe DSL) guarda el PTX en el compilado
(atributo __ptx__) para comprobar las instrucciones MMA. Requiere la imagen cutlass.
"""
import importlib.util
import os

os.environ.setdefault("CUTE_DSL_KEEP", "ptx")  # antes de importar cutlass
# Por defecto CuTe DSL vuelca esos ficheros en el directorio actual (la raiz del repo).
os.environ.setdefault("CUTE_DSL_DUMP_DIR", os.path.join(os.environ.get("TFM_RAIZ", "."), ".cache", "cute_dsl"))
os.makedirs(os.environ["CUTE_DSL_DUMP_DIR"], exist_ok=True)

import torch  # noqa: E402

_EJEMPLO = os.path.join("examples", "python", "CuTeDSL", "cute", "ampere", "kernel", "attention",
                        "flash_attention_v2.py")
BLOQUES_POR_DEFECTO = (128, 128, 128)  # m_block_size, n_block_size, num_threads: los del ejemplo
_modulo = None


def ejemplo():
    """Modulo flash_attention_v2.py oficial (cargado una vez desde $CUTLASS_DIR)."""
    global _modulo
    if _modulo is None:
        ruta = os.path.join(os.environ.get("CUTLASS_DIR", "/opt/cutlass"), _EJEMPLO)
        if not os.path.isfile(ruta):
            raise RuntimeError(f"No encuentro el ejemplo de CUTLASS: {ruta} (usa la imagen cutlass)")
        spec = importlib.util.spec_from_file_location("cutlass_ejemplo_fa2", ruta)
        _modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_modulo)
    return _modulo


def _tensor_cute(t):
    from cutlass.cute.runtime import from_dlpack

    # Como create_tensor() de run() en el ejemplo (fp16: divisibility = 128 // 16 = 8).
    return (from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=3, stride_order=t.dim_order(), divisibility=8))


def preparar(q, k, v, causal, escala, bloques=BLOQUES_POR_DEFECTO):
    """Compila el kernel para Q, K, V (B, H, N, D) fp16 y devuelve (fn, compilado).

    fn() lanza solo el kernel y devuelve O (B, H, N, D); compilado.__ptx__ es su PTX.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    assert q.dtype == torch.float16, "el banco mide fp16"
    m_block, n_block, hilos = bloques
    D = q.shape[-1]
    fa2 = ejemplo().FlashAttentionForwardAmpere
    if not fa2.can_implement(cutlass.Float16, D, m_block, n_block, hilos, causal):
        raise ValueError(f"FlashAttentionForwardAmpere.can_implement rechaza D={D} bloques={bloques} causal={causal}")

    qs, ks, vs = (x.transpose(1, 2).contiguous() for x in (q, k, v))  # (B, N, H, D), fuera de la medida
    os_ = torch.empty_like(qs)
    args = [_tensor_cute(x) for x in (qs, ks, vs, os_)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compilado = cute.compile(fa2(D, m_block, n_block, hilos, causal), *args, escala, stream)

    def fn():
        compilado(*args, escala, stream)
        return os_.transpose(1, 2)

    return fn, compilado
