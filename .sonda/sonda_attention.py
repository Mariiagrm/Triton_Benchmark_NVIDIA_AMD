"""Sonda TEMPORAL: que implementaciones oficiales de FlashAttention funcionan en esta GPU.

Para cada imagen (TFM_DSL) ejecuta las implementaciones oficiales candidatas en una forma
pequena y las compara con torch SDPA (atol 1e-2, como el tutorial de Triton y el ejemplo de
CuTe). No es un benchmark: solo dice OK / FALLO con el error exacto.
"""
import importlib.util
import math
import os
import subprocess
import sys
import traceback

import torch
import torch.nn.functional as F

AQUI = os.path.dirname(os.path.abspath(__file__))
DSL = os.environ.get("TFM_DSL", "?")
cap = torch.cuda.get_device_capability()
print(f"== SONDA {DSL} | {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} | torch {torch.__version__}")
try:
    import triton
    print(f"   triton {triton.__version__}")
except Exception as e:
    print(f"   triton: {e}")


def cargar(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = m
    spec.loader.exec_module(m)
    return m


def qkv(B, H, N, D, layout="bhsd"):
    torch.manual_seed(0)
    t = [torch.randn((B, H, N, D), device="cuda", dtype=torch.float16) for _ in range(3)]
    return t


def ref(q, k, v, causal, scale):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


def probar(nombre, fn):
    for D in (64, 128):
        for causal in (False, True):
            q, k, v = qkv(1, 2, 1024, D)
            scale = 1.0 / math.sqrt(D)
            try:
                o = fn(q, k, v, causal, scale)
                torch.cuda.synchronize()
                err = (o.float() - ref(q, k, v, causal, scale).float()).abs().max().item()
                ok = torch.allclose(o.float(), ref(q, k, v, causal, scale).float(), atol=1e-2, rtol=0)
                print(f"[{'OK' if ok else 'MAL'}] {nombre} D={D} causal={causal}: max|err|={err:.2e}", flush=True)
            except Exception as e:
                ultima = traceback.format_exception_only(type(e), e)[-1].strip()
                print(f"[FALLO] {nombre} D={D} causal={causal}: {ultima[-900:]}", flush=True)
                if "LLVM" in ultima:
                    return


if DSL == "triton-tlx":
    sys.path.insert(0, "/workspace/tfm/src")
    from CBKernels.triton import fused_attention as fa
    probar("triton tutorial 06-fused-attention", lambda q, k, v, c, s: fa.attention(q, k, v, c, s, False))
    try:
        import utlx_plugin  # noqa: F401  registra triton.language.extra.tlx (utlx_plugin/__init__.py)
        tlx = cargar("tlx_hopper_fa_ws", os.path.join(AQUI, "tlx_hopper_fa_ws.py"))
        probar("tlx hopper_fa_ws (no causal)", lambda q, k, v, c, s: tlx.attention(q, k, v, s) if not c else (_ for _ in ()).throw(RuntimeError("tutorial sin causal")))
    except Exception as e:
        print(f"[FALLO] tlx hopper_fa_ws import: {traceback.format_exception_only(type(e), e)[-1].strip()[:600]}")

elif DSL == "gluon":
    try:
        g = cargar("gluon_attention_forward", os.path.join(AQUI, "gluon_attention_forward.py"))
        probar("gluon examples/01-attention-forward", lambda q, k, v, c, s: g.attention_forward(q, k, v, c, s)[0])
    except Exception as e:
        print(f"[FALLO] gluon import: {traceback.format_exception_only(type(e), e)[-1].strip()[:600]}")

elif DSL == "helion":
    rev = subprocess.run(["git", "-C", "/workspace/helion", "log", "-1", "--format=%h %cs"], capture_output=True, text=True).stdout.strip()
    print(f"   helion /workspace/helion @ {rev}")
    os.environ["HELION_AUTOTUNE_EFFORT"] = "none"
    sys.path.insert(0, "/workspace/helion")
    ex = cargar("helion_examples_attention", "/workspace/helion/examples/attention.py")
    kernels = [n for n in dir(ex) if "attention" in n and not n.startswith("_")]
    print(f"   funciones en examples/attention.py: {kernels}")
    if hasattr(ex, "attention_output") and hasattr(ex, "causal_attention_output"):
        probar("helion examples/attention.py (attention_output / causal_attention_output)",
               lambda q, k, v, c, s: (ex.causal_attention_output if c else ex.attention_output)(q, k, v))
    else:
        import inspect
        print(inspect.signature(ex.attention))
        probar("helion examples/attention.py (attention)", lambda q, k, v, c, s: ex.attention(q, k, v) if not c else (_ for _ in ()).throw(RuntimeError("sin causal")))

elif DSL == "cutlass":
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack
    print(f"   nvidia-cutlass-dsl {getattr(cutlass, '__version__', '?')}")
    ruta = os.path.join(os.environ["CUTLASS_DIR"], "examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py")
    fa2 = cargar("cute_fa2", ruta)

    def cute_t(t):
        return (from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=3)
                .mark_compact_shape_dynamic(mode=3, stride_order=t.dim_order(), divisibility=8))

    def fn(q, k, v, causal, scale):
        # el ejemplo usa (B, S, H, D) contiguo
        qs, ks, vs = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        os_ = torch.empty_like(qs)
        D = q.shape[-1]
        assert fa2.FlashAttentionForwardAmpere.can_implement(cutlass.Float16, D, 128, 128, 128, causal)
        kern = fa2.FlashAttentionForwardAmpere(D, 128, 128, 128, causal)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = [cute_t(x) for x in (qs, ks, vs, os_)]
        comp = cute.compile(kern, *args, scale, stream)
        comp(*args, scale, stream)
        return os_.transpose(1, 2)

    probar("cutlass CuTeDSL ampere/flash_attention_v2.py", fn)

print("== FIN SONDA")
