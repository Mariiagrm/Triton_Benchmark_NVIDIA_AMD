"""Sonda TEMPORAL: ¿que cambia el PR triton-lang/triton#9734 en sm_120/121?

El PR ("[NVIDIA] Fix PTX codegen segfaults on consumer Blackwell (sm_120)") sostiene que
`sm_arch_from_capability` no deberia anadir la "a" en Blackwell GeForce: con `sm_120a`, LLVM y
ptxas generarian instrucciones de memoria de tensor (tcgen05) inexistentes y el kernel daria
SIGSEGV. Se integro el 2026-03-16 y se revirtio al dia siguiente (#9755) porque NVIDIA no lo
considero correcto. Solo toca Python (third_party/nvidia/backend/compiler.py), asi que se aplica
en caliente al Triton 3.8.0 de la imagen, sin recompilar. Variantes:

    base       Triton 3.8.0 sin cambios: sm_12xa en LLVM, PTX y ptxas.
    fusionado  version integrada (a657bf8): sin "a" SOLO si capability == 120. En GB10 (121)
               no cambia nada; se mide igualmente para comprobarlo.
    original   primera version del PR (456da33): sin "a" en todo sm_12x, sm_12x por el
               pipeline de Ampere/Hopper en make_ttgir y la regex de .target con la "a".

Cada variante corre en un proceso nuevo con TRITON_CACHE_DIR vacio: la clave de la cache de
Triton incluye el hash de compiler.py, que el parche en memoria no cambia, asi que con la cache
compartida se reutilizarian binarios de otra variante.

Para cada kernel: compila / error, .target del PTX, si aparece tcgen05, error frente a PyTorch y
TFLOP/s (o GB/s). No es un benchmark: una medida de do_bench, en 4096 (ver bitacora para cifras).

Uso (imagen gluon: Triton 3.8.0 + Gluon, sin plugin TLX):
    DSL=gluon VALIDAR=0 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp .sonda/sonda_pr9734.py
"""
import inspect
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import traceback

VARIANTES = ["base", "fusionado", "original"]


# ---------------------------------------------------------------------------------------------
# Parches del PR sobre triton.backends.nvidia.compiler (Triton 3.8.0)
# ---------------------------------------------------------------------------------------------
def aplicar(variante):
    import triton.backends.nvidia.compiler as nv
    if variante == "base":
        return
    if variante == "fusionado":
        def sm_arch(capability: int):
            suffix = "a" if capability >= 90 and capability != 120 else ""
            return f"sm_{capability}{suffix}"
        nv.sm_arch_from_capability = sm_arch
        return
    assert variante == "original", variante

    def sm_arch(capability: int):
        suffix = "a" if 90 <= capability < 120 else ""
        return f"sm_{capability}{suffix}"
    nv.sm_arch_from_capability = sm_arch

    def reescribir(fn, cambios):
        src = textwrap.dedent(inspect.getsource(fn))
        src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("@staticmethod"))
        for viejo, nuevo in cambios:
            assert src.count(viejo) == 1, f"no encuentro en {fn.__name__}: {viejo!r} (¿otra version de Triton?)"
            src = src.replace(viejo, nuevo)
        espacio = {}
        exec(compile(src, f"<pr9734 {fn.__name__}>", "exec"), nv.__dict__, espacio)
        return espacio[fn.__name__]

    ttgir = reescribir(nv.CUDABackend.make_ttgir, [
        ("if capability // 10 in [8, 9]:", "if capability // 10 in [8, 9] or capability >= 120:"),
        ("elif capability // 10 >= 10:", "elif 100 <= capability < 120:"),
    ])
    nv.CUDABackend.make_ttgir = staticmethod(ttgir)
    nv.CUDABackend.make_ptx = reescribir(nv.CUDABackend.make_ptx, [
        ("re.sub(r'\\.target sm_\\d+', f'.target sm_{capability}'", "re.sub(r'\\.target sm_\\d+a?', f'.target {proc}'"),
    ])


# ---------------------------------------------------------------------------------------------
# Pruebas (proceso hijo)
# ---------------------------------------------------------------------------------------------
def info_ptx(k):
    """(.target, '¿tcgen05?', '¿tmem en ttgir?') del kernel compilado."""
    asm = getattr(k, "asm", {}) or {}
    ptx = asm.get("ptx", "")
    tgt = re.search(r"\.target (\S+)", ptx)
    ttgir = str(asm.get("ttgir", ""))
    return (tgt.group(1) if tgt else "?", "tcgen05" in ptx,
            ("ttng.tmem_alloc" in ttgir) or ("tc_gen5_mma" in ttgir))


def mma_ptx(k):
    ptx = (getattr(k, "asm", {}) or {}).get("ptx", "")
    return sorted(set(re.findall(r"mma\.sync\.aligned\.[\w.:]+", ptx)))[:3]


def probar(nombre, fn):
    """fn() -> (kernel compilado o None, error relativo o None, rendimiento str)."""
    import torch
    try:
        k, err, rend = fn()
        torch.cuda.synchronize()
        tgt, tc5, tmem = info_ptx(k) if k is not None else ("?", False, False)
        e = f"err {err:.1e}" if err is not None else "err —"
        print(f"[OK]    {nombre:<28} .target {tgt:<9} tcgen05={'SI' if tc5 else 'no'} tmem={'SI' if tmem else 'no'} "
              f"| {e} | {rend} | {', '.join(mma_ptx(k)) if k is not None else ''}", flush=True)
    except Exception as ex:
        ultima = traceback.format_exception_only(type(ex), ex)[-1].strip().replace("\n", " ")
        print(f"[FALLO] {nombre:<28} {ultima[:700]}", flush=True)


def hijo(variante):
    import torch
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    aplicar(variante)
    import triton.backends.nvidia.compiler as nv
    cap = torch.cuda.get_device_capability()
    cap = cap[0] * 10 + cap[1]
    print(f"\n######## variante={variante} | sm_arch_from_capability({cap}) = {nv.sm_arch_from_capability(cap)} "
          f"| cache {os.environ.get('TRITON_CACHE_DIR')}", flush=True)
    bench = triton.testing.do_bench
    torch.manual_seed(0)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    # 1. El test del propio PR (solo compilar), con la capability real y con 120.
    @triton.jit
    def simple_dot(a_base, b_base, out):
        SIZE: tl.constexpr = 64
        a_ptr = a_base + tl.arange(0, SIZE)[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
        b_ptr = b_base + tl.arange(0, SIZE)[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
        c = tl.dot(tl.load(a_ptr), tl.load(b_ptr))
        out_ptr = out + tl.arange(0, SIZE)[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
        tl.store(out_ptr, c)

    for c in sorted({cap, 120}):
        def test_pr(c=c):
            k = triton.compile(triton.compiler.ASTSource(
                fn=simple_dot, signature={"a_base": "*fp16", "b_base": "*fp16", "out": "*fp16"}, constexprs={}),
                target=GPUTarget("cuda", c, 32))
            assert k.asm["cubin"] != b""
            return k, None, "solo compilar"
        probar(f"test del PR (cuda:{c})", test_pr)

    n = 4096
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    ref = a @ b
    tf = lambda ms: f"{2 * n ** 3 / (ms * 1e-3) / 1e12:6.1f} TFLOP/s"

    # 2. Matmul Triton (autotune), con TMA (descriptores en el kernel: tensormap) y FP8.
    def mm():
        from CBKernels.triton.matmul import matmul
        c, k = matmul(a, b, devolver_kernel=True)
        return k, rel(c, ref), tf(bench(lambda: matmul(a, b)))
    probar("Triton matmul fp16", mm)

    def mm_tma():
        from CBKernels.triton.matmul_tma import matmul_tma
        c, k = matmul_tma(a, b, devolver_kernel=True)
        return k, rel(c, ref), tf(bench(lambda: matmul_tma(a, b)))
    probar("Triton matmul TMA fp16", mm_tma)

    def mm_fp8():
        from CBKernels.triton.matmul_fp8 import matmul_fp8
        a8, b8 = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
        c, k = matmul_fp8(a8, b8, devolver_kernel=True)
        return k, rel(c, a8.float() @ b8.float()), tf(bench(lambda: matmul_fp8(a8, b8)))
    probar("Triton matmul FP8", mm_fp8)

    # 3. FlashAttention del tutorial (descriptores TMA de host en capability >= 9).
    def attn():
        import torch.nn.functional as F
        from CBKernels.triton.fused_attention import attention
        B, H, N, D = 4, 32, 4096, 128
        q, kk, v = (torch.randn((B, H, N, D), device="cuda", dtype=torch.float16) for _ in range(3))
        o = attention(q, kk, v, False, D ** -0.5, False)
        r = F.scaled_dot_product_attention(q, kk, v, scale=D ** -0.5)
        ms = bench(lambda: attention(q, kk, v, False, D ** -0.5, False))
        return None, rel(o, r), f"{4 * B * H * N * N * D / (ms * 1e-3) / 1e12:6.1f} TFLOP/s"
    probar("Triton FlashAttention fp16", attn)

    # 4. Gluon: matmul propio (mma_v2 + cp.async) y RMSNorm.
    def g_mm():
        from CBKernels.gluon.matmul import matmul as gmm
        c, k = gmm(a, b, devolver_kernel=True)
        return k, rel(c, ref), tf(bench(lambda: gmm(a, b)))
    probar("Gluon matmul fp16", g_mm)

    x = torch.randn((16384, 8192), device="cuda", dtype=torch.float16)
    w = torch.randn((8192,), device="cuda", dtype=torch.float16)
    ref_rms = torch.nn.functional.rms_norm(x.float(), (8192,), w.float(), 1e-5)
    gbs = lambda ms: f"{(2 * x.numel() + w.numel()) * 2 / (ms * 1e-3) / 1e9:6.1f} GB/s"
    for nombre, modulo in (("Triton RMSNorm fp16", "MBKernels.triton.rmsnorm_baseline"),
                           ("Gluon RMSNorm fp16", "MBKernels.gluon.rmsnorm")):
        def rms(modulo=modulo):
            f = __import__(modulo, fromlist=["rmsnorm"]).rmsnorm
            y, k = f(x, w, devolver_kernel=True)
            return k, rel(y, ref_rms), gbs(bench(lambda: f(x, w)))
        probar(nombre, rms)

    # 5. tl.dot_scaled MXFP4 (e2m1 con escala ue8m0): las MMA block-scaled solo existen en sm_12xa.
    @triton.jit
    def dot_mx(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        rm, rn, rk2, rks = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K // 2), tl.arange(0, K // 32)
        av = tl.load(a_ptr + rm[:, None] * (K // 2) + rk2[None, :])
        bv = tl.load(b_ptr + rk2[:, None] * N + rn[None, :])
        sa = tl.load(as_ptr + rm[:, None] * (K // 32) + rks[None, :])
        sb = tl.load(bs_ptr + rn[:, None] * (K // 32) + rks[None, :])
        c = tl.dot_scaled(av, sa, "e2m1", bv, sb, "e2m1")
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)

    def mx():
        M = N = K = 128
        ap = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
        bp = torch.randint(0, 256, (K // 2, N), device="cuda", dtype=torch.uint8)
        sa = torch.full((M, K // 32), 127, device="cuda", dtype=torch.uint8)
        sb = torch.full((N, K // 32), 127, device="cuda", dtype=torch.uint8)
        c = torch.empty((M, N), device="cuda", dtype=torch.float32)
        k = dot_mx[(1,)](ap, sa, bp, sb, c, M, N, K)
        return k, None, f"suma de C = {c.sum().item():.6g} (debe coincidir entre variantes)"
    probar("tl.dot_scaled MXFP4", mx)


# ---------------------------------------------------------------------------------------------
# Proceso padre: una variante por proceso, cada una con su cache de Triton vacia
# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        hijo(sys.argv[1])
        sys.exit(0)
    import torch
    import triton
    print(f"== SONDA PR 9734 | {torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability()))} "
          f"| torch {torch.__version__} | triton {triton.__version__} | {os.environ.get('TFM_DSL', '?')}", flush=True)
    estados = {}
    for v in VARIANTES:
        with tempfile.TemporaryDirectory(prefix=f"triton-{v}-") as cache:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), v],
                               env={**os.environ, "TRITON_CACHE_DIR": cache})
            estados[v] = r.returncode
            if r.returncode != 0:
                print(f"[FALLO] variante {v}: el proceso termino con codigo {r.returncode}"
                      f"{' (SIGSEGV)' if r.returncode == -11 else ''}", flush=True)
    print(f"\n== FIN SONDA: codigos de salida {estados}", flush=True)
