"""Validacion critica del contenedor TFM: compila y ejecuta kernels reales en GPU.

Uso (en hennessy; hay que ejecutarlo como FICHERO, no por stdin, porque
@triton.jit y @helion.kernel necesitan leer el codigo fuente del kernel):
    ~/tfm_entorno/ejecutar.sh validar
    (o a mano: docker run --rm --gpus all -v ~/tfm_entorno:/workspace/tfm tfm:ngc-arm64 \
        python /workspace/tfm/src/validar_gpu.py)
Sale con codigo 1 si alguna comprobacion falla.
"""
import importlib.metadata as md
import os
import subprocess
import sys
import traceback

if not os.path.isfile(globals().get("__file__", "")):
    sys.exit("ERROR: ejecuta este script como fichero (python /ruta/validar_gpu.py), no con 'python - <'.")

# Imports y kernels a nivel de modulo: Triton necesita el fuente en un fichero y
# Helion no admite closures (variables locales capturadas por el kernel).
errores_import = {}
try:
    import torch
except Exception as e:
    errores_import["torch"] = e
try:
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

    @triton.jit
    def dot_kernel(a_ptr, b_ptr, c_ptr, B: tl.constexpr):
        offs = tl.arange(0, B)
        a = tl.load(a_ptr + offs[:, None] * B + offs[None, :])
        b = tl.load(b_ptr + offs[:, None] * B + offs[None, :])
        tl.store(c_ptr + offs[:, None] * B + offs[None, :], tl.dot(a, b))
except Exception as e:
    errores_import["triton"] = e
try:
    import helion
    import helion.language as hl

    # Config fija para no lanzar el autotuner (tarda minutos).
    @helion.kernel(config=helion.Config(block_sizes=[32, 32]))
    def helion_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(out.size()):
            out[tile] = x[tile] + y[tile]
        return out
except Exception as e:
    errores_import["helion"] = e

resultados = []


def requiere(modulo):
    if modulo in errores_import:
        raise errores_import[modulo]


def check(nombre):
    def deco(fn):
        print(f"\n=== {nombre} ===", flush=True)
        try:
            detalle = fn()
            resultados.append((nombre, True, detalle or ""))
            print(f"[OK] {detalle or ''}")
        except Exception as e:
            traceback.print_exc()
            resultados.append((nombre, False, f"{type(e).__name__}: {e}"))
            print(f"[FALLO] {type(e).__name__}: {e}")
        return fn
    return deco


@check("1. PyTorch + CUDA")
def _():
    requiere("torch")
    assert torch.cuda.is_available(), "torch.cuda.is_available() es False"
    x = torch.randn(1024, device="cuda")
    assert torch.allclose((x * 2).cpu(), x.cpu() * 2)
    cap = torch.cuda.get_device_capability(0)
    return (f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
            f"GPU {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")


@check("2. Triton: compilar y ejecutar kernel en GPU (con TRITON_PLUGIN_PATHS activo)")
def _():
    requiere("torch")
    requiere("triton")
    n = 98432
    x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y)
    target = triton.runtime.driver.active.get_current_target()
    return f"triton {triton.__version__}, target {target.backend}/{target.arch}"


@check("3. triton-utlx (plugin TLX)")
def _():
    version = md.version("triton-utlx")
    ruta = os.environ.get("TRITON_PLUGIN_PATHS", "")
    assert ruta, "TRITON_PLUGIN_PATHS no esta definida"
    for p in ruta.split(":"):
        assert os.path.isfile(os.path.realpath(p)), f"plugin no encontrado: {p} -> {os.path.realpath(p)}"
    import ctypes
    for p in ruta.split(":"):
        ctypes.CDLL(p)  # falla si la .so no enlaza con esta arquitectura/libtriton
    import utlx_plugin  # noqa: F401
    # El check 2 ya compilo un kernel con el plugin cargado; si la .so rompiera
    # el pipeline de Triton, habria fallado alli. Se intenta tambien la API TLX:
    try:
        import triton.language.extra.tlx as tlx  # noqa: F401
        api = "API triton.language.extra.tlx disponible"
    except ImportError as e:
        api = f"AVISO: API tlx no importable ({e}); solo validado el plugin nativo"
    return f"triton-utlx {version}, plugin {ruta} carga OK; {api}"


@check("4. Helion: compilar y ejecutar kernel en GPU")
def _():
    requiere("torch")
    requiere("helion")
    x, y = torch.randn(512, 768, device="cuda"), torch.randn(512, 768, device="cuda")
    out = helion_add(x, y)
    torch.cuda.synchronize()
    assert out.is_cuda
    torch.testing.assert_close(out, x + y)
    return f"helion {md.version('helion')}, kernel ejecutado en {out.device}"


@check("5. Tensor cores: tl.dot compila a instrucciones MMA (fp16, bf16, tf32)")
def _():
    requiere("torch")
    requiere("triton")
    import re
    torch.backends.cuda.matmul.allow_tf32 = True  # referencia cuBLAS tambien en tensor cores
    vistas = set()
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        a, b = torch.randn(64, 64, device="cuda", dtype=dtype), torch.randn(64, 64, device="cuda", dtype=dtype)
        c = torch.empty(64, 64, device="cuda", dtype=torch.float32)
        k = dot_kernel[(1,)](a, b, c, B=64)
        torch.testing.assert_close(c, (a.float() @ b.float()), atol=1e-1, rtol=1e-2)
        mma = set(re.findall(r"\b(?:tcgen05\.mma|wgmma\.mma_async|mma\.sync)[\w.]*", k.asm["ptx"]))
        assert mma, f"tl.dot con {dtype} NO genera MMA de tensor core (iria a CUDA cores)"
        vistas |= mma
    return ", ".join(sorted(vistas))


@check("6. Resto de herramientas")
def _():
    import huggingface_hub, jupyterlab, matplotlib, ninja, pandas, pytest  # noqa: F401,E401
    return "huggingface_hub, jupyterlab, matplotlib, ninja, pandas, pytest importan"


# Si Triton fallo, repetir sin el plugin para saber si la culpa es de la .so.
if not resultados[1][1] and os.environ.get("TRITON_PLUGIN_PATHS") and not os.environ.get("VALIDAR_GPU_HIJO"):
    print("\n=== Diagnostico: Triton SIN TRITON_PLUGIN_PATHS ===")
    env = {k: v for k, v in os.environ.items() if k != "TRITON_PLUGIN_PATHS"}
    env["VALIDAR_GPU_HIJO"] = "1"  # evita recursion
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, capture_output=True, text=True)
    linea = next((l for l in r.stdout.splitlines() if "2. Triton" in l and ("OK" in l or "FALLO" in l)), None)
    print(linea or r.stdout[-2000:] or r.stderr[-2000:])
    if linea and linea.strip().startswith("OK"):
        print("=> El plugin TLX rompe la compilacion de Triton (incompatible con este triton).")

print("\n" + "=" * 60 + "\nRESUMEN")
for nombre, ok, detalle in resultados:
    print(f"  {'OK   ' if ok else 'FALLO'} {nombre}\n        {detalle}")
fallos = sum(not ok for _, ok, _ in resultados)
print(f"\n{len(resultados) - fallos}/{len(resultados)} comprobaciones superadas")
sys.exit(1 if fallos else 0)
