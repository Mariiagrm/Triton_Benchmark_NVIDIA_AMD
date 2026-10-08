"""Banco COMUN de FlashAttention (forward): mismas formas, validacion, medida y guardado
para todos los DSLs.

Cada benchmarks/CBKernels/<dsl>/run_attention_<dsl>.py solo aporta como preparar su kernel;
asi la comparacion entre DSLs es justa y todas las ejecuciones acaban en
results/attention_metrics.csv. Por cada forma se mide el kernel del DSL y, como referencia,
torch.nn.functional.scaled_dot_product_attention (SDPA de PyTorch), en TFLOP/s.

Metodologia (la del benchmark del tutorial oficial de Triton 06-fused-attention.py, v3.8.0):
    - Q, K, V de forma (B, H, N, D), contiguas, fp16, torch.randn; solo forward.
    - B = 4, H = 32, D en {64, 128}, causal y no causal; N variable (el tutorial: 1024..16384).
    - FLOPs = 2 matmul x 2*B*H*N*N*D; la mitad si es causal (solo el triangulo inferior).
    - Escala del softmax 1/sqrt(D) (la de SDPA por defecto).
Validacion: torch.testing.assert_close(o, SDPA, atol=1e-2, rtol=0), la tolerancia del
test_op (forward fp16) del mismo tutorial. Se compara con SDPA y no con la atencion
"ingenua" del test porque esa materializa las matrices N x N (con B=4, H=32, N=16384 en fp32:
4*32*16384^2*4 B = 137 GB, mas que la memoria de la GPU).

    preparar(q, k, v, causal, escala) -> fn    compila/autotunea (cronometrado aparte:
                                               t_compilacion_s); fn() lanza solo el kernel
                                               y devuelve O (B, H, N, D)
"""
import argparse
import math
import time

import torch
import torch.nn.functional as F

import comun
import validation

TOL_ATTENTION = {"atol": 1e-2, "rtol": 0}


def parser(descripcion):
    p = argparse.ArgumentParser(description=descripcion, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=4, help="B (el tutorial de Triton: 4)")
    p.add_argument("--heads", type=int, default=32, help="H (el tutorial de Triton: 32)")
    p.add_argument("--seqlens", type=int, nargs="+", default=[1024, 4096, 16384], help="N: longitud de secuencia")
    p.add_argument("--head-dims", type=int, nargs="+", default=[64, 128], help="D: dimension de cabeza")
    p.add_argument("--causal", choices=["no", "si", "ambos"], default="ambos")
    return p


def flops_attention(B, H, N, D, causal):
    """FLOPs del forward, como el tutorial de Triton: 2 matmul de 2*B*H*N*N*D; x0.5 si causal."""
    f = 2 * (2.0 * B * H * N * N * D)
    return f * 0.5 if causal else f


def ejecutar(args, dsl, preparar, detalle=None):
    """Mide el kernel de `preparar` frente a SDPA y guarda los resultados.

    detalle(q, k, v, causal, fn): texto opcional por forma (configuracion elegida, etc.),
    que se guarda en la columna 'detalle'.
    """
    comun.imprimir_contexto()
    causales = {"no": [False], "si": [True], "ambos": [False, True]}[args.causal]
    B, H = args.batch, args.heads

    filas = []
    resumen = {}
    fallos = []
    for D in args.head_dims:
        for causal in causales:
            for N in args.seqlens:
                torch.manual_seed(0)
                q, k, v = (torch.randn((B, H, N, D), device="cuda", dtype=torch.float16) for _ in range(3))
                escala = 1.0 / math.sqrt(D)
                flops = flops_attention(B, H, N, D, causal)
                bytes_movidos = 4 * B * H * N * D * q.element_size()  # leer Q, K, V y escribir O
                sdpa = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=escala)

                t0 = time.perf_counter()
                nombre = f"attention {dsl} B{B} H{H} N{N} D{D} causal={causal}"
                try:
                    fn = preparar(q, k, v, causal, escala)
                    o = fn()
                    torch.cuda.synchronize()
                    t_compilacion = round(time.perf_counter() - t0, 2)
                    validation.comprobar(o, sdpa(), nombre=nombre, **TOL_ATTENTION)
                    err = round(validation.error_rel(o, sdpa()), 6)
                    info = detalle(q, k, v, causal, fn) if detalle else ""
                except Exception as e:
                    # Una forma que falla (p. ej. el autotuning no encuentra configuracion) no
                    # hace perder las demas: se anota el error y se sigue.
                    error = f"{type(e).__name__}: {e}".splitlines()[0][:300]
                    print(f"== ERROR en {nombre}: {error}\n", flush=True)
                    filas.append({"variante": dsl, "dtype": "fp16", "B": B, "H": H, "N_CTX": N, "HEAD_DIM": D,
                                  "causal": causal, "error": error})
                    fallos.append(nombre)
                    del q, k, v
                    torch.cuda.empty_cache()
                    continue
                kernel_sdpa = ";".join(comun.kernels_cuda(sdpa))

                print(f"== B={B} H={H} N={N} D={D} causal={causal}: 1a llamada {dsl} {t_compilacion} s "
                      f"| error rel. {err:.1e} {info}")
                tflops = {}
                for variante, f in {dsl: fn, "sdpa": sdpa}.items():
                    r = comun.medir(f, flops=flops, bytes_movidos=bytes_movidos)
                    tflops[variante] = r["tflops"]
                    print(f"   [{variante:7s}] {r['ms']:8.3f} ms  {r['tflops']:7.2f} TFLOP/s", flush=True)
                    filas.append({"variante": variante, "dtype": "fp16", "B": B, "H": H, "N_CTX": N, "HEAD_DIM": D,
                                  "causal": causal, **r,
                                  "error_rel": err if variante == dsl else "",
                                  "t_compilacion_s": t_compilacion if variante == dsl else "",
                                  "detalle": info if variante == dsl else kernel_sdpa})
                sp = tflops[dsl] / tflops["sdpa"]
                print(f"   -> {dsl}/SDPA: {sp:.1%}\n", flush=True)
                resumen[f"N{N}_D{D}_{'causal' if causal else 'nocausal'}"] = {
                    **{f"{k}_tflops": v for k, v in tflops.items()}, f"{dsl}_vs_sdpa": round(sp, 4)}
                del q, k, v, o
                torch.cuda.empty_cache()

    comun.guardar(filas, parametros={**vars(args), "dsl": dsl}, resumen={**resumen, "fallos": fallos})
    if fallos:
        raise SystemExit(f"ERROR: han fallado {len(fallos)} formas (anotadas en resultados.csv): {fallos}")
