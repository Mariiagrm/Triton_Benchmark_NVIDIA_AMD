"""Rendimiento SOSTENIDO: pico de mma.sync y matmul clave con el reloj ya estabilizado.

Motivo (sonda job 20845): bajo carga continua GB10 baja el reloj SM de ~2 410 a ~2 180 MHz tras
~0.5 s, y NVML solo actualiza reloj/potencia cada ~0.5 s. Una medida de do_bench (~0.6 s) cae
sobre todo en el transitorio, asi que ni el pico de run_pico_mma ni los TFLOP/s de los
benchmarks son los de regimen. Aqui cada kernel se ejecuta --segundos seguidos y el
rendimiento, el reloj y la potencia (instantanea de NVML) se calculan SOLO sobre el tramo final,
descartando los primeros --descarte segundos.

Kernels: mma.sync f16->f32 y e4m3->f32 (microbenchmark de src/Microbench, 8 warps/SM x ILP 4),
cuBLAS fp16, Triton y Triton + TMA fp16, cuBLASLt FP8 y Triton FP8, todos a 8192^3. Para cada
uno: TFLOP/s sostenidos, reloj y potencia, y el % del pico de mma.sync AL MISMO RELOJ
(1 024 FLOP/ciclo/SM en fp16, 2 048 en fp8; docs/TFM/pico_mma.md).

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_sostenido [--segundos 6 --descarte 2]
Resultados: results/<maquina>/run_sostenido/<fecha>_job<JOBID>/
"""
import argparse
import os
import shutil
import time

import informe

try:  # --informe regenera tabla y grafica sin GPU ni torch (p. ej. desde el login)
    import torch

    import comun
    from Microbench import pico_mma
except ImportError:
    torch = comun = pico_mma = None

# FLOP por ciclo y SM de mma.sync en sm_12x (run_pico_mma, job 20834).
FLOP_CICLO_SM = {"fp16": 1024, "fp8": 2048}


def sostenido(fn, flops, segundos, descarte):
    """Ejecuta fn en bucle `segundos` y mide solo tras `descarte` s: (TFLOP/s, reloj, potencia)."""
    ritmo, reloj, potencia = comun.sostenido(fn, segundos, descarte)
    return round(ritmo * flops / 1e12, 2), reloj, potencia


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=8192, help="M = N = K de los matmul")
    p.add_argument("--segundos", type=float, default=6.0, help="duracion de cada ejecucion continua")
    p.add_argument("--descarte", type=float, default=2.0, help="segundos iniciales que no se cuentan")
    args = p.parse_args()

    comun.imprimir_contexto()
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    n = args.n
    flops = 2 * n ** 3
    torch.manual_seed(0)
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    s = torch.tensor(1.0, device="cuda")
    a8, b8 = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
    b8_col = b8.t().contiguous().t()  # column-major, como pide cuBLASLt fp8

    from CBKernels.triton.matmul import matmul
    from CBKernels.triton.matmul_fp8 import matmul_fp8
    from CBKernels.triton.matmul_tma import matmul_tma

    casos = []
    for var, prec in (("f16·f32acc", "fp16"), ("e4m3·f32acc", "fp8")):
        if pico_mma.disponible(var):
            iters = 4096
            fn = pico_mma.lanzador(var, 4, n_sm, 8 * 32, iters)
            casos.append((f"mma.sync {var}", prec, fn, n_sm * 8 * iters * 4 * pico_mma.flop_por_mma(var)))
    casos += [
        ("cuBLAS fp16", "fp16", lambda: torch.matmul(a, b), flops),
        ("Triton fp16", "fp16", lambda: matmul(a, b), flops),
        ("Triton + TMA fp16", "fp16", lambda: matmul_tma(a, b), flops),
        ("cuBLASLt FP8", "fp8", lambda: torch._scaled_mm(a8, b8_col, scale_a=s, scale_b=s, out_dtype=torch.float16),
         flops),
        ("Triton FP8", "fp8", lambda: matmul_fp8(a8, b8), flops),
    ]

    filas = []
    for nombre, prec, fn, fl in casos:
        fn()  # autotuning (Triton) fuera de la medida
        torch.cuda.synchronize()
        time.sleep(3)  # reposo: todos parten del mismo estado termico
        tflops, reloj, potencia = sostenido(fn, fl, args.segundos, args.descarte)
        pico = FLOP_CICLO_SM[prec] * n_sm * reloj * 1e6 / 1e12 if reloj else None
        fila = {"kernel": nombre, "precision": prec, "tflops": tflops, "reloj_mhz": reloj, "potencia_w": potencia,
                "pico_mma_a_ese_reloj": round(pico, 1) if pico else None,
                "pct_pico_mismo_reloj": round(100 * tflops / pico, 1) if pico else None}
        filas.append(fila)
        print(f"{nombre:<24} {tflops:8.2f} TFLOP/s  reloj {reloj} MHz  {potencia} W  "
              f"pico a ese reloj {fila['pico_mma_a_ese_reloj']}  -> {fila['pct_pico_mismo_reloj']} %", flush=True)

    comun.guardar(filas, parametros=vars(args), resumen={"n_sm": n_sm, "filas": filas}, informe=generar_informe)


# --- Informe propio: el generico supone tamanos y do_bench, que aqui no aplican -----------------

def _pie(meta):
    c, p = meta.get("contexto", {}), meta.get("parametros", {})
    return (f"{c.get('gpu', '?')} (sm_{c.get('sm', '?')}); job {c.get('slurm_job')}, {c.get('fecha', '')}. "
            f"Régimen sostenido: {p.get('segundos', '?')} s seguidos por kernel tras 3 s de reposo; rendimiento, "
            f"reloj y potencia ({c.get('potencia_fuente') or 'NVML'}) solo de los últimos "
            f"{(p.get('segundos') or 0) - (p.get('descarte') or 0):g} s. Matmul {p.get('n', '?')}^3.")


def grafica(filas, meta, destino):
    import matplotlib.pyplot as plt

    filas = [f for f in filas if isinstance(f.get("tflops"), (int, float))]
    nombres = [f["kernel"] for f in filas][::-1]
    ys = range(len(filas))
    fig, (e1, e2) = plt.subplots(1, 2, figsize=(12, 0.55 * len(filas) + 2.2), sharey=True,
                                 gridspec_kw={"width_ratios": [1.3, 1]})
    # (a) % del pico de mma.sync al reloj medido durante el kernel (un solo color: no hay series).
    pct = [f.get("pct_pico_mismo_reloj") or 0 for f in filas][::-1]
    e1.barh(list(ys), pct, 0.6, color=informe.PALETA[0], zorder=2)
    for y, f in zip(ys, filas[::-1]):
        e1.text(f.get("pct_pico_mismo_reloj") or 0, y, f"  {f.get('pct_pico_mismo_reloj')} %  ({f['tflops']:g} TFLOP/s)",
                va="center", fontsize=8, color=informe.TINTA)
    e1.set_xlim(0, 135)
    e1.set_xticks([0, 25, 50, 75, 100])
    e1.set_xlabel("% del pico de mma.sync al mismo reloj", color=informe.TINTA_2)
    e1.set_title("(a) Eficiencia por ciclo", color=informe.TINTA, fontsize=11, loc="left")
    # (b) reloj sostenido: puntos (no barras) para poder acercar el eje sin truncar barras.
    relojes = [f.get("reloj_mhz") or 0 for f in filas][::-1]
    e2.scatter(relojes, list(ys), s=70, color=informe.PALETA[1], edgecolors="white", linewidths=1.5, zorder=3)
    for y, f in zip(ys, filas[::-1]):
        e2.annotate(f"{f.get('reloj_mhz'):g} MHz · {f.get('potencia_w')} W", (f.get("reloj_mhz") or 0, y),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=informe.TINTA)
    ref = next((f["reloj_mhz"] for f in filas if f["kernel"].startswith("mma.sync") and f.get("reloj_mhz")), None)
    if ref:
        e2.axvline(ref, color=informe.TINTA_2, linewidth=1, linestyle="--", zorder=1)
        e2.text(ref, len(filas) - 0.45, " MMA pura", fontsize=8, color=informe.TINTA_2, va="bottom")
    lo = min(r for r in relojes if r)
    e2.set_xlim(lo - 150, max(relojes) + 450)
    e2.set_xlabel("reloj SM sostenido (MHz) · potencia de GPU", color=informe.TINTA_2)
    e2.set_title("(b) Reloj que sostiene cada kernel", color=informe.TINTA, fontsize=11, loc="left")
    e1.set_yticks(list(ys), nombres)
    for eje in (e1, e2):
        eje.grid(axis="x", color=informe.REJILLA, linewidth=0.8, zorder=0)
        eje.set_ylim(-0.6, len(filas) - 0.2)
        eje.set_axisbelow(True)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            eje.spines[lado].set_color(informe.TINTA_2)
        eje.tick_params(colors=informe.TINTA_2, length=0)
    c = meta.get("contexto", {})
    fig.suptitle(f"Rendimiento sostenido frente al pico de mma.sync — {c.get('gpu', 'GPU')}", color=informe.TINTA,
                 fontsize=12, x=0.02, ha="left")
    fig.text(0.02, 0.01, _pie(meta), fontsize=7, color=informe.TINTA_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(destino, f"grafica.{ext}"), dpi=200, facecolor="white")
    plt.close(fig)


def generar_informe(ejecucion):
    ejecucion = os.path.realpath(ejecucion)
    filas, meta = informe.leer(ejecucion)
    nombre = meta.get("experimento", "run_sostenido")
    maquina = informe.maquina_de(meta, ejecucion)
    columnas = list(dict.fromkeys(k for f in filas for k in f))
    pie = _pie(meta)
    md = informe.tabla_md(filas, meta, columnas).rsplit("\n\n*", 1)[0] + f"\n\n*{pie}*\n"
    tex = informe.tabla_tex(filas, meta, columnas, nombre)
    tex = tex.replace(informe.pie(meta).replace("%", r"\%").replace("_", r"\_"), pie.replace("%", r"\%").replace("_", r"\_").replace("^3", r"$^3$"))
    for nom, txt in (("tabla.md", md), ("tabla.tex", tex)):
        with open(os.path.join(ejecucion, nom), "w") as f:
            f.write(txt)
    grafica(filas, meta, ejecucion)
    docs = os.path.join(informe.DOCS, "TFM", "resultados", maquina)
    os.makedirs(os.path.join(docs, "figuras"), exist_ok=True)
    with open(os.path.join(docs, f"{nombre}.tex"), "w") as f:
        f.write(tex)
    for ext in ("png", "pdf"):
        shutil.copy(os.path.join(ejecucion, f"grafica.{ext}"), os.path.join(docs, "figuras", f"{nombre}.{ext}"))
    origen = os.path.relpath(ejecucion, informe.RAIZ)
    with open(os.path.join(docs, f"{nombre}.md"), "w") as f:
        f.write(f"# {nombre} ({maquina})\n\nGenerado automáticamente desde `{origen}/`. Análisis en "
                f"[../../pico_mma.md](../../pico_mma.md).\n\n![{nombre}](figuras/{nombre}.png)\n\n{md}")
    print(f"Informe generado en {origen}/ y docs/TFM/resultados/{maquina}/{nombre}.md", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--informe":  # regenerar: run_sostenido.py --informe <ejecucion>
        generar_informe(sys.argv[2])
    else:
        main()
