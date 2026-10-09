"""Pico REAL de los tensor cores (mma.sync) en esta GPU: microbenchmark sin memoria.

Todos los kernels de los DSLs en sm_120/sm_121 acaban en mma.sync (no hay wgmma ni
tcgen05), asi que su techo no es la cifra de catalogo sino lo que da esa instruccion. Aqui
se mide directamente: cada warp encadena MMAs sobre registros (src/Microbench/pico_mma_ext.cu),
barriendo warps por SM y acumuladores independientes (ILP) hasta saturar la unidad MMA.

Para cada instruccion (f16/bf16/tf32/fp8/int8 y, si ptxas las acepta en sm_12Xa, las
block-scaled MXFP8/MXFP4/NVFP4) da:
  - TFLOP/s maximos y la configuracion que los alcanza (warps/SM x ILP),
  - reloj SM y potencia durante la medida (NVML), y FLOP/ciclo/SM = TFLOP/s / (SMs x reloj),
    la cifra intrinseca de la arquitectura, independiente del reloj,
  - ciclos por MMA y warp (con 1 warp e ILP=1: la latencia de la instruccion),
  - el opcode SASS que la ejecuta (HMMA/QMMA/...), como evidencia de que es tensor core.

Uso:
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_pico_mma
    bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_pico_mma --variantes "f16·f32acc" --warps 4 8
Resultados: results/<maquina>/run_pico_mma/<fecha>_job<JOBID>/  (+ docs/TFM/resultados/<maquina>/run_pico_mma.md)
"""
import argparse
import os
import shutil

import informe

try:  # --informe regenera tablas y grafica sin GPU ni torch (p. ej. desde el login)
    import torch

    import comun
    from Microbench import pico_mma
except ImportError:
    torch = comun = pico_mma = None


def calibrar(variante, ilp, bloques, hilos, objetivo_ms):
    """Iteraciones para que un lanzamiento dure ~objetivo_ms."""
    iters = 256
    while True:
        ms = comun.medir(pico_mma.lanzador(variante, ilp, bloques, hilos, iters), warmup=5, rep=20)["ms"]
        if ms >= objetivo_ms / 4 or iters >= 1 << 22:
            return max(16, int(iters * objetivo_ms / ms))
        iters *= 4


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variantes", nargs="+", choices=list(pico_mma.VARIANTES), default=list(pico_mma.VARIANTES))
    parser.add_argument("--warps", type=int, nargs="+", default=[1, 4, 8, 16, 32], help="warps por SM (un bloque por SM)")
    parser.add_argument("--ilp", type=int, nargs="+", choices=[1, 2, 4, 8], default=[1, 2, 4, 8],
                        help="acumuladores independientes por warp")
    parser.add_argument("--ms", type=float, default=2.0, help="duracion objetivo de cada lanzamiento")
    args = parser.parse_args()

    comun.imprimir_contexto()
    props = torch.cuda.get_device_properties(0)
    n_sm = props.multi_processor_count
    lib = pico_mma.biblioteca()
    sass = pico_mma.sass_mma()
    errores = pico_mma.errores()
    print(f"SMs: {n_sm} | {pico_mma.arquitectura()} | {lib.ruta}\n", flush=True)

    m = comun.Muestreador()
    reloj_max = m.reloj_maximo()
    filas = []
    for var in args.variantes:
        instr = pico_mma.VARIANTES[var][2]
        unidad = pico_mma.VARIANTES[var][5]
        if not pico_mma.disponible(var):
            print(f"[{var}] NO disponible: {errores.get(var, '?')}\n", flush=True)
            filas.append({"variante": var, "instruccion": instr, "error": errores.get(var, "no compila")})
            continue
        if not sass[var]:
            raise RuntimeError(f"{var}: el SASS no contiene ninguna instruccion MMA (no usaria tensor cores)")
        print(f"[{var}] {instr}  SASS: {', '.join(sass[var])}", flush=True)
        for w in args.warps:
            for ilp in args.ilp:
                hilos = 32 * w
                try:
                    iters = calibrar(var, ilp, n_sm, hilos, args.ms)
                    fn = pico_mma.lanzador(var, ilp, n_sm, hilos, iters)
                    mmas = n_sm * w * iters * ilp
                    r = comun.medir(fn, flops=mmas * pico_mma.flop_por_mma(var), warmup=50, rep=300)
                    reloj, potencia = r.get("reloj_mhz"), r.get("potencia_w")
                except Exception as e:  # p. ej. demasiados registros para 32 warps x ILP 8
                    print(f"   warps/SM={w:>2} ILP={ilp}: ERROR {type(e).__name__}: {e}", flush=True)
                    filas.append({"variante": var, "instruccion": instr, "warps_sm": w, "ilp": ilp,
                                  "error": f"{type(e).__name__}: {e}"})
                    continue
                fila = {"variante": var, "instruccion": instr, "unidad": unidad, "sass": ";".join(sass[var]),
                        "warps_sm": w, "ilp": ilp, "mma_en_vuelo_sm": w * ilp, "iters": iters, **r}
                if reloj:
                    ciclos = r["ms"] * 1e-3 * reloj * 1e6
                    fila["flop_ciclo_sm"] = round(r["tflops"] * 1e12 / (n_sm * reloj * 1e6), 1)
                    fila["ciclos_mma_warp"] = round(ciclos / (iters * ilp), 2)
                if potencia:
                    fila["tflops_por_w"] = round(r["tflops"] / potencia, 3)
                filas.append(fila)
                print(f"   warps/SM={w:>2} ILP={ilp}: {r['tflops']:8.2f} {unidad}  reloj {reloj} MHz  "
                      f"{potencia} W  {fila.get('flop_ciclo_sm', '?')} FLOP/ciclo/SM  "
                      f"{fila.get('ciclos_mma_warp', '?')} ciclos/MMA/warp", flush=True)
        print(flush=True)

    medidas = [f for f in filas if "tflops" in f]
    pico = {}
    for f in medidas:
        if f["variante"] not in pico or f["tflops"] > pico[f["variante"]]["tflops"]:
            pico[f["variante"]] = f
    print("*** Pico por instruccion ***")
    for var, f in pico.items():
        print(f"   {var:<12} {f['tflops']:8.2f} {f['unidad']}  (warps/SM={f['warps_sm']}, ILP={f['ilp']}, "
              f"{f.get('flop_ciclo_sm', '?')} FLOP/ciclo/SM, {f['reloj_mhz']} MHz)", flush=True)
    comun.guardar(filas, parametros=vars(args), informe=generar_informe,
                  resumen={"n_sm": n_sm, "arquitectura": pico_mma.arquitectura(), "reloj_max_mhz": reloj_max,
                           "pico": {v: {k: f.get(k) for k in ("tflops", "warps_sm", "ilp", "reloj_mhz",
                                                              "potencia_w", "flop_ciclo_sm", "sass")}
                                    for v, f in pico.items()},
                           "no_disponibles": errores})


# --- Informe propio: tabla del pico por instruccion + grafica de saturacion -------------------

def _picos(filas):
    pico = {}
    for f in filas:
        if isinstance(f.get("tflops"), (int, float)) and (f["variante"] not in pico
                                                           or f["tflops"] > pico[f["variante"]]["tflops"]):
            pico[f["variante"]] = f
    return pico


def _latencia(filas, var):
    f = next((f for f in filas if f["variante"] == var and f.get("warps_sm") == 1 and f.get("ilp") == 1), None)
    return f.get("ciclos_mma_warp", "") if f else ""


def tabla_pico(filas):
    pico = _picos(filas)
    out = []
    for var in dict.fromkeys(f["variante"] for f in filas):
        if var in pico:
            f = pico[var]
            out.append({"instrucción": var, "PTX": f["instruccion"], "SASS": f.get("sass", ""),
                        "pico": f"{f['tflops']:g} {f['unidad']}", "warps/SM × ILP": f"{f['warps_sm']} × {f['ilp']}",
                        "FLOP/ciclo/SM": f.get("flop_ciclo_sm", ""), "latencia (ciclos)": _latencia(filas, var),
                        "reloj (MHz)": f.get("reloj_mhz", ""), "potencia (W)": f.get("potencia_w", "")})
        else:
            f = next(f for f in filas if f["variante"] == var)
            out.append({"instrucción": var, "PTX": f["instruccion"], "pico": "no disponible",
                        "SASS": str(f.get("error", ""))[:60]})
    return out


def grafica_pico(filas, meta, destino):
    import matplotlib.pyplot as plt

    pico = _picos(filas)
    if not pico:
        return False
    c = meta.get("contexto", {})
    fig, (e1, e2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) Pico por instruccion: barras horizontales de un solo color (no hay series), ordenadas.
    orden = sorted(pico.values(), key=lambda f: f["tflops"])
    ys = range(len(orden))
    e1.barh(list(ys), [f["tflops"] for f in orden], 0.6, color=informe.PALETA[0], zorder=2)
    for y, f in zip(ys, orden):
        e1.text(f["tflops"], y, f"  {f['tflops']:.0f}", va="center", fontsize=8, color=informe.TINTA)
    e1.set_yticks(list(ys), [f["variante"] for f in orden])
    e1.set_xlabel("TFLOP/s (TOP/s en int8), mayor es mejor", color=informe.TINTA_2)
    e1.set_xlim(0, max(f["tflops"] for f in orden) * 1.15)
    e1.set_title("(a) Pico medido por instrucción", color=informe.TINTA, fontsize=11, loc="left")

    # (b) Saturacion: % del pico frente a warps por SM, una serie por ILP. Se normaliza al pico de
    # cada instruccion, asi que vale para todas; se dibuja la de referencia (f16 -> f32).
    ref = "f16·f32acc" if "f16·f32acc" in pico else next(iter(pico))
    ilps = sorted({f["ilp"] for f in filas if f["variante"] == ref and isinstance(f.get("tflops"), (int, float))})
    for i, ilp in enumerate(ilps):
        pts = sorted((f["warps_sm"], 100 * f["tflops"] / pico[ref]["tflops"]) for f in filas
                     if f["variante"] == ref and f.get("ilp") == ilp and isinstance(f.get("tflops"), (int, float)))
        e2.plot([x for x, _ in pts], [y for _, y in pts], color=informe.PALETA[i], linewidth=2, marker="o",
                markersize=5, label=f"ILP={ilp}", zorder=2)
    e2.set_xscale("log", base=2)
    xs = sorted({f["warps_sm"] for f in filas if f["variante"] == ref and "warps_sm" in f})
    e2.set_xticks(xs, [str(x) for x in xs])
    e2.set_xlabel("Warps por SM (un bloque por SM; 4 subparticiones)", color=informe.TINTA_2)
    e2.set_ylabel(f"% del pico de {ref}", color=informe.TINTA_2)
    e2.set_ylim(0, 105)
    e2.set_title(f"(b) Saturación ({ref}): hacen falta ≥ 2 warps por subpartición", color=informe.TINTA,
                 fontsize=11, loc="left")
    e2.legend(frameon=False, fontsize=8, labelcolor=informe.TINTA, loc="lower right", title="acumuladores/warp",
              title_fontsize=8)

    for eje in (e1, e2):
        eje.grid(axis="x" if eje is e1 else "y", color=informe.REJILLA, linewidth=0.8, zorder=0)
        eje.set_axisbelow(True)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            eje.spines[lado].set_color(informe.TINTA_2)
        eje.tick_params(colors=informe.TINTA_2, length=0)
    fig.suptitle(f"Pico de mma.sync — {c.get('gpu', 'GPU')} (sm_{c.get('sm', '?')})", color=informe.TINTA,
                 fontsize=12, x=0.02, ha="left")
    fig.text(0.02, 0.01, f"Mediana de do_bench; un bloque por SM; sin accesos a memoria. CUDA {c.get('cuda', '?')}; "
             f"job {c.get('slurm_job')}.", fontsize=7, color=informe.TINTA_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(destino, f"grafica.{ext}"), dpi=200, facecolor="white")
    plt.close(fig)
    return True


def generar_informe(ejecucion):
    ejecucion = os.path.realpath(ejecucion)
    filas, meta = informe.leer(ejecucion)
    nombre = meta.get("experimento", "run_pico_mma")
    maquina = informe.maquina_de(meta, ejecucion)
    tabla = tabla_pico(filas)
    columnas = list(dict.fromkeys(k for f in tabla for k in f))
    md = informe.tabla_md(tabla, meta, columnas)
    with open(os.path.join(ejecucion, "tabla.md"), "w") as f:
        f.write(md)
    hay = grafica_pico(filas, meta, ejecucion)

    docs = os.path.join(informe.DOCS, "TFM", "resultados", maquina)
    os.makedirs(os.path.join(docs, "figuras"), exist_ok=True)
    img = ""
    if hay:
        for ext in ("png", "pdf"):
            shutil.copy(os.path.join(ejecucion, f"grafica.{ext}"), os.path.join(docs, "figuras", f"{nombre}.{ext}"))
        img = f"![{nombre}](figuras/{nombre}.png)\n\n"
    origen = os.path.relpath(ejecucion, informe.RAIZ)
    with open(os.path.join(docs, f"{nombre}.md"), "w") as f:
        f.write(f"# {nombre} ({maquina})\n\nGenerado automáticamente desde `{origen}/` "
                f"(barrido completo en `resultados.csv`).\n\n{img}{md}")
    print(f"Informe generado en {origen}/ y docs/TFM/resultados/{maquina}/{nombre}.md", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--informe":  # regenerar: run_pico_mma.py --informe <ejecucion>
        generar_informe(sys.argv[2])
    else:
        main()
