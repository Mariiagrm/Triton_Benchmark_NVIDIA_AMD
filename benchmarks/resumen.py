"""Resumen de rendimiento: UNA TABLA POR ARQUITECTURA, filas = DSL, columnas = algoritmo.

Cada celda guarda el ULTIMO resultado de esa arquitectura: el de la ejecucion mas reciente
que tenga ese dato (aunque haya otros entornos de la misma GPU, p. ej. hennessy-580 y
hennessy en GB10), en el MAYOR tamano medido, con la configuracion aplicada (tamano del
problema, bloque/tile, warps e hilos, etapas, ...) y el entorno y el job de los que sale.

Metrica por tipo de algoritmo:
    CB (compute-bound, matmul)  -> TFLOP/s
    MB (memory-bound, RMSNorm)  -> GB/s y % del pico de memoria (sus TFLOP/s serian ~0 y no
                                   dirian nada: lo que limita es el ancho de banda)

Se genera a partir de la ultima ejecucion de cada benchmark en cada entorno
(results/<maquina>/<benchmark>/ultimo). Salida:
    results/resumen.csv                          formato largo (una fila por celda y entorno)
    docs/TFM/resultados/resumen_<arq>.md   una tabla por arquitectura
    docs/TFM/resultados/resumen.md               todas las arquitecturas juntas

Lo llama comun.guardar() al final de cada ejecucion, que ademas imprime en el log la tabla
de la arquitectura recien medida. A mano (en el login, sin GPU; imprime todas):
    python3.11 benchmarks/resumen.py
"""
import csv
import json
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fichas  # noqa: E402
import informe  # noqa: E402

RAIZ = informe.RAIZ

# Filas (DSL) y columnas (algoritmo, tipo) en orden de presentacion.
FILAS = ["Triton", "Triton (block-ptr)", "Triton + TMA", "Triton + TLX", "Gluon", "Helion",
         "CUTLASS", "Referencia"]
COLUMNAS = [("Matmul fp16", "CB"), ("Matmul FP8", "CB"), ("FlashAttention fp16", "CB"),
            ("FlashAttention causal fp16", "CB"), ("RMSNorm fp16", "MB"), ("RMSNorm bf16", "MB")]
TIPO = dict(COLUMNAS)

# Fila de cada variante de los benchmarks de RMSNorm y del banco de matmul.
FILA_DSL = {"triton": "Triton", "tlx": "Triton + TLX", "gluon": "Gluon", "helion": "Helion",
            "cutlass": "CUTLASS"}


def _num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return v


def _hilos(warps):
    return f"{warps} warps ({int(warps) * 32} hilos)"


def _cfg_triton(f):
    return (f"BM={f['BLOCK_SIZE_M']} BN={f['BLOCK_SIZE_N']} BK={f['BLOCK_SIZE_K']} "
            f"GROUP_M={f['GROUP_SIZE_M']} · {_hilos(f['num_warps'])} · {f['num_stages']} etapas")


def _cfg_detalle(detalle, dsl, N=None):
    """Configuracion a partir de la columna 'detalle' (pares clave=valor) de los bancos."""
    d = dict(re.findall(r"(\w+)=(\S+)", detalle or ""))
    partes = []
    if dsl == "cutlass" and detalle and detalle.startswith("rmsnorm_twoPassAlgo"):
        # Bloque de hilos que elige cutlass::rmsnorm (device_rmsnorm.h): uno por fila.
        e8 = detalle.endswith("_e8")
        hilos = min(1024, (N // 8 + 31) // 32 * 32) if e8 else min(1024, ((N + 31) // 32 + 31) // 32 * 32)
        return f"cutlass::rmsnorm {'_e8 (float4)' if e8 else '_e1 (escalar)'} · {hilos} hilos/bloque · 1 bloque/fila"
    if (detalle or "").startswith("FlashAttentionForwardAmpere"):
        return (f"CuTe DSL FlashAttentionForwardAmpere · BM={d.get('BM')} BN={d.get('BN')} · "
                f"{d.get('hilos')} hilos ({int(d.get('hilos', 0)) // 32} warps)")
    if "Sm120GemmKernel" in (detalle or ""):
        return f"CuTe DSL Sm120GemmKernel · tile {d.get('tile', '?')}"
    for k in ("BM", "BN", "BK"):
        if k in d:
            partes.append(f"{k}={d[k]}")
    if "stages" in d:
        partes.append(f"{d['stages']} etapas")
    if "warps" in d and "x" in d["warps"]:
        wm, wn = map(int, d["warps"].split("x"))
        partes.append(f"{wm * wn} warps {wm}x{wn} ({wm * wn * 32} hilos)")
    elif "num_warps" in d:
        partes.append(_hilos(d["num_warps"]))
    if "vec" in d:
        partes.append(f"{d['vec']} elem/hilo (16 B)")
    if "grid" in d:
        partes.append(f"grid={d['grid']} (persistente)")
    if "carga" in d:
        partes.append(f"carga={d['carga']}")
    if "block_sizes" in d:
        partes.append(f"bloques={d['block_sizes']}")
    if "num_stages" in d:
        partes.append(f"{d['num_stages']} etapas")
    if "autotune" in d:
        partes.append(f"autotune={d['autotune']}")
    return " · ".join(partes) or (detalle or "")


def _kernel_corto(nombre):
    if "flash_fwd_kernel" in (nombre or ""):  # el FlashAttention-2 que incluye PyTorch (SDPA)
        return "pytorch_flash::flash_fwd_kernel"
    m = re.search(r"(cutlass_80_tensorop_\w+?|nvjet_\w+)(?:\(|$|::)", nombre or "")
    return m.group(1) if m else (nombre or "")[:60]


def celdas_de(maquina, exp, ultimo):
    """(fila, columna, valor, unidad, extra, tamano, config) de la ultima ejecucion de exp."""
    with open(os.path.join(ultimo, "resultados.csv")) as fh:
        filas = [{k: _num(v) for k, v in f.items()} for f in csv.DictReader(fh)]
    out = []

    def add(fila, col, valor, unidad, f, config, extra="", prefijo=""):
        # Columnas fuera de COLUMNAS (p. ej. matmul en bf16 pedido a mano) no entran en el resumen.
        if fila is None or col not in TIPO or valor in ("", None):
            return
        if "N_CTX" in f:  # atencion: (B, H, N, D); el trabajo crece con B*H*N*N*D
            tam = f"B={f['B']} H={f['H']} N={f['N_CTX']} D={f['HEAD_DIM']}"
            orden = _prod([f["B"], f["H"], f["N_CTX"], f["N_CTX"], f["HEAD_DIM"]])
        else:
            dims = [f[k] for k in ("M", "N", "K") if f.get(k) not in ("", None)]
            tam, orden = "×".join(map(str, dims)), _prod(dims)
        # datos: la fila de resultados de la que sale la celda (sin el prefijo de run_matmul, que
        # guarda Triton y cuBLAS en la misma fila); de ahi lee benchmarks/energia.py.
        datos = {k[len(prefijo):]: v for k, v in f.items() if k.startswith(prefijo)}
        out.append({"fila": fila, "columna": col, "valor": valor, "unidad": unidad, "extra": extra,
                    "tamano": tam, "orden": orden, "config": config, "datos": datos})

    for f in filas:
        dt = f.get("dtype", "fp16")
        if exp == "run_matmul":
            col = f"Matmul {dt}"
            add("Triton", col, f["triton_tflops"], "TFLOP/s", f, _cfg_triton(f), prefijo="triton_")
            add("Referencia", col, f["cublas_tflops"], "TFLOP/s", f, "cuBLAS " + _kernel_corto(f.get("cublas_kernel")),
                prefijo="cublas_")
        elif exp == "run_matmul_tma":
            fila = {"blockptr": "Triton (block-ptr)", "tma": "Triton + TMA"}.get(f["variante"])
            add(fila, f"Matmul {dt}", f["tflops"], "TFLOP/s", f, _cfg_triton(f) + f" · carga={f['carga']}")
        elif exp == "run_matmul_fp8":
            v = f["variante"]
            if v == "fp8":
                add("Triton", "Matmul FP8", f["tflops"], "TFLOP/s", f, _cfg_triton(f) + f" · error {f['error_rel']:.2%}")
            elif v == "fp8_tma":
                add("Triton + TMA", "Matmul FP8", f["tflops"], "TFLOP/s", f,
                    _cfg_triton(f) + f" · carga={f['carga']} · error {f['error_rel']:.2%}")
            elif v == "fp8_cublas":
                add("Referencia", "Matmul FP8", f["tflops"], "TFLOP/s", f, f"cuBLASLt (torch._scaled_mm) · error {f['error_rel']:.2%}")
        elif exp.startswith("run_matmul_"):  # bancos de matmul: gluon, helion, cutlass
            v = f["variante"]
            if v in FILA_DSL:
                add(FILA_DSL[v], f"Matmul {dt}", f["tflops"], "TFLOP/s", f, _cfg_detalle(f.get("detalle"), v))
        elif exp.startswith("run_attention_"):
            v = f["variante"]
            col = "FlashAttention causal fp16" if str(f["causal"]) == "True" else "FlashAttention fp16"
            if v == "sdpa":
                add("Referencia", col, f["tflops"], "TFLOP/s", f, "PyTorch SDPA " + _kernel_corto(f.get("detalle")))
            elif v in FILA_DSL:
                add(FILA_DSL[v], col, f["tflops"], "TFLOP/s", f, _cfg_detalle(f.get("detalle"), v))
        elif exp.startswith("run_rmsnorm_"):
            v = f["variante"]
            col = f"RMSNorm {dt}"
            extra = f"{f['pct_pico']} % del pico" if f.get("pct_pico") not in ("", None) else ""
            if v in FILA_DSL:
                cfg = _cfg_detalle(f.get("detalle"), v, f.get("N"))
                if not cfg and v == "triton":
                    # Ejecuciones anteriores a la columna 'detalle': la heuristica de num_warps de
                    # MBKernels/triton/rmsnorm_baseline.py (sin cambios desde entonces) la determina.
                    bloque = 1 << (int(f["N"]) - 1).bit_length()
                    cfg = _hilos(min(max(bloque // 256, 1), 16)) + " (derivado del código)"
                add(FILA_DSL[v], col, f["gbs"], "GB/s", f, cfg, extra)
            elif v == "torch":
                add("Referencia", col, f["gbs"], "GB/s", f, "PyTorch F.rms_norm", extra)
    for c in out:
        c["maquina"], c["experimento"] = maquina, exp
    return out


def _prod(dims):
    p = 1
    for d in dims:
        p *= int(d)
    return p


def arquitectura(ultimo):
    with open(os.path.join(ultimo, "meta.json")) as fh:
        c = json.load(fh).get("contexto", {})
    return f"{c.get('gpu', '?')} (sm_{c.get('sm', '?')})", os.path.basename(os.path.realpath(ultimo))


def _fecha(c):
    """Marca temporal de la ejecucion ('20261002-091508_job20500' -> '20261002-091508')."""
    return c["ejecucion"].split("_job")[0]


def recoger():
    """Celdas finales: por (arquitectura, fila, columna), la de la ejecucion mas reciente
    (de cualquier entorno de esa arquitectura) y, dentro de ella, la del mayor tamano."""
    mejores = {}
    for maquina, exp, ultimo in informe.ultimas_ejecuciones():
        # Entornos *-frio (TFM_CACHE_FRIA=1) y *-energia (TFM_ENERGIA=1): miden compilacion o
        # energia con menos formas; si entraran, por ser mas recientes sustituirian las celdas.
        if maquina.endswith(("-frio", "-energia")):
            continue
        arq, ejecucion = arquitectura(ultimo)
        for c in celdas_de(maquina, exp, ultimo):
            c["arquitectura"], c["ejecucion"] = arq, ejecucion
            clave = (arq, c["fila"], c["columna"])
            # Mas reciente; luego mayor tamano; a igualdad, el mejor valor (p. ej. Triton en
            # run_matmul y run_matmul_tma de la misma ejecucion).
            if clave not in mejores or (_fecha(c), c["orden"], c["valor"]) > (
                    _fecha(mejores[clave]), mejores[clave]["orden"], mejores[clave]["valor"]):
                mejores[clave] = c
    return list(mejores.values())


def _job(c):
    m = re.search(r"job(\d+)", c["ejecucion"])
    return f"job {m.group(1)}" if m else c["ejecucion"]


def _texto_celda(c, sep):
    extra = f" ({c['extra']})" if c["extra"] else ""
    return (f"**{c['valor']} {c['unidad']}**{extra}{sep}{c['tamano']} · {c['config']}"
            f"{sep}_{c['maquina']}, {_job(c)}_")


def tabla_md(arq, celdas):
    cols = [col for col, _ in COLUMNAS if any(c["columna"] == col for c in celdas)]
    filas = [f for f in FILAS if any(c["fila"] == f for c in celdas)]
    cab = "| DSL | " + " | ".join(f"{col} ({TIPO[col]})" for col in cols) + " |"
    lineas = [cab, "|:---|" + "|".join([":---"] * len(cols)) + "|"]
    for fila in filas:
        textos = []
        for col in cols:
            cs = sorted((c for c in celdas if c["fila"] == fila and c["columna"] == col), key=lambda c: c["maquina"])
            textos.append("<br><br>".join(_texto_celda(c, "<br>") for c in cs) or "—")
        lineas.append(f"| **{fila}** | " + " | ".join(textos) + " |")
    return f"## {arq}\n\n" + "\n".join(lineas) + "\n"


def tabla_texto(arq, celdas, ancho=30):
    """Tabla para el terminal/log: celdas de varias lineas (valor, tamano, configuracion, origen)."""
    cols = [col for col, _ in COLUMNAS if any(c["columna"] == col for c in celdas)]
    filas = [f for f in FILAS if any(c["fila"] == f for c in celdas)]

    def lineas(c):
        if c is None:
            return ["—"]
        partes = [f"{c['valor']} {c['unidad']}" + (f" ({c['extra']})" if c["extra"] else ""), c["tamano"]]
        partes += c["config"].split(" · ") + [f"[{c['maquina']}, {_job(c)}]"]
        return [t for p in partes for t in (textwrap.wrap(p, ancho) or [""])]

    cab = [["DSL"]] + [[f"{col} ({TIPO[col]})"] for col in cols]
    cuerpo = []
    for fila in filas:
        de_fila = {c["columna"]: c for c in celdas if c["fila"] == fila}
        cuerpo.append([[fila]] + [lineas(de_fila.get(col)) for col in cols])
    anchos = [max(len(l) for r in [cab] + cuerpo for l in r[i]) for i in range(len(cab))]
    sep = "+" + "+".join("-" * (a + 2) for a in anchos) + "+"

    def pintar(r):
        alto = max(len(x) for x in r)
        return [("| " + " | ".join((r[i][k] if k < len(r[i]) else "").ljust(anchos[i]) for i in range(len(r))) + " |")
                for k in range(alto)]

    out = [f"== {arq} ==", sep, *pintar(cab), sep.replace("-", "=")]
    for r in cuerpo:
        out += pintar(r) + [sep]
    return "\n".join(out)


def graficas(celdas, docs):
    """Dos graficas de barras agrupadas, una faceta por arquitectura (seaborn.catplot):
    CB (TFLOP/s, viridis) y MB (GB/s, magma); barras por DSL (hue). Si faltan seaborn/pandas
    se avisa y se omite (no debe romper el resumen). Salida: docs/.../figuras/resumen_*.{png,pdf}."""
    try:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"AVISO: no genero las graficas del resumen (falta '{e.name}'; instala con "
              f"'pip install seaborn').", flush=True)
        return

    df = pd.DataFrame(celdas)
    if df.empty:
        return
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["valor"])

    figs = os.path.join(docs, "figuras")
    os.makedirs(figs, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    orden_dsl = list(FILAS)                     # orden de la leyenda (DSL)
    orden_col = [c for c, _ in COLUMNAS]        # orden del eje x (algoritmo)

    # (unidad, paleta, titulo, etiqueta_y, nombre, rotacion_x, height, aspect)
    specs = [
        ("TFLOP/s", "viridis", "Compute Bound (TFLOP/s)", "Rendimiento (TFLOP/s)",
         "resumen_compute_bound", 45, 7, 1.3),
        ("GB/s", "magma", "Memory Bound (GB/s)", "Ancho de banda (GB/s)",
         "resumen_memory_bound", 20, 6, 1.2),
    ]
    generadas = []
    for unidad, palette, titulo, ylab, nombre, rot, height, aspect in specs:
        sub = df[df["unidad"] == unidad]
        if sub.empty:
            continue
        cols = [c for c in orden_col if c in set(sub["columna"])]
        dsls = [f for f in orden_dsl if f in set(sub["fila"])]
        g = sns.catplot(data=sub, kind="bar", x="columna", y="valor", hue="fila",
                        col="arquitectura", order=cols, hue_order=dsls,
                        height=height, aspect=aspect, palette=palette)
        g.set_axis_labels("Kernel (operación)", ylab)
        g.set_titles("{col_name}")
        for ax in g.axes.flat:
            ax.tick_params(axis="x", rotation=rot)
        # Pico de la ficha de NVIDIA de cada GPU: un tramo por algoritmo (la precision cambia
        # entre columnas: fp16, FP8). GB10 no tiene pico oficial de fp16/FP8 (benchmarks/fichas.py).
        con_pico = False
        for arq, ax in g.axes_dict.items():
            for x, col in enumerate(cols):
                p = fichas.pico(arq, unidad, fichas.precision_de(col))
                if p:
                    fichas.dibujar(ax, p[0], "Pico ficha NVIDIA", x - 0.45, x + 0.45)
                    con_pico = True
            if unidad == "TFLOP/s" and not any(fichas.pico(arq, unidad, fichas.precision_de(c)) for c in cols):
                ax.text(0.98, 0.97, "sin pico oficial de NVIDIA\nen fp16/FP8", transform=ax.transAxes,
                        ha="right", va="top", fontsize=11, color=fichas.COLOR)
        if con_pico:
            from matplotlib.lines import Line2D
            ley = g._legend
            asas = ley.legend_handles + [Line2D([], [], color=fichas.COLOR, linestyle="--", linewidth=1.6)]
            textos = [t.get_text() for t in ley.texts] + ["Pico ficha NVIDIA"]
            ley.remove()
            g.fig.legend(asas, textos, loc="center right", frameon=False, title="fila")
        g.fig.subplots_adjust(top=0.85, bottom=0.28)
        g.fig.suptitle(f"Comparativa de rendimiento — {titulo}", fontsize=18, fontweight="bold")
        for ext in ("png", "pdf"):
            g.savefig(os.path.join(figs, f"{nombre}.{ext}"), dpi=150)
        plt.close(g.fig)
        generadas.append(f"{nombre}.png")
    if generadas:
        print("Graficas del resumen: " + ", ".join("docs/TFM/resultados/figuras/" + n for n in generadas),
              flush=True)


def generar(mostrar=None):
    """Regenera los ficheros del resumen e imprime la tabla de la arquitectura de la ejecucion
    'mostrar' (ruta de resultados), o todas si mostrar='todas'."""
    celdas = recoger()
    if not celdas:
        return
    destino_csv = os.path.join(RAIZ, "results", "resumen.csv")
    campos = ["arquitectura", "fila", "columna", "tipo", "valor", "unidad", "extra", "tamano", "config",
              "maquina", "experimento", "ejecucion"]
    celdas.sort(key=lambda c: (c["arquitectura"], FILAS.index(c["fila"]), [k for k, _ in COLUMNAS].index(c["columna"]), c["maquina"]))
    with open(destino_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows({**c, "tipo": TIPO[c["columna"]]} for c in celdas)

    docs = os.path.join(informe.DOCS, "TFM", "resultados")
    os.makedirs(docs, exist_ok=True)
    nota = ("*Generado por `benchmarks/resumen.py` a partir de la última ejecución de cada benchmark en cada "
            "entorno. CB = compute-bound (TFLOP/s); MB = memory-bound (GB/s y % del pico de memoria). Cada celda: "
            "rendimiento en el mayor tamaño medido · tamaño · configuración aplicada · entorno y job.*\n\n")
    completo = ["# Resumen de rendimiento por arquitectura\n\n", nota]
    for arq in sorted({c["arquitectura"] for c in celdas}):
        de_arq = [c for c in celdas if c["arquitectura"] == arq]
        slug = re.sub(r"[^a-z0-9]+", "_", arq.lower()).strip("_")
        md = tabla_md(arq, de_arq)
        with open(os.path.join(docs, f"resumen_{slug}.md"), "w") as fh:
            fh.write(f"# Resumen: {arq}\n\n{nota}{md.split(chr(10), 2)[2]}")
        completo.append(md + "\n")
    with open(os.path.join(docs, "resumen.md"), "w") as fh:
        fh.write("".join(completo))
    # Graficas (no deben romper el resumen si faltan seaborn/pandas).
    try:
        graficas(celdas, docs)
    except Exception as e:
        print(f"AVISO: fallo al generar las graficas del resumen ({type(e).__name__}: {e}).", flush=True)
    print(f"Resumen por arquitectura en docs/TFM/resultados/resumen*.md y {os.path.relpath(destino_csv, RAIZ)}",
          flush=True)
    if mostrar:
        arqs = sorted({c["arquitectura"] for c in celdas}) if mostrar == "todas" else [arquitectura(mostrar)[0]]
        for arq in arqs:
            print("\n" + tabla_texto(arq, [c for c in celdas if c["arquitectura"] == arq]), flush=True)


if __name__ == "__main__":
    generar(mostrar="todas")
