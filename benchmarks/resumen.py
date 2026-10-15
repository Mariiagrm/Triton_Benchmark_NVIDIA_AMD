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

# Variante de run_pico_mma con la que se compara cada precision (acumulacion en FP32, como los kernels).
VARIANTE_PICO = {"fp16": "f16·f32acc", "bf16": "bf16·f32acc", "fp8": "e4m3·f32acc"}


def motivo(arq, fila, col):
    """(clase, explicacion) de una celda sin dato. Clases: 'no aplica' (la variante no existe para
    ese algoritmo), 'sin kernel', 'no compila' y 'no medido'. Fuentes: procedencia_kernels.md y
    bitacora.md (jobs 20640, 20644, 20778-20781)."""
    alg = col.split()[0]
    if fila == "Triton (block-ptr)" and col != "Matmul fp16":
        return "no aplica", "variante solo de matmul fp16"
    if fila == "Triton + TMA" and alg != "Matmul":
        return "no aplica", "variante solo de matmul"
    if fila == "Triton + TLX":
        if alg == "Matmul":
            return "sin kernel", "no hay matmul en TLX"
        if alg == "FlashAttention":
            return "no compila", "los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor)"
        if "GB10" in arq:
            return "no compila", "triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes)"
    if fila == "Gluon" and alg == "FlashAttention":
        return "no compila", "el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st"
    if col == "Matmul FP8" and fila in ("Gluon", "Helion", "CUTLASS"):
        if fila == "CUTLASS":
            return "sin kernel", "el dense_gemm de CuTe solo admite fp16/bf16 (su FP8 en GeForce es MXFP8)"
        return "sin kernel", "no hay matmul FP8 en este DSL"
    return "no medido", "no se ha ejecutado en esta GPU"


def picos_mma():
    """Pico medido de mma.sync por arquitectura y precision (ultimo run_pico_mma de cada una):
    {arq: {'fp16': (TFLOP/s, 'maquina, job N'), ...}}."""
    picos, fechas = {}, {}
    for maquina, exp, ultimo in informe.ultimas_ejecuciones():
        if exp != "run_pico_mma":
            continue
        arq, ejecucion = arquitectura(ultimo)
        if ejecucion <= fechas.get(arq, ""):
            continue
        with open(os.path.join(ultimo, "meta.json")) as fh:
            pico = json.load(fh).get("resumen", {}).get("pico", {})
        origen = f"{maquina}, {_job({'ejecucion': ejecucion})}"
        picos[arq] = {p: (pico[v]["tflops"], origen) for p, v in VARIANTE_PICO.items() if v in pico}
        fechas[arq] = ejecucion
    return picos


def pico_cb(arq, col, picos):
    """(TFLOP/s, nombre) del pico con que se expresa el % de una celda CB: el de mma.sync medido en
    esa GPU si lo hay (run_pico_mma); si no, el de la ficha de NVIDIA; si no, (None, None)."""
    prec = fichas.precision_de(col)
    medido = picos.get(arq, {}).get(prec)
    if medido:
        return medido[0], "pico mma.sync medido"
    p = fichas.pico(arq, "TFLOP/s", prec)
    return (p[0], "pico ficha NVIDIA") if p else (None, None)


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

    def add(fila, col, valor, unidad, f, config, extra="", prefijo="", pct=None):
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
                    "tamano": tam, "orden": orden, "config": config, "datos": datos, "pct": pct})

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
            pct = f.get("pct_pico") if f.get("pct_pico") not in ("", None) else None
            extra = ""
            if v in FILA_DSL:
                cfg = _cfg_detalle(f.get("detalle"), v, f.get("N"))
                if not cfg and v == "triton":
                    # Ejecuciones anteriores a la columna 'detalle': la heuristica de num_warps de
                    # MBKernels/triton/rmsnorm_baseline.py (sin cambios desde entonces) la determina.
                    bloque = 1 << (int(f["N"]) - 1).bit_length()
                    cfg = _hilos(min(max(bloque // 256, 1), 16)) + " (derivado del código)"
                add(FILA_DSL[v], col, f["gbs"], "GB/s", f, cfg, extra, pct=pct)
            elif v == "torch":
                add("Referencia", col, f["gbs"], "GB/s", f, "PyTorch F.rms_norm", extra, pct=pct)
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


def slug(arq):
    """Nombre de fichero de una arquitectura ('NVIDIA GB10 (sm_121)' -> 'nvidia_gb10_sm_121')."""
    return re.sub(r"[^a-z0-9]+", "_", arq.lower()).strip("_")


def _fecha(c):
    """Marca temporal de la ejecucion ('20261002-091508_job20500' -> '20261002-091508')."""
    return c["ejecucion"].split("_job")[0]


def recoger():
    """Celdas finales: por (arquitectura, fila, columna), la de la ejecucion mas reciente
    (de cualquier entorno de esa arquitectura) y, dentro de ella, la del mayor tamano. Las CB
    llevan en 'pct' el % del pico (pico_cb) y en 'ref_pico' cual."""
    mejores, frias = {}, {}
    for maquina, exp, ultimo in informe.ultimas_ejecuciones():
        # Entornos *-energia (TFM_ENERGIA=1): miden energia con menos formas; si entraran, por ser
        # mas recientes sustituirian las celdas. Los *-frio (TFM_CACHE_FRIA=1) tampoco, salvo para
        # una celda que no tenga otro dato: su rendimiento se mide igual (do_bench tras compilar).
        # Los *-ncu (TFM_NCU=1) tampoco: el proceso corre bajo Nsight Compute (benchmarks/ncu.py).
        if maquina.endswith(("-energia", "-ncu")):
            continue
        destino = frias if maquina.endswith("-frio") else mejores
        arq, ejecucion = arquitectura(ultimo)
        for c in celdas_de(maquina, exp, ultimo):
            c["arquitectura"], c["ejecucion"] = arq, ejecucion
            clave = (arq, c["fila"], c["columna"])
            # Mas reciente; luego mayor tamano; a igualdad, el mejor valor (p. ej. Triton en
            # run_matmul y run_matmul_tma de la misma ejecucion).
            if clave not in destino or (_fecha(c), c["orden"], c["valor"]) > (
                    _fecha(destino[clave]), destino[clave]["orden"], destino[clave]["valor"]):
                destino[clave] = c
    for clave, c in frias.items():
        if clave not in mejores:
            c["extra"] = "solo medido con caché fría"
            mejores[clave] = c
    picos = picos_mma()
    for c in mejores.values():
        if TIPO[c["columna"]] == "CB":
            pico, nombre = pico_cb(c["arquitectura"], c["columna"], picos)
            if pico:
                c["pct"], c["ref_pico"] = round(100 * c["valor"] / pico, 1), nombre
    return list(mejores.values())


def _job(c):
    m = re.search(r"job(\d+)", c["ejecucion"])
    return f"job {m.group(1)}" if m else c["ejecucion"]


def _pct_texto(c):
    """'76 % del pico mma.sync medido' / '89 % del pico de memoria', o '' si no hay pico."""
    if c.get("pct") is None:
        return ""
    return f"{c['pct']:.0f} % del {c.get('ref_pico', 'pico de memoria')}"


def _texto_celda(c, sep):
    pct = f" · **{c['pct']:.0f} %** del {c.get('ref_pico', 'pico de memoria')}" if c.get("pct") is not None else ""
    extra = f" ({c['extra']})" if c["extra"] else ""
    return (f"**{c['valor']} {c['unidad']}**{pct}{extra}{sep}{c['tamano']} · {c['config']}"
            f"{sep}_{c['maquina']}, {_job(c)}_")


def _texto_vacia(arq, fila, col):
    clase, texto = motivo(arq, fila, col)
    return "n/a" if clase == "no aplica" else f"— *{clase}*: {texto}"


NOTA_PICOS = ("*% del pico: en compute-bound, sobre el pico de `mma.sync` medido en esa GPU con "
              "`run_pico_mma` (acumulación FP32; en GB10, 122.7 TFLOP/s fp16 y 243.3 FP8, en ráfaga a "
              "~2.5 GHz) o, si no se ha medido, sobre la ficha de NVIDIA (RTX 5090: 209.5 fp16 y 419 FP8 "
              "con acumulación FP32, al boost de 2 407 MHz). Por encima del 100 % de la ficha, la GPU va "
              "más rápida que el boost o no aplica el recorte de la acumulación FP32 (pendiente: "
              "`run_pico_mma` en pascal). En memory-bound, sobre el ancho de banda de la ficha. "
              "n/a = la variante no existe para ese algoritmo.*\n\n")


def tabla_md(arq, celdas):
    cols = [col for col, _ in COLUMNAS if any(c["columna"] == col for c in celdas)]
    cab = "| DSL | " + " | ".join(f"{col} ({TIPO[col]})" for col in cols) + " |"
    lineas = [cab, "|:---|" + "|".join([":---"] * len(cols)) + "|"]
    for fila in FILAS:
        textos = []
        for col in cols:
            cs = sorted((c for c in celdas if c["fila"] == fila and c["columna"] == col), key=lambda c: c["maquina"])
            textos.append("<br><br>".join(_texto_celda(c, "<br>") for c in cs) or _texto_vacia(arq, fila, col))
        lineas.append(f"| **{fila}** | " + " | ".join(textos) + " |")
    return f"## {arq}\n\n" + "\n".join(lineas) + "\n"


def tabla_texto(arq, celdas, ancho=30):
    """Tabla para el terminal/log: celdas de varias lineas (valor, % del pico, tamano,
    configuracion, origen) o el motivo de que falte."""
    cols = [col for col, _ in COLUMNAS if any(c["columna"] == col for c in celdas)]

    def lineas(c, fila, col):
        if c is None:
            clase, texto = motivo(arq, fila, col)
            partes = ["n/a"] if clase == "no aplica" else [f"— {clase}:", texto]
        else:
            partes = [f"{c['valor']} {c['unidad']}" + (f" ({c['extra']})" if c["extra"] else "")]
            partes += [_pct_texto(c)] if c.get("pct") is not None else []
            partes += [c["tamano"]] + c["config"].split(" · ") + [f"[{c['maquina']}, {_job(c)}]"]
        return [t for p in partes for t in (textwrap.wrap(p, ancho) or [""])]

    cab = [["DSL"]] + [[f"{col} ({TIPO[col]})"] for col in cols]
    cuerpo = []
    for fila in FILAS:
        de_fila = {c["columna"]: c for c in celdas if c["fila"] == fila}
        cuerpo.append([[fila]] + [lineas(de_fila.get(col), fila, col) for col in cols])
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


def _pico_tramo(ax, valor, etiqueta, x, estilo):
    """Tramo del pico sobre la columna x con su valor en el extremo DERECHO: a la izquierda lo
    taparian los % de las primeras barras."""
    fichas.dibujar(ax, valor, etiqueta, x - 0.45, x + 0.45, texto=False, estilo=estilo)
    ax.annotate(f"{valor:g}", (x + 0.45, valor), xytext=(0, 3), textcoords="offset points", fontsize=9,
                color=fichas.COLOR, va="bottom", ha="right", zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))


def graficas(celdas, docs):
    """Por arquitectura, dos graficas de barras agrupadas (seaborn.catplot): CB (TFLOP/s, viridis)
    y MB (GB/s, magma); barras por DSL (hue). Cada GPU va en su propia imagen: juntas, las escalas
    de GB10 y RTX 5090 no se leen bien. Si faltan seaborn/pandas se avisa y se omite (no debe
    romper el resumen). Salida: docs/.../figuras/resumen_{compute,memory}_bound_<arq>.{png,pdf}."""
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

    picos = picos_mma()
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
        for arq in sorted(set(df["arquitectura"])):
            sub = df[(df["unidad"] == unidad) & (df["arquitectura"] == arq)]
            if sub.empty:
                continue
            cols = [c for c in orden_col if c in set(sub["columna"])]
            # DSL con datos y tambien los que no tienen ninguno por un fallo (no por 'no aplica'),
            # para que el hueco y su motivo se vean.
            dsls = [f for f in orden_dsl if f in set(sub["fila"])
                    or any(motivo(arq, f, col)[0] != "no aplica" for col in cols)]
            g = sns.catplot(data=sub, kind="bar", x="columna", y="valor", hue="fila",
                            order=cols, hue_order=dsls, height=height, aspect=aspect, palette=palette)
            g.set_axis_labels("Kernel (operación)", ylab)
            ax = g.ax
            ax.tick_params(axis="x", rotation=rot)
            from matplotlib.lines import Line2D
            extra = {}  # entradas de la leyenda ademas de los DSL
            # Pico de la ficha de NVIDIA: un tramo por algoritmo (la precision cambia entre
            # columnas: fp16, FP8). GB10 no tiene pico oficial de fp16/FP8 (benchmarks/fichas.py),
            # pero si el de mma.sync medido (run_pico_mma), que es contra el que se da su %.
            for x, col in enumerate(cols):
                p = fichas.pico(arq, unidad, fichas.precision_de(col))
                if p:
                    _pico_tramo(ax, p[0], "Pico ficha NVIDIA", x, "--")
                    extra["Pico ficha NVIDIA"] = Line2D([], [], color=fichas.COLOR, linestyle="--", linewidth=1.6)
                elif unidad == "TFLOP/s":
                    medido, _ = pico_cb(arq, col, picos)
                    if medido:
                        _pico_tramo(ax, medido, "Pico mma.sync medido", x, ":")
                        extra["Pico mma.sync medido"] = Line2D([], [], color=fichas.COLOR, linestyle=":",
                                                               linewidth=1.6)
            # % del pico encima de cada barra: un contenedor por DSL, en el orden de hue_order (seaborn
            # los etiqueta '_containerN'); la x de cada barra da su columna.
            for fila, cont in zip(dsls, ax.containers):
                pcts = []
                for barra in cont:
                    col = cols[int(round(barra.get_x() + barra.get_width() / 2))]
                    v = sub.loc[(sub["fila"] == fila) & (sub["columna"] == col), "pct"]
                    pcts.append(f"{v.iloc[0]:.0f} %" if len(v) and pd.notna(v.iloc[0]) else "")
                for t in ax.bar_label(cont, labels=pcts, rotation=90, padding=3, fontsize=10, color="#333333"):
                    t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))  # legible sobre el pico
                    t.set_zorder(6)  # la linea del pico va en zorder 4
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 1.12)
            # Barras que faltan: '✕ (n) clase' en el hueco y, bajo la grafica, el motivo de cada n (no
            # se rotulan las 'no aplica', que no son un fallo).
            ancho = 0.8 / len(dsls)
            notas = {}  # (clase, motivo) -> n
            for i, fila in enumerate(dsls):
                for x, col in enumerate(cols):
                    if ((sub["fila"] == fila) & (sub["columna"] == col)).any():
                        continue
                    clase, texto = motivo(arq, fila, col)
                    if clase != "no aplica":
                        n = notas.setdefault((clase, f"{fila}: {texto}"), len(notas) + 1)
                        ax.text(x - 0.4 + ancho * (i + 0.5), hi * 0.01, f"✕ ({n}) {clase}", rotation=90,
                                ha="center", va="bottom", fontsize=9, color="#777777")
            ley = g._legend
            asas = ley.legend_handles + list(extra.values())
            textos = [t.get_text() for t in ley.texts] + list(extra)
            ley.remove()
            g._legend = None  # add_legend vuelve a reservar el ancho que necesita la leyenda
            g.add_legend(legend_data=dict(zip(textos, asas)), label_order=textos, title="DSL")
            # Motivos de los huecos: la figura crece hacia abajo para que no pisen el eje x.
            alto = g.fig.get_figheight()
            extra_alto = 0.32 * len(notas)
            g.fig.set_figheight(alto + extra_alto)
            if notas:
                g.fig.text(0.01, 0.01, "\n".join(f"({n}) {clase} — {t}" for (clase, t), n in notas.items()),
                           fontsize=11, color="#555555", ha="left", va="bottom")
            g.fig.subplots_adjust(top=1 - 0.15 * alto / (alto + extra_alto),
                                  bottom=(0.28 * alto + extra_alto + 0.2) / (alto + extra_alto))
            g.fig.suptitle(f"{arq} — {titulo}", fontsize=18, fontweight="bold")
            fichero = f"{nombre}_{slug(arq)}"
            for ext in ("png", "pdf"):
                g.savefig(os.path.join(figs, f"{fichero}.{ext}"), dpi=150)
            plt.close(g.fig)
            generadas.append(f"{fichero}.png")
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
    campos = ["arquitectura", "fila", "columna", "tipo", "valor", "unidad", "pct", "ref_pico", "extra", "tamano",
              "config", "maquina", "experimento", "ejecucion"]
    celdas.sort(key=lambda c: (c["arquitectura"], FILAS.index(c["fila"]), [k for k, _ in COLUMNAS].index(c["columna"]), c["maquina"]))
    with open(destino_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows({**c, "tipo": TIPO[c["columna"]]} for c in celdas)

    docs = os.path.join(informe.DOCS, "TFM", "resultados")
    os.makedirs(docs, exist_ok=True)
    nota = ("*Generado por `benchmarks/resumen.py` a partir de la última ejecución de cada benchmark en cada "
            "entorno. CB = compute-bound (TFLOP/s); MB = memory-bound (GB/s y % del pico de memoria). Cada celda: "
            "rendimiento en el mayor tamaño medido · % del pico · tamaño · configuración aplicada · entorno y job. "
            "Una celda sin dato dice por qué falta.*\n\n" + NOTA_PICOS)
    completo = ["# Resumen de rendimiento por arquitectura\n\n", nota]
    for arq in sorted({c["arquitectura"] for c in celdas}):
        de_arq = [c for c in celdas if c["arquitectura"] == arq]
        md = tabla_md(arq, de_arq)
        tipos = {TIPO[c["columna"]] for c in de_arq}
        figuras = "".join(f"![{t}](figuras/resumen_{n}_{slug(arq)}.png)\n\n"
                          for tipo, t, n in (("CB", "Compute-bound", "compute_bound"),
                                             ("MB", "Memory-bound", "memory_bound")) if tipo in tipos)
        with open(os.path.join(docs, f"resumen_{slug(arq)}.md"), "w") as fh:
            fh.write(f"# Resumen: {arq}\n\n{nota}{md.split(chr(10), 2)[2]}\n{figuras}")
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
    # Procedencia de cada kernel junto a estos resultados (kernels.md) y metricas de ncu (ncu.md).
    try:
        import kernels
        kernels.generar()
        import ncu
        ncu.generar()
        import comparativa  # documento unico: metodologia + todo lo anterior
        comparativa.generar()
    except Exception as e:
        print(f"AVISO: no se pudo generar kernels.md ({type(e).__name__}: {e}). "
              f"Regeneralo con: python3.11 benchmarks/kernels.py", flush=True)
    if mostrar:
        arqs = sorted({c["arquitectura"] for c in celdas}) if mostrar == "todas" else [arquitectura(mostrar)[0]]
        for arq in arqs:
            print("\n" + tabla_texto(arq, [c for c in celdas if c["arquitectura"] == arq]), flush=True)


if __name__ == "__main__":
    generar(mostrar="todas")
