"""Informe GENERICO de un experimento: tabla (Markdown + LaTeX) y grafica (PNG + PDF).

Lo llama comun.guardar() automaticamente al final de CADA experimento que no
proporcione su propio informe, asi que cualquier experimento nuevo obtiene tabla y
grafica sin escribir codigo extra. Funciona a partir del resultados.csv + meta.json,
no necesita GPU ni torch (solo matplotlib).

Regenerar a mano cualquier ejecucion:
    python3.11 benchmarks/informe.py results/<maquina>/<experimento>/<ejecucion>
    python3.11 benchmarks/informe.py results/<maquina>/<experimento>/ultimo

Salida en la ejecucion: tabla.md, tabla.tex, grafica.png, grafica.pdf
Copia para la memoria: docs/TFM/resultados/<maquina>/<experimento>.{md,tex} (+ figuras/)

Los resultados se separan por MAQUINA (entorno de ejecucion: por defecto el nodo; se
puede fijar con TFM_MAQUINA para distinguir entornos en el mismo nodo).

La grafica usa la metrica de rendimiento disponible: "tflops" (compute-bound) o, si no
la hay, "gbs" (memory-bound, p. ej. RMSNorm). Sin ninguna de las dos, solo tablas.
"""
import csv
import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fichas  # noqa: E402

RAIZ = os.environ.get("TFM_RAIZ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Documentos para la memoria (tablas, figuras, secciones): FUERA del repositorio, en ../tfm/docs
# (p. ej. ~/hennessy/tfm/docs para ~/hennessy/tfm_entorno). En el contenedor, ejecutar.sh lo monta
# en /workspace/docs y lo pasa en TFM_DOCS.
DOCS = os.environ.get("TFM_DOCS") or os.path.join(os.path.dirname(RAIZ), "tfm", "docs")

# Paleta categorica de referencia (validada en modo claro) y tintas de texto.
PALETA = ["#2a78d6", "#eb6834", "#2ca089", "#a65cd6", "#d6b12a", "#d6425c"]
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"

# Columnas de rendimiento (por orden de preferencia, con su etiqueta del eje) y
# candidatas a etiqueta/serie.
METRICAS = {"tflops": "TFLOP/s (mayor es mejor)", "gbs": "GB/s (mayor es mejor)"}
COLS_SERIE = ["variante", "dtype", "kernel"]


def leer(ejecucion):
    with open(os.path.join(ejecucion, "resultados.csv")) as f:
        filas = list(csv.DictReader(f))
    with open(os.path.join(ejecucion, "meta.json")) as f:
        meta = json.load(f)
    # Ejecuciones antiguas sin contexto.maquina: se deduce de la ruta (results/<maquina>/...).
    meta.setdefault("contexto", {}).setdefault("maquina", maquina_de(meta, ejecucion))
    for fila in filas:
        for k, v in fila.items():
            try:
                fila[k] = int(v)
            except (ValueError, TypeError):
                try:
                    fila[k] = float(v)
                except (ValueError, TypeError):
                    pass
    return filas, meta


def etiqueta_tamano(f):
    if "N_CTX" in f:  # atencion (B, H, N, D): B y H son fijos en cada ejecucion
        return f"N={f['N_CTX']}\nD={f['HEAD_DIM']}" + ("\ncausal" if str(f.get("causal")) == "True" else "")
    if all(k in f for k in ("M", "N", "K")):
        return f"{f['M']}×{f['N']}×{f['K']}"
    if "M" in f and "N" in f:
        return f"{f['M']}×{f['N']}"
    if "M" in f:
        return str(f["M"])
    return ""


def pie(meta):
    c = meta.get("contexto", {})
    return (f"{c.get('gpu', '?')} (sm_{c.get('sm', '?')}), torch {c.get('torch', '?')}, "
            f"triton {c.get('triton', '?')}, CUDA {c.get('cuda', '?')}; job {c.get('slurm_job')}, "
            f"{c.get('fecha', '')}. Mediana de triton.testing.do_bench.")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def tabla_md(filas, meta, columnas):
    cab = "| " + " | ".join(columnas) + " |"
    sep = "|" + "|".join([":---"] * len(columnas)) + "|"
    cuerpo = ["| " + " | ".join(_fmt(f.get(c, "")) for c in columnas) + " |" for f in filas]
    return "\n".join([cab, sep] + cuerpo) + f"\n\n*{pie(meta)}*\n"


def etiqueta_tex(meta, nombre):
    """\\label unica por entorno: la misma tabla de dos GPUs (p. ej. hennessy y pascal) puede
    incluirse en la misma memoria sin que LaTeX las confunda. tab:<maquina>-<nombre>."""
    maquina = meta.get("contexto", {}).get("maquina")
    return "tab:" + "-".join(x.replace("_", "-") for x in (maquina, nombre) if x)


def tabla_tex(filas, meta, columnas, nombre):
    esc = lambda s: str(s).replace("×", r"$\times$").replace("%", r"\%").replace("_", r"\_")  # noqa: E731
    align = "".join("r" if all(isinstance(f.get(c), (int, float)) for f in filas) else "l" for c in columnas)
    cuerpo = "\n".join("    " + " & ".join(esc(_fmt(f.get(c, ""))) for c in columnas) + r" \\" for f in filas)
    return (r"% Requiere \usepackage{booktabs}" "\n"
            r"\begin{table}[htbp]" "\n" r"  \centering\small" "\n"
            r"  \begin{tabular}{" + align + "}\n" r"    \toprule" "\n"
            "    " + " & ".join(esc(c) for c in columnas) + r" \\" "\n" r"    \midrule" "\n"
            + cuerpo + "\n" r"    \bottomrule" "\n" r"  \end{tabular}" "\n"
            r"  \caption{" + esc(nombre) + ": " + esc(pie(meta)) + "}\n"
            r"  \label{" + etiqueta_tex(meta, nombre) + "}\n" r"\end{table}" "\n")


def grafica(filas, meta, destino, nombre):
    """Barras de TFLOP/s o GB/s. Facetas por dtype; series por 'variante' (o la 1a categoria util)."""
    metrica = next((m for m in METRICAS if filas and m in filas[0]), None)
    if metrica is None:
        return False  # sin columna de rendimiento: no hay grafica

    facet_col = "dtype" if "dtype" in filas[0] else None
    serie_col = next((c for c in COLS_SERIE if c in filas[0] and c != facet_col), None)

    facetas = list(dict.fromkeys(f[facet_col] for f in filas)) if facet_col else [None]
    series = list(dict.fromkeys(f[serie_col] for f in filas)) if serie_col else [nombre]
    color = {s: PALETA[i % len(PALETA)] for i, s in enumerate(series)}

    # Ancho por faceta segun el numero de series y de formas del eje x (con muchas formas, p. ej.
    # las 12 de atencion, las etiquetas se pisaban).
    n_tam = max(len({etiqueta_tamano(f) for f in filas if facet_col is None or f[facet_col] == fac}) for fac in facetas)
    ancho_faceta = max(5.0, 2.2 * len(series) + 2.5, n_tam * (0.35 * len(series) + 0.45) + 1.5)
    fig, ejes = plt.subplots(1, len(facetas), figsize=(ancho_faceta * len(facetas), 4.4),
                             sharey=True, squeeze=False)
    tope = 0
    for eje, fac in zip(ejes[0], facetas):
        ff = [f for f in filas if (facet_col is None or f[facet_col] == fac)]
        tamanos = list(dict.fromkeys(etiqueta_tamano(f) for f in ff))
        xs = range(len(tamanos))
        ancho = min(0.8 / max(len(series), 1), 0.38)
        for i, s in enumerate(series):
            med, pos = [], []
            for x, t in zip(xs, tamanos):
                fila = next((f for f in ff if etiqueta_tamano(f) == t
                             and (serie_col is None or f[serie_col] == s)), None)
                if fila is None or not isinstance(fila.get(metrica), (int, float)):
                    continue  # sin medida (p. ej. una forma que fallo, con columna 'error')
                v = fila[metrica]
                med.append(v)
                pos.append(x + (i - (len(series) - 1) / 2) * (ancho + 0.02))
                tope = max(tope, v)
            eje.bar(pos, med, ancho, color=color[s], label=str(s), zorder=2)
            for p, m in zip(pos, med):
                eje.text(p, m, f"{m:.0f}", ha="center", va="bottom", fontsize=8, color=TINTA)
        eje.set_xticks(list(xs), tamanos, rotation=0)
        eje.set_xlabel(f"Forma (B={ff[0]['B']}, H={ff[0]['H']})" if "N_CTX" in ff[0]
                       else "Tamaño (M×N×K)" if "K" in ff[0] else "Tamaño (M×N)", color=TINTA_2)
        if fac is not None:
            eje.set_title(str(fac), color=TINTA, fontsize=11)
        eje.grid(axis="y", color=REJILLA, linewidth=0.8, zorder=0)
        eje.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            eje.spines[lado].set_visible(False)
        eje.spines["bottom"].set_color(TINTA_2)
        eje.tick_params(colors=TINTA_2, length=0)
    ejes[0][0].set_ylabel(METRICAS[metrica], color=TINTA_2)
    ejes[0][0].set_ylim(0, tope * 1.18 or 1)
    c = meta.get("contexto", {})
    # Pico de la ficha de NVIDIA: memoria si la metrica es GB/s; si es TFLOP/s, el de la precision
    # de la faceta (dtype) o, sin faceta, la del experimento (run_matmul_fp8 lleva fp16 y FP8).
    unidad = "GB/s" if metrica == "gbs" else "TFLOP/s"
    for eje, fac in zip(ejes[0], facetas):
        if unidad == "GB/s":
            precs = [None]
        elif fac is not None:
            precs = [fichas.precision_de(str(fac))]
        else:
            precs = ["fp8", "fp16"] if "fp8" in nombre else ["fp16"]
        for prec in precs:
            p = fichas.pico(c.get("gpu"), unidad, prec or "fp16")
            if p:
                fichas.dibujar(eje, *p, estilo=":" if len(precs) > 1 and prec == "fp16" else "--")
    fig.suptitle(f"{nombre} — {c.get('gpu', 'GPU')}", color=TINTA, fontsize=12, x=0.02, ha="left")
    asas, etiquetas = ejes[0][0].get_legend_handles_labels()
    if serie_col or len(asas) > len(series):
        fig.legend(asas, etiquetas, frameon=False, ncol=len(asas),
                   loc="upper left", bbox_to_anchor=(0.01, 0.95), labelcolor=TINTA, fontsize=9)
    fig.text(0.02, 0.01, f"Barras: mediana (do_bench). torch {c.get('torch', '?')}, "
             f"triton {c.get('triton', '?')}, CUDA {c.get('cuda', '?')}; job {c.get('slurm_job')}.",
             fontsize=7, color=TINTA_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(destino, f"grafica.{ext}"), dpi=200, facecolor="white")
    plt.close(fig)
    return True


def maquina_de(meta, ejecucion):
    """Maquina de una ejecucion: la registrada en meta.json o, si no, la de su ruta
    results/<maquina>/<experimento>/<ejecucion>."""
    c = meta.get("contexto", {})
    if c.get("maquina"):
        return c["maquina"]
    padre = os.path.basename(os.path.dirname(os.path.dirname(os.path.realpath(ejecucion))))
    return padre if padre != "results" else (c.get("host") or "desconocida")


def generar(ejecucion):
    ejecucion = os.path.realpath(ejecucion)
    filas, meta = leer(ejecucion)
    nombre = meta.get("experimento") or os.path.basename(os.path.dirname(ejecucion))
    maquina = maquina_de(meta, ejecucion)
    columnas = list(dict.fromkeys(k for f in filas for k in f))  # union en orden de aparicion

    md = tabla_md(filas, meta, columnas)
    with open(os.path.join(ejecucion, "tabla.md"), "w") as f:
        f.write(md)
    with open(os.path.join(ejecucion, "tabla.tex"), "w") as f:
        f.write(tabla_tex(filas, meta, columnas, nombre))
    hay_grafica = grafica(filas, meta, ejecucion, nombre)

    # Copia para la memoria del TFM, separada por maquina.
    docs = os.path.join(DOCS, "TFM", "resultados", maquina)
    figs = os.path.join(docs, "figuras")
    os.makedirs(figs, exist_ok=True)
    with open(os.path.join(docs, f"{nombre}.tex"), "w") as f:
        f.write(tabla_tex(filas, meta, columnas, nombre))
    cabecera_img = ""
    if hay_grafica:
        for ext in ("png", "pdf"):
            shutil.copy(os.path.join(ejecucion, f"grafica.{ext}"), os.path.join(figs, f"{nombre}.{ext}"))
        cabecera_img = f"![{nombre}](figuras/{nombre}.png)\n\n"
    origen = os.path.relpath(ejecucion, RAIZ)
    with open(os.path.join(docs, f"{nombre}.md"), "w") as f:
        f.write(f"# {nombre} ({maquina})\n\nGenerado automáticamente desde `{origen}/`.\n\n{cabecera_img}{md}")
    print(f"Informe generado en {origen}/ y docs/TFM/resultados/{maquina}/{nombre}.md", flush=True)


def familia(nombre):
    """Familia de un experimento por su nombre: run_matmul_fp8 -> matmul, run_rmsnorm -> rmsnorm.
    None si no sigue el convenio run_<familia>[_<variante>] (p. ej. la plantilla)."""
    partes = nombre.split("_")
    return partes[1] if len(partes) > 1 and partes[0] == "run" else None


def ultimas_ejecuciones():
    """(maquina, experimento, ruta de 'ultimo') de cada results/<maquina>/<experimento>/ultimo."""
    base = os.path.join(RAIZ, "results")
    for maquina in sorted(os.listdir(base)):
        dir_maquina = os.path.join(base, maquina)
        if not os.path.isdir(dir_maquina):
            continue
        for exp in sorted(os.listdir(dir_maquina)):
            ultimo = os.path.join(dir_maquina, exp, "ultimo")
            if os.path.isfile(os.path.join(ultimo, "resultados.csv")):
                yield maquina, exp, ultimo


def _escribir_csv(destino, filas):
    columnas = list(dict.fromkeys(k for f in filas for k in f))  # union en orden de aparicion
    with open(destino, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)
    print(f"Metricas consolidadas en {os.path.relpath(destino, RAIZ)}", flush=True)


def actualizar_metricas(fam):
    """Regenera las tablas consolidadas de una familia con la ULTIMA ejecucion de cada
    experimento EN CADA MAQUINA (results/<maquina>/run_<fam>*/ultimo):
        results/<fam>_metrics.csv            todas las maquinas (columna 'maquina')
        results/<maquina>/<fam>_metrics.csv  solo esa maquina
    Idempotente: no acumula duplicados. Los datos crudos siguen en cada ejecucion."""
    por_maquina = {}
    for maquina, exp, ultimo in ultimas_ejecuciones():
        if familia(exp) != fam:
            continue
        with open(os.path.join(ultimo, "resultados.csv")) as f:
            medidas = list(csv.DictReader(f))
        with open(os.path.join(ultimo, "meta.json")) as f:
            c = json.load(f).get("contexto", {})
        origen = {"maquina": maquina, "experimento": exp,
                  "ejecucion": os.path.basename(os.path.realpath(ultimo)),
                  "gpu": c.get("gpu"), "driver_nvidia": c.get("driver_nvidia"), "fecha": c.get("fecha")}
        por_maquina.setdefault(maquina, []).extend({**origen, **m} for m in medidas)
    if not por_maquina:
        return None
    for maquina, filas in por_maquina.items():
        _escribir_csv(os.path.join(RAIZ, "results", maquina, f"{fam}_metrics.csv"), filas)
    destino = os.path.join(RAIZ, "results", f"{fam}_metrics.csv")
    _escribir_csv(destino, [f for filas in por_maquina.values() for f in filas])
    return destino


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("uso: python3.11 benchmarks/informe.py results/<maquina>/<experimento>/<ejecucion|ultimo>")
    generar(sys.argv[1])
    for fam in sorted({familia(e) for _, e, _ in ultimas_ejecuciones()} - {None}):
        actualizar_metricas(fam)
