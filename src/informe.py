"""Informe GENERICO de un experimento: tabla (Markdown + LaTeX) y grafica (PNG + PDF).

Lo llama comun.guardar() automaticamente al final de CADA experimento que no
proporcione su propio informe, asi que cualquier experimento nuevo obtiene tabla y
grafica sin escribir codigo extra. Funciona a partir del resultados.csv + meta.json,
no necesita GPU ni torch (solo matplotlib).

Regenerar a mano cualquier ejecucion:
    python3.11 src/informe.py results/<experimento>/<ejecucion>
    python3.11 src/informe.py results/<experimento>/ultimo

Salida en la ejecucion: tabla.md, tabla.tex, grafica.png, grafica.pdf
Copia para la memoria: docs/resultados/<experimento>.{md,tex} (+ figuras/<experimento>.{png,pdf})

La grafica se dibuja si hay una columna de rendimiento ("tflops"); si no, solo tablas.
"""
import csv
import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = os.environ.get("TFM_RAIZ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paleta categorica de referencia (validada en modo claro) y tintas de texto.
PALETA = ["#2a78d6", "#eb6834", "#2ca089", "#a65cd6", "#d6b12a", "#d6425c"]
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"

# Columna de rendimiento y candidatas a etiqueta/serie (por orden de preferencia).
COL_METRICA = "tflops"
COLS_SERIE = ["variante", "dtype", "kernel"]


def leer(ejecucion):
    with open(os.path.join(ejecucion, "resultados.csv")) as f:
        filas = list(csv.DictReader(f))
    with open(os.path.join(ejecucion, "meta.json")) as f:
        meta = json.load(f)
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
    if all(k in f for k in ("M", "N", "K")):
        return f"{f['M']}×{f['N']}×{f['K']}"
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
            r"  \label{tab:" + nombre.replace("_", "-") + "}\n" r"\end{table}" "\n")


def grafica(filas, meta, destino, nombre):
    """Barras de TFLOP/s. Facetas por dtype; series por 'variante' (o la 1a categoria util)."""
    if not filas or COL_METRICA not in filas[0]:
        return False  # sin columna de rendimiento: no hay grafica

    facet_col = "dtype" if "dtype" in filas[0] else None
    serie_col = next((c for c in COLS_SERIE if c in filas[0] and c != facet_col), None)

    facetas = list(dict.fromkeys(f[facet_col] for f in filas)) if facet_col else [None]
    series = list(dict.fromkeys(f[serie_col] for f in filas)) if serie_col else [nombre]
    color = {s: PALETA[i % len(PALETA)] for i, s in enumerate(series)}

    fig, ejes = plt.subplots(1, len(facetas), figsize=(max(5.0, 2.2 * len(series) + 2.5) * len(facetas), 4.4),
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
                if fila is None:
                    continue
                v = fila[COL_METRICA]
                med.append(v)
                pos.append(x + (i - (len(series) - 1) / 2) * (ancho + 0.02))
                tope = max(tope, v)
            eje.bar(pos, med, ancho, color=color[s], label=str(s), zorder=2)
            for p, m in zip(pos, med):
                eje.text(p, m, f"{m:.0f}", ha="center", va="bottom", fontsize=8, color=TINTA)
        eje.set_xticks(list(xs), tamanos, rotation=0)
        eje.set_xlabel("Tamaño (M×N×K)", color=TINTA_2)
        if fac is not None:
            eje.set_title(str(fac), color=TINTA, fontsize=11)
        eje.grid(axis="y", color=REJILLA, linewidth=0.8, zorder=0)
        eje.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            eje.spines[lado].set_visible(False)
        eje.spines["bottom"].set_color(TINTA_2)
        eje.tick_params(colors=TINTA_2, length=0)
    ejes[0][0].set_ylabel("TFLOP/s (mayor es mejor)", color=TINTA_2)
    ejes[0][0].set_ylim(0, tope * 1.18 or 1)
    c = meta.get("contexto", {})
    fig.suptitle(f"{nombre} — {c.get('gpu', 'GPU')}", color=TINTA, fontsize=12, x=0.02, ha="left")
    if serie_col:
        fig.legend(*ejes[0][0].get_legend_handles_labels(), frameon=False, ncol=len(series),
                   loc="upper left", bbox_to_anchor=(0.01, 0.95), labelcolor=TINTA, fontsize=9)
    fig.text(0.02, 0.01, f"Barras: mediana (do_bench). torch {c.get('torch', '?')}, "
             f"triton {c.get('triton', '?')}, CUDA {c.get('cuda', '?')}; job {c.get('slurm_job')}.",
             fontsize=7, color=TINTA_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(destino, f"grafica.{ext}"), dpi=200, facecolor="white")
    plt.close(fig)
    return True


def generar(ejecucion=None):
    ejecucion = os.path.realpath(ejecucion or os.path.join(RAIZ, "results", "ultimo"))
    filas, meta = leer(ejecucion)
    nombre = meta.get("experimento") or os.path.basename(os.path.dirname(ejecucion))
    columnas = list(dict.fromkeys(k for f in filas for k in f))  # union en orden de aparicion

    md = tabla_md(filas, meta, columnas)
    with open(os.path.join(ejecucion, "tabla.md"), "w") as f:
        f.write(md)
    with open(os.path.join(ejecucion, "tabla.tex"), "w") as f:
        f.write(tabla_tex(filas, meta, columnas, nombre))
    hay_grafica = grafica(filas, meta, ejecucion, nombre)

    # Copia para la memoria del TFM.
    docs = os.path.join(RAIZ, "docs", "resultados")
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
        f.write(f"# {nombre}\n\nGenerado automáticamente desde `{origen}/`.\n\n{cabecera_img}{md}")
    print(f"Informe generado en {origen}/ y docs/resultados/{nombre}.md", flush=True)


if __name__ == "__main__":
    generar(sys.argv[1] if len(sys.argv) > 1 else None)
