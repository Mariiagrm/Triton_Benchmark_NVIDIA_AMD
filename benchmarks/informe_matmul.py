"""Informe del baseline matmul: tabla (Markdown y LaTeX) y grafica (PNG y PDF).

Lo llama run_matmul.py al terminar. Tambien se puede regenerar a mano para
cualquier ejecucion (no necesita GPU ni torch; basta matplotlib):
    python3.11 benchmarks/informe_matmul.py                       # la ultima
    python3.11 benchmarks/informe_matmul.py results/run_matmul/<ejecucion>

Salida:
    <ejecucion>/tabla.md, tabla.tex, grafica.png, grafica.pdf
    docs/TFM/resultados/run_matmul.md (+ figuras/)  -> copia de la ultima, para la memoria
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

# Paleta categorica de referencia (slots 1 y 2, validados en modo claro) y tintas de texto.
COLOR = {"triton": "#2a78d6", "cublas": "#eb6834"}
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def leer(ejecucion):
    with open(os.path.join(ejecucion, "resultados.csv")) as f:
        filas = list(csv.DictReader(f))
    with open(os.path.join(ejecucion, "meta.json")) as f:
        meta = json.load(f)
    for fila in filas:
        for k, v in fila.items():
            try:
                fila[k] = int(v)
            except ValueError:
                try:
                    fila[k] = float(v)
                except ValueError:
                    pass
    return filas, meta


def tflops(fila, ms):
    return 2 * fila["M"] * fila["N"] * fila["K"] / (ms * 1e-3) / 1e12


def filas_tabla(filas):
    """Filas ya formateadas: comunes a Markdown y LaTeX."""
    out = []
    for f in filas:
        out.append([
            f["dtype"], f"{f['M']}×{f['N']}×{f['K']}",
            f"{f['BLOCK_SIZE_M']}×{f['BLOCK_SIZE_N']}×{f['BLOCK_SIZE_K']}, w{f['num_warps']}, s{f['num_stages']}",
            f"{f['triton_ms']:.3f}", f"{f['triton_tflops']:.1f}",
            f"{f['cublas_ms']:.3f}", f"{f['cublas_tflops']:.1f}",
            f"{f['triton_tflops'] / f['cublas_tflops']:.1%}",
        ])
    return out


CABECERA = ["dtype", "M×N×K", "Config. Triton (BM×BN×BK, warps, stages)", "Triton (ms)",
            "Triton (TFLOP/s)", "cuBLAS (ms)", "cuBLAS (TFLOP/s)", "Triton/cuBLAS"]


def pie(meta):
    c = meta["contexto"]
    return (f"{c['gpu']} (sm_{c['sm']}), torch {c['torch']}, triton {c['triton']}, CUDA {c['cuda']}; "
            f"job {c['slurm_job']}, {c['fecha']}. Mediana de triton.testing.do_bench.")


def tabla_md(filas, meta):
    lineas = ["| " + " | ".join(CABECERA) + " |",
              "|" + "|".join([":---", ":---", ":---"] + ["---:"] * 5) + "|"]
    lineas += ["| " + " | ".join(r) + " |" for r in filas_tabla(filas)]
    texto = "\n".join(lineas) + f"\n\n*{pie(meta)}*\n"
    mma = sorted({i for f in filas for i in str(f.get("triton_mma") or "").split(";") if i})
    if mma:  # ejecuciones anteriores a la comprobacion de tensor cores no traen la columna
        texto += f"\nTensor cores (PTX Triton): `{'`, `'.join(mma)}`\n"
    return texto


def tabla_tex(filas, meta):
    esc = lambda s: s.replace("×", r"$\times$").replace("%", r"\%")  # noqa: E731
    cab = ["dtype", r"$M \times N \times K$", "Config. Triton", "Triton (ms)", "Triton (TFLOP/s)",
           "cuBLAS (ms)", "cuBLAS (TFLOP/s)", "Triton/cuBLAS"]
    cuerpo = "\n".join("    " + " & ".join(esc(x) for x in r) + r" \\" for r in filas_tabla(filas))
    return (r"% Requiere \usepackage{booktabs}" "\n"
            r"\begin{table}[htbp]" "\n" r"  \centering\small" "\n"
            r"  \begin{tabular}{lllrrrrr}" "\n" r"    \toprule" "\n"
            "    " + " & ".join(cab) + r" \\" "\n" r"    \midrule" "\n"
            + cuerpo + "\n" r"    \bottomrule" "\n" r"  \end{tabular}" "\n"
            r"  \caption{Baseline matmul: Triton con \texttt{@triton.autotune} frente a cuBLAS. "
            + esc(pie(meta)).replace("_", r"\_") + "}\n"
            r"  \label{tab:baseline-matmul}" "\n" r"\end{table}" "\n")


def grafica(filas, meta, destino):
    dtypes = list(dict.fromkeys(f["dtype"] for f in filas))
    fig, ejes = plt.subplots(1, len(dtypes), figsize=(6.4 * len(dtypes), 4.2), sharey=True, squeeze=False)
    ancho = 0.38
    for eje, dt in zip(ejes[0], dtypes):
        fs = [f for f in filas if f["dtype"] == dt]
        xs = range(len(fs))
        for i, (lib, nombre) in enumerate([("triton", "Triton (autotune)"), ("cublas", "cuBLAS")]):
            pos = [x + (i - 0.5) * (ancho + 0.02) for x in xs]  # hueco entre barras
            med = [f[f"{lib}_tflops"] for f in fs]
            # Bigotes p20-p80 (en TFLOP/s: menos ms = mas TFLOP/s).
            bajo = [m - tflops(f, f[f"{lib}_ms_p80"]) for m, f in zip(med, fs)]
            alto = [tflops(f, f[f"{lib}_ms_p20"]) - m for m, f in zip(med, fs)]
            eje.bar(pos, med, ancho, color=COLOR[lib], label=nombre, zorder=2)
            eje.errorbar(pos, med, yerr=[bajo, alto], fmt="none", ecolor=TINTA_2, elinewidth=1, capsize=3, zorder=3)
            for p, m, a in zip(pos, med, alto):
                eje.text(p, m + a + 1.5, f"{m:.1f}", ha="center", va="bottom", fontsize=9, color=TINTA)
        eje.set_xticks(list(xs), [f"{f['M']}³" for f in fs])
        eje.set_xlabel("Tamaño (M = N = K)", color=TINTA_2)
        eje.set_title(dt, color=TINTA, fontsize=11)
        eje.grid(axis="y", color=REJILLA, linewidth=0.8, zorder=0)
        eje.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            eje.spines[lado].set_visible(False)
        eje.spines["bottom"].set_color(TINTA_2)
        eje.tick_params(colors=TINTA_2, length=0)
    ejes[0][0].set_ylabel("TFLOP/s (mayor es mejor)", color=TINTA_2)
    tope = max(max(tflops(f, f["triton_ms_p20"]), tflops(f, f["cublas_ms_p20"])) for f in filas)
    ejes[0][0].set_ylim(0, tope * 1.15)
    c = meta["contexto"]
    fig.suptitle(f"Matmul en {c['gpu']}: Triton frente a cuBLAS", color=TINTA, fontsize=12, x=0.02, ha="left")
    # Leyenda fuera del area de datos, bajo el titulo, para no tapar barras.
    fig.legend(*ejes[0][0].get_legend_handles_labels(), frameon=False, ncol=2, loc="upper left",
               bbox_to_anchor=(0.01, 0.94), labelcolor=TINTA, fontsize=9)
    fig.text(0.02, 0.01, f"Barras: mediana; bigotes: percentiles 20–80 (do_bench). "
             f"torch {c['torch']}, triton {c['triton']}, CUDA {c['cuda']}; job {c['slurm_job']}.",
             fontsize=7, color=TINTA_2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(destino, f"grafica.{ext}"), dpi=200, facecolor="white")
    plt.close(fig)


def generar(ejecucion=None):
    ejecucion = os.path.realpath(ejecucion or os.path.join(RAIZ, "results", "run_matmul", "ultimo"))
    filas, meta = leer(ejecucion)
    md = tabla_md(filas, meta)
    with open(os.path.join(ejecucion, "tabla.md"), "w") as f:
        f.write(md)
    with open(os.path.join(ejecucion, "tabla.tex"), "w") as f:
        f.write(tabla_tex(filas, meta))
    grafica(filas, meta, ejecucion)

    # Copia para la memoria del TFM (siempre la ejecucion pedida, normalmente la ultima).
    docs = os.path.join(RAIZ, "docs", "TFM", "resultados")
    figs = os.path.join(docs, "figuras")
    os.makedirs(figs, exist_ok=True)
    for ext in ("png", "pdf"):
        shutil.copy(os.path.join(ejecucion, f"grafica.{ext}"), os.path.join(figs, f"run_matmul.{ext}"))
    shutil.copy(os.path.join(ejecucion, "tabla.tex"), os.path.join(docs, "run_matmul.tex"))
    origen = os.path.relpath(ejecucion, RAIZ)
    with open(os.path.join(docs, "run_matmul.md"), "w") as f:
        f.write(f"# Baseline matmul\n\nGenerado automáticamente desde `{origen}/`.\n\n"
                f"![Baseline matmul](figuras/run_matmul.png)\n\n{md}")
    print(f"Informe generado en {origen}/ y docs/TFM/resultados/run_matmul.md", flush=True)


if __name__ == "__main__":
    generar(sys.argv[1] if len(sys.argv) > 1 else None)
