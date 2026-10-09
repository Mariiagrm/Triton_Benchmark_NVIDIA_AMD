"""Eficiencia energetica por arquitectura: filas = DSL, columnas = algoritmo (como resumen.py).

Reune las ejecuciones hechas con TFM_ENERGIA=1 (entornos results/<nodo>-energia/), en las que
comun.medir() anade el regimen sostenido de cada kernel. Por celda: el trabajo por julio en el
mayor tamano de la ejecucion mas reciente,
    CB (compute-bound)  -> GFLOP/J = TFLOP/s sostenidos / W
    MB (memory-bound)   -> GB/J    = GB/s sostenidos / W
con la potencia sostenida de la GPU (NVML instantanea), la de reposo, el rendimiento y el reloj
sostenidos, la energia por llamada y la eficiencia respecto a la referencia (cuBLAS, SDPA,
F.rms_norm) en la misma GPU. La version "dinamica" descuenta la potencia en reposo.

Salida:
    results/energia.csv                           formato largo (una fila por celda)
    docs/TFM/resultados/energia.md          una tabla por arquitectura
    docs/TFM/resultados/figuras/energia_{compute,memory}_bound.{png,pdf}

Lo llama comun.guardar() al final de cada ejecucion con TFM_ENERGIA=1. A mano (login, sin GPU):
    python3.11 benchmarks/energia.py
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import informe  # noqa: E402
import resumen  # noqa: E402

RAIZ = informe.RAIZ
UNIDAD = {"CB": "GFLOP/J", "MB": "GB/J"}
CLAVE = {"CB": ("gflop_j", "gflop_j_din", "tflops_sost", "TFLOP/s"),
         "MB": ("gb_j", "gb_j_din", "gbs_sost", "GB/s")}


# Metodo de calculo: se escribe al principio de energia.md (comun.energia()).
METODO_MD = """## Cómo se calcula

La energía no se mide con un contador de julios: se mide la **potencia** de la GPU mientras el
kernel se ejecuta sin parar y se combina con el trabajo hecho en ese tiempo (`comun.energia()`,
activado con `TFM_ENERGIA=1`).

1. **Potencia en reposo**, una vez por script: 4 s sin carga y mediana de los 2 últimos
   (`potencia_reposo_w`).
2. **Régimen sostenido** de cada kernel, después de su medida con `do_bench`: 3 s de reposo
   (todos parten del mismo estado térmico) y 6 s en bucle, en lotes de ~50 ms seguidos de una
   sincronización. Los 2 primeros segundos se descartan, porque el reloj tarda ~0.5 s en
   estabilizarse bajo carga y NVML en reflejarlo. Cuenta el tramo final de 4 s: llamadas
   completadas / 4 s = **ritmo** (llamadas/s). Duraciones configurables con
   `TFM_ENERGIA_SEGUNDOS` y `TFM_ENERGIA_DESCARTE`.
3. **Potencia y reloj**: un hilo lee de NVML cada 20 ms el reloj SM y la potencia
   **instantánea** de la GPU (`NVML_FI_DEV_POWER_INSTANT`; la de `nvmlDeviceGetPowerUsage` es un
   promedio retrasado ~1 s). Se toma la mediana de las muestras del tramo medido.
4. **Cálculo**:

| magnitud | fórmula |
|:---|:---|
| rendimiento sostenido | ritmo × FLOP por llamada (TFLOP/s) o × bytes por llamada (GB/s) |
| energía por llamada | potencia / ritmo (mJ) |
| eficiencia CB | TFLOP/s sostenidos × 1000 / W = **GFLOP/J** (1 W = 1 J/s) |
| eficiencia MB | GB/s sostenidos / W = **GB/J** |
| eficiencia dinámica | igual, dividiendo por (potencia − potencia en reposo) |
| × referencia | eficiencia del DSL / eficiencia de cuBLAS, SDPA o `F.rms_norm` en la misma GPU y forma |

**Limitaciones.** Es solo la potencia de la GPU según NVML: no incluye CPU, memoria ni el resto
del sistema, así que no es comparable con el TDP de 140 W de todo el SoC de GB10. NVML actualiza
cada ~0.5 s, así que el tramo de 4 s tiene unas 8 lecturas distintas. La potencia en reposo se
mide justo después del autotuning y varía entre scripts (12–17 W en GB10): la eficiencia total es
fiable, y la dinámica depende de esa cifra. Cada kernel se mide una sola vez, sin repeticiones.

"""


def _es_energia(maquina, ultimo):
    if maquina.endswith("-energia"):
        return True
    try:
        with open(os.path.join(ultimo, "meta.json")) as fh:
            return bool(json.load(fh).get("contexto", {}).get("energia"))
    except (OSError, ValueError):
        return False


def recoger():
    """Por (arquitectura, DSL, algoritmo): la celda de la ejecucion mas reciente con datos de
    energia y, dentro de ella, la del mayor tamano."""
    mejores = {}
    for maquina, exp, ultimo in informe.ultimas_ejecuciones():
        if not _es_energia(maquina, ultimo):
            continue
        arq, ejecucion = resumen.arquitectura(ultimo)
        for c in resumen.celdas_de(maquina, exp, ultimo):
            tipo = resumen.TIPO[c["columna"]]
            efic, din, sost, unidad_sost = CLAVE[tipo]
            d = c["datos"]
            if d.get(efic) in ("", None):
                continue
            c.update({"arquitectura": arq, "ejecucion": ejecucion, "tipo": tipo,
                      "eficiencia": d[efic], "eficiencia_din": d.get(din), "unidad": UNIDAD[tipo],
                      "rend_sost": d.get(sost), "unidad_rend": unidad_sost,
                      "potencia_w": d.get("potencia_sost_w"), "reposo_w": d.get("potencia_reposo_w"),
                      "reloj_mhz": d.get("reloj_sost_mhz"), "mj_llamada": d.get("mj_llamada")})
            clave = (arq, c["fila"], c["columna"])
            if clave not in mejores or (resumen._fecha(c), c["orden"]) > (
                    resumen._fecha(mejores[clave]), mejores[clave]["orden"]):
                mejores[clave] = c
    celdas = list(mejores.values())
    # Eficiencia relativa a la referencia de la misma GPU y algoritmo.
    ref = {(c["arquitectura"], c["columna"]): c["eficiencia"] for c in celdas if c["fila"] == "Referencia"}
    for c in celdas:
        r = ref.get((c["arquitectura"], c["columna"]))
        c["vs_ref"] = round(c["eficiencia"] / r, 2) if r and c["fila"] != "Referencia" else None
    return celdas


def _cols_filas(celdas):
    cols = [col for col, _ in resumen.COLUMNAS if any(c["columna"] == col for c in celdas)]
    filas = [f for f in resumen.FILAS if any(c["fila"] == f for c in celdas)]
    return cols, filas


def _lineas(c):
    """(valor principal, detalle, origen) de una celda."""
    rel = f" (×{c['vs_ref']:g} ref.)" if c["vs_ref"] else ""
    din = f" · {c['eficiencia_din']:g} {c['unidad']} dinámica" if c["eficiencia_din"] not in ("", None) else ""
    detalle = (f"{c['potencia_w']:g} W (reposo {c['reposo_w']:g} W){din} · {c['rend_sost']:g} {c['unidad_rend']} "
               f"sost. · {c['reloj_mhz']:g} MHz · {c['mj_llamada']:.4g} mJ/llamada · {c['tamano']}")
    return f"{c['eficiencia']:g} {c['unidad']}{rel}", detalle, f"{c['maquina']}, {resumen._job(c)}"


def tabla_md(arq, celdas):
    cols, filas = _cols_filas(celdas)
    lineas = ["| DSL | " + " | ".join(f"{col} ({resumen.TIPO[col]})" for col in cols) + " |",
              "|:---|" + "|".join([":---"] * len(cols)) + "|"]
    for fila in filas:
        de_fila = {c["columna"]: c for c in celdas if c["fila"] == fila}
        textos = []
        for col in cols:
            c = de_fila.get(col)
            if c is None:
                textos.append("—")
                continue
            v, det, org = _lineas(c)
            textos.append(f"**{v}**<br>{det}<br>_{org}_")
        lineas.append(f"| **{fila}** | " + " | ".join(textos) + " |")
    return f"## {arq}\n\n" + "\n".join(lineas) + "\n"


def graficas(celdas, docs):
    """Barras agrupadas de GFLOP/J (CB) y GB/J (MB), una faceta por arquitectura, como resumen.py."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as e:
        print(f"AVISO: no genero las graficas de energia (falta '{e.name}').", flush=True)
        return
    df = pd.DataFrame(celdas)
    if df.empty:
        return
    figs = os.path.join(docs, "figuras")
    os.makedirs(figs, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    specs = [("CB", "viridis", "Compute Bound (GFLOP/J)", "Eficiencia (GFLOP/J)", "energia_compute_bound", 45, 7, 1.3),
             ("MB", "magma", "Memory Bound (GB/J)", "Eficiencia (GB/J)", "energia_memory_bound", 20, 6, 1.2)]
    for tipo, paleta, titulo, ylab, nombre, rot, alto, aspecto in specs:
        sub = df[df["tipo"] == tipo]
        if sub.empty:
            continue
        cols = [c for c, _ in resumen.COLUMNAS if c in set(sub["columna"])]
        dsls = [f for f in resumen.FILAS if f in set(sub["fila"])]
        g = sns.catplot(data=sub, kind="bar", x="columna", y="eficiencia", hue="fila", col="arquitectura",
                        order=cols, hue_order=dsls, height=alto, aspect=aspecto, palette=paleta)
        g.set_axis_labels("Kernel (operación)", ylab)
        g.set_titles("{col_name}")
        for ax in g.axes.flat:
            ax.tick_params(axis="x", rotation=rot)
        g.fig.subplots_adjust(top=0.85, bottom=0.28)
        g.fig.suptitle(f"Eficiencia energética sostenida — {titulo}", fontsize=18, fontweight="bold")
        for ext in ("png", "pdf"):
            g.savefig(os.path.join(figs, f"{nombre}.{ext}"), dpi=150)
        plt.close(g.fig)


def generar():
    celdas = recoger()
    if not celdas:
        print("Sin ejecuciones con datos de energia (TFM_ENERGIA=1).", flush=True)
        return
    celdas.sort(key=lambda c: (c["arquitectura"], resumen.FILAS.index(c["fila"]),
                               [k for k, _ in resumen.COLUMNAS].index(c["columna"])))
    campos = ["arquitectura", "fila", "columna", "tipo", "eficiencia", "unidad", "eficiencia_din", "vs_ref",
              "potencia_w", "reposo_w", "rend_sost", "unidad_rend", "reloj_mhz", "mj_llamada", "tamano", "config",
              "maquina", "experimento", "ejecucion"]
    destino_csv = os.path.join(RAIZ, "results", "energia.csv")
    with open(destino_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(celdas)

    docs = os.path.join(informe.DOCS, "TFM", "resultados")
    os.makedirs(docs, exist_ok=True)
    nota = ("*Generado por `benchmarks/energia.py` a partir de las ejecuciones con `TFM_ENERGIA=1`. CB en GFLOP/J, "
            "MB en GB/J; «×» = eficiencia relativa a la referencia de la misma GPU; «dinámica» descuenta la potencia "
            "en reposo.*\n\n")
    md = ["# Eficiencia energética por arquitectura\n\n", nota, METODO_MD,
          "## Gráficas\n\n![Compute-bound](figuras/energia_compute_bound.png)\n\n"
          "![Memory-bound](figuras/energia_memory_bound.png)\n\n"]
    for arq in sorted({c["arquitectura"] for c in celdas}):
        de_arq = [c for c in celdas if c["arquitectura"] == arq]
        md.append(tabla_md(arq, de_arq) + "\n")
    with open(os.path.join(docs, "energia.md"), "w") as fh:
        fh.write("".join(md))
    try:
        graficas(celdas, docs)
    except Exception as e:
        print(f"AVISO: fallo al generar las graficas de energia ({type(e).__name__}: {e}).", flush=True)
    print(f"Eficiencia energetica en docs/TFM/resultados/energia.md y {os.path.relpath(destino_csv, RAIZ)}",
          flush=True)


if __name__ == "__main__":
    generar()
