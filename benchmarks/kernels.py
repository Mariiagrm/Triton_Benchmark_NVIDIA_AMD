"""Kernels del TFM: PROCEDENCIA + RESULTADOS en una sola tabla por algoritmo.

Junta, por DSL, de donde sale cada kernel (benchmarks/procedencia.py: elaboracion y
fuente) con su ultimo resultado en cada arquitectura (results/resumen.csv, el del resumen):
rendimiento, % del pico y entorno/job. Una celda sin dato dice por que falta (resumen.motivo).

La procedencia se lee de benchmarks/procedencia.py (unica fuente, escrita a mano); este modulo solo
sabe a que celda del resumen corresponde cada kernel (MAPA). Salida:
    docs/TFM/resultados/kernels.md

Lo llama resumen.generar() (y con el, comun.guardar() tras cada ejecucion). A mano, en el login:
    python3.11 benchmarks/kernels.py
"""
import csv
import os
import re

import informe
import resumen

# Kernel (primer `...` de la 1a columna de las tablas de procedencia.TEXTO) -> filas/columnas del resumen.
FA = ["FlashAttention fp16", "FlashAttention causal fp16"]
RMS = ["RMSNorm fp16", "RMSNorm bf16"]
MAPA = {
    "triton/matmul.py": ("Triton", ["Matmul fp16"]),
    "triton/matmul_blockptr.py": ("Triton (block-ptr)", ["Matmul fp16"]),
    "triton/matmul_tma.py": ("Triton + TMA", ["Matmul fp16"]),
    "triton/matmul_fp8.py": ("Triton", ["Matmul FP8"]),
    "triton/matmul_fp8_tma.py": ("Triton + TMA", ["Matmul FP8"]),
    "gluon/matmul.py": ("Gluon", ["Matmul fp16"]),
    "helion/matmul.py": ("Helion", ["Matmul fp16"]),
    "cutlass/matmul_cute.py": ("CUTLASS", ["Matmul fp16"]),
    "triton/fused_attention.py": ("Triton", FA),
    "helion/attention.py": ("Helion", FA),
    "cutlass/attention_fa2.py": ("CUTLASS", FA),
    "triton/rmsnorm_baseline.py": ("Triton", RMS),
    "triton_tlx/rmsnorm.py": ("Triton + TLX", RMS),
    "gluon/rmsnorm.py": ("Gluon", RMS),
    "helion/rmsnorm.py": ("Helion", RMS),
    "cutlass/rmsnorm_ext.cu": ("CUTLASS", RMS),
    "torch.matmul": ("Referencia", ["Matmul fp16"]),
    "torch._scaled_mm": ("Referencia", ["Matmul FP8"]),
    "torch.nn.functional.scaled_dot_product_attention": ("Referencia", FA),
    "torch.nn.functional.rms_norm": ("Referencia", RMS),
}
# Filas sin kernel en procedencia ('—' en la 1a columna): por DSL, en la seccion de FlashAttention.
DSL_FILA = {"Triton": "Triton", "Gluon": "Gluon", "Triton + TLX": "Triton + TLX", "Helion": "Helion",
            "CUTLASS (CuTe DSL)": "CUTLASS", "CUTLASS (C++/CUDA)": "CUTLASS", "CUTLASS": "CUTLASS"}

GRUPOS = [("Matmul", ["Matmul fp16", "Matmul FP8"]),
          ("FlashAttention (forward)", FA),
          ("RMSNorm", RMS)]

import procedencia  # noqa: E402  (texto de la procedencia: benchmarks/procedencia.py)


def _celdas_md(linea):
    return [c.strip() for c in linea.strip().strip("|").split("|")]


def leer_procedencia(texto=None):
    """{(fila, columna): [ {kernel, elaboracion, fuente} ]} y el texto de la seccion 'Fuentes'."""
    texto = texto or procedencia.TEXTO
    proc, seccion = {}, ""
    for linea in texto.splitlines():
        if linea.startswith("## "):
            seccion = linea[3:]
            continue
        if not linea.startswith("| ") or linea.startswith("|:") or seccion == "Fuentes":
            continue
        c = _celdas_md(linea)
        m = re.match(r"`([^`]+)`", c[0])
        if seccion.startswith("Referencias"):
            if not m or m.group(1) not in MAPA:
                continue
            kernel, elab, fuente = c[0], "Referencia (no es kernel propio)", c[2]
        elif len(c) >= 4 and c[1] != "DSL":
            kernel, elab, fuente = c[0], c[2], c[3]
            if len(c) >= 5 and c[4]:
                fuente += f" · ¿corre en sm_120/121? {c[4].replace('ver abajo', 'ver notas de procedencia')}"
        else:
            continue
        if m and m.group(1) in MAPA:
            fila, cols = MAPA[m.group(1)]
        elif c[0] == "—" and "FlashAttention" in seccion and c[1] in DSL_FILA:
            fila, cols = DSL_FILA[c[1]], FA
        else:
            continue
        for col in cols:
            proc.setdefault((fila, col), []).append({"kernel": kernel, "elaboracion": elab, "fuente": fuente})
    fuentes = texto.split("## Fuentes", 1)[1].strip() if "## Fuentes" in texto else ""
    return proc, fuentes


def leer_resultados():
    """{(arquitectura, fila, columna): celda} de results/resumen.csv."""
    ruta = os.path.join(resumen.RAIZ, "results", "resumen.csv")
    with open(ruta) as fh:
        return {(c["arquitectura"], c["fila"], c["columna"]): c for c in csv.DictReader(fh)}


def corto(arq):
    """'NVIDIA GeForce RTX 5090 (sm_120)' -> 'RTX 5090'."""
    return re.sub(r"^NVIDIA (GeForce )?| \(sm_\d+\)$", "", arq)


def _resultado(arq, fila, col, res):
    c = res.get((arq, fila, col))
    if c is None:
        return resumen._texto_vacia(arq, fila, col)
    pct = f" · **{float(c['pct']):.0f} %**" if c.get("pct") not in ("", None) else " · *sin pico de referencia*"
    extra = f" ({c['extra']})" if c.get("extra") else ""
    return f"**{c['valor']} {c['unidad']}**{pct}{extra}<br>_{c['maquina']}, {resumen._job(c)}_"


def _unir(entradas, campo, varias):
    """Valores distintos de un campo; con varias columnas (fp16/FP8) se dice a cual corresponde."""
    vistos = {}
    for col, e in entradas:
        vistos.setdefault(e[campo], []).append(col)
    if len(vistos) == 1:
        return next(iter(vistos))
    return "<br>".join(f"*{', '.join(fichas_cols(cs))}*: {v}" if varias else v for v, cs in vistos.items())


def fichas_cols(cols):
    return [c.split()[-1] if "causal" not in c else "causal" for c in cols]


def tabla(grupo, cols, proc, res, arqs):
    cab = ["DSL", "Kernel", "Elaboración", "Fuente"] + [f"{corto(a)} {fichas_cols([c])[0]}" for a in arqs
                                                        for c in cols]
    lineas = ["| " + " | ".join(cab) + " |", "|" + "|".join([":---"] * len(cab)) + "|"]
    for fila in resumen.FILAS:
        entradas = [(c, e) for c in cols for e in proc.get((fila, c), [])]
        vacias = [resumen.motivo(a, fila, c)[0] for a in arqs for c in cols if (a, fila, c) not in res]
        if not entradas and all(v == "no aplica" for v in vacias) and len(vacias) == len(arqs) * len(cols):
            continue  # la variante no existe en este algoritmo (p. ej. block-ptr en atencion)
        if entradas:
            kernel = _unir(entradas, "kernel", len(cols) > 1)
            elab = _unir(entradas, "elaboracion", len(cols) > 1)
            fuente = _unir(entradas, "fuente", len(cols) > 1)
        else:
            kernel, elab, fuente = "—", "sin kernel", "—"
        celdas = [_resultado(a, fila, c, res) for a in arqs for c in cols]
        lineas.append(f"| **{fila}** | {kernel} | {elab} | {fuente} | " + " | ".join(celdas) + " |")
    return f"## {grupo}\n\n" + "\n".join(lineas) + "\n"


def generar():
    proc, fuentes = leer_procedencia()
    res = leer_resultados()
    arqs = sorted({a for a, _, _ in res}, key=lambda a: ("GB10" not in a, a))
    partes = [
        "# Kernels: procedencia y resultados\n",
        "*Generado por `benchmarks/kernels.py`. Para cada DSL: de dónde sale el kernel (detalle en "
        "la sección de procedencia) y su último resultado en cada arquitectura "
        "(el mismo que el resumen: [resumen.md](resumen.md), con la configuración aplicada). Cada "
        "resultado: rendimiento en el mayor tamaño medido · **% del pico** · entorno y job. Una celda "
        "sin dato dice por qué falta; n/a = la variante no existe para ese algoritmo.*\n",
        resumen.NOTA_PICOS.strip() + "\n",
    ]
    for grupo, cols in GRUPOS:
        partes.append(tabla(grupo, cols, proc, res, arqs))
    if fuentes:
        partes.append("## Fuentes\n\n" + fuentes + "\n")
    return "\n".join(partes)  # el documento es docs/TFM/comparativa.md, que usa tabla()

if __name__ == "__main__":
    print(generar())
