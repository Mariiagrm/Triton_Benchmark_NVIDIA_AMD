"""Productividad frente a rendimiento de cada DSL: cuanto codigo y cuanto ajuste cuesta cada kernel.

Para cada (algoritmo, DSL) del resumen reune, sin GPU:
  - LINEAS DE CODIGO LOGICAS (sin blancos, comentarios ni docstrings) que implementan el
    algoritmo, separadas en
      kernel     el codigo del algoritmo (propio u oficial), con su lanzador y su espacio de
                 configuraciones; si el repo carga un ejemplo oficial de fuera (CUTLASS,
                 Helion), se cuenta ESE codigo, descargado en la version fijada del TFM,
      envoltura  codigo propio del repo para cargar/adaptar un kernel externo,
      ajuste     codigo propio para elegir la configuracion cuando el DSL no trae autotuning;
  - el MODO DE AJUSTE y el tamano del espacio de configuraciones (los de Triton/Gluon/CuTe,
    contados en su codigo; los de Helion, del log: "after searching N configs");
  - el TIEMPO DE COMPILACION + AUTOTUNING EN FRIO de la 1a llamada y el rendimiento de esa
    misma ejecucion, de results/<entorno frio>/ (ejecutar.sh con TFM_CACHE_FRIA=1).

El rendimiento se expresa como % del pico: el de mma.sync medido por run_pico_mma (CB) o el de
memoria de la GPU (MB, columna pct_pico de RMSNorm).

Uso (login, sin GPU):
    python3.11 benchmarks/productividad.py [--frio hennessy-frio] [--pico hennessy]
Salida: results/productividad.csv y docs/TFM/resultados/productividad.{md,tex} (+ figuras/)
Detalle y lectura: docs/TFM/productividad.md
"""
import argparse
import ast
import csv
import glob
import io
import json
import os
import re
import subprocess
import tokenize

import informe

RAIZ = informe.RAIZ
FUENTES = os.path.join(RAIZ, ".cache", "fuentes")

# Versiones fijadas del codigo oficial (bitacora: "Enlaces de origen de cada kernel").
CUTLASS = "NVIDIA/cutlass@v4.8.0"
HELION = "pytorch/helion@0f3b242"

# Cada entrada: celda del resumen (columna, fila), experimento en frio que la mide y de donde
# sale su codigo. Fuente: ruta del repo o "<repo>@<rev>:<ruta>" (externa). Seleccion: None
# (todo el fichero) o lista de nombres de nivel superior ("Clase.metodo" para un metodo) y
# rangos "a-b" de lineas (para bloques que no son definiciones con nombre).
ESPEC = [
    # ---- Matmul fp16 ------------------------------------------------------------------------
    dict(columna="Matmul fp16", fila="Triton", exp="run_matmul", fila_csv=None,
         kernel=[("src/CBKernels/triton/matmul.py", None)],
         ajuste_tipo="autotune (triton.autotune)", configs=64,
         configs_nota="3×3×2×2×2 = 72 (BM, BN, BK, warps, etapas), menos 256×256"),
    dict(columna="Matmul fp16", fila="Triton + TMA", exp="run_matmul_tma", fila_csv="tma",
         kernel=[("src/CBKernels/triton/matmul_tma.py", None)],
         ajuste_tipo="autotune (triton.autotune)", configs=128,
         configs_nota="3×3×2×2×4 = 144, menos 256×256"),
    dict(columna="Matmul fp16", fila="Gluon", exp="run_matmul_gluon", fila_csv="gluon",
         kernel=[("src/CBKernels/gluon/matmul.py", None)],
         ajuste=[("benchmarks/CBKernels/gluon/run_matmul_gluon.py", ["elegida", "preparar"])],
         ajuste_tipo="barrido propio (sin autotuning)", configs=7, configs_nota="lista CONFIGS"),
    dict(columna="Matmul fp16", fila="Helion", exp="run_matmul_helion", fila_csv="helion",
         kernel=[("src/CBKernels/helion/matmul.py", None)],
         ajuste_tipo="autotune Helion (quick)", configs="log"),
    dict(columna="Matmul fp16", fila="CUTLASS", exp="run_matmul_cutlass", fila_csv="cutlass",
         kernel=[(f"{CUTLASS}:examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py",
                  ["Sm120GemmKernel"])],
         envoltura=[("src/CBKernels/cutlass/matmul_cute.py", None)],
         ajuste_tipo="fija (tile del ejemplo)", configs=1, configs_nota="--tile a mano"),
    # ---- FlashAttention fp16 (no causal) ----------------------------------------------------
    dict(columna="FlashAttention fp16", fila="Triton", exp="run_attention_triton", fila_csv="triton",
         kernel=[("src/CBKernels/triton/fused_attention.py",
                  ["_attn_fwd_inner", "_host_descriptor_pre_hook", "126-147", "keep", "prune_invalid_configs",
                   "_maybe_make_tensor_desc", "_attn_fwd", "_attention.forward"])],
         ajuste_tipo="autotune (triton.autotune)", configs=36,
         configs_nota="2×3×3×2 (BM, BN, etapas, warps); la poda solo actua en sm_90 o con BM > N"),
    dict(columna="FlashAttention fp16", fila="Helion", exp="run_attention_helion", fila_csv="helion",
         kernel=[(f"{HELION}:examples/attention.py", ["attention_output"])],
         envoltura=[("src/CBKernels/helion/attention.py", None)],
         ajuste_tipo="autotune Helion (quick)", configs="log"),
    dict(columna="FlashAttention fp16", fila="CUTLASS", exp="run_attention_cutlass", fila_csv="cutlass",
         kernel=[(f"{CUTLASS}:examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py",
                  ["FlashAttentionForwardAmpere"])],
         envoltura=[("src/CBKernels/cutlass/attention_fa2.py", None)],
         ajuste=[("benchmarks/CBKernels/cutlass/run_attention_cutlass.py", ["CANDIDATAS", "33-56"])],
         ajuste_tipo="barrido propio (sin autotuning)", configs="candidatas",
         configs_nota="2×2×2 (BM, BN, hilos) filtradas por can_implement"),
    # ---- RMSNorm fp16 -----------------------------------------------------------------------
    dict(columna="RMSNorm fp16", fila="Triton", exp="run_rmsnorm_triton", fila_csv="triton",
         kernel=[("src/MBKernels/triton/rmsnorm_baseline.py", None)],
         ajuste_tipo="heurística (warps por N)", configs=1),
    dict(columna="RMSNorm fp16", fila="Triton + TLX", exp="run_rmsnorm_tlx", fila_csv="tlx",
         kernel=[("src/MBKernels/triton_tlx/rmsnorm.py", None)],
         ajuste_tipo="heurística + --stages a mano", configs=1,
         sin_medida="no compila en GB10 (triton-utlx 3.8.0.post1, ver procedencia_kernels.md)"),
    dict(columna="RMSNorm fp16", fila="Gluon", exp="run_rmsnorm_gluon", fila_csv="gluon",
         kernel=[("src/MBKernels/gluon/rmsnorm.py", None)],
         ajuste_tipo="heurística (warps y layout por N)", configs=1),
    dict(columna="RMSNorm fp16", fila="Helion", exp="run_rmsnorm_helion", fila_csv="helion",
         kernel=[("src/MBKernels/helion/rmsnorm.py", None)],
         ajuste_tipo="autotune Helion (quick)", configs="log"),
    dict(columna="RMSNorm fp16", fila="CUTLASS", exp="run_rmsnorm_cutlass", fila_csv="cutlass",
         kernel=[(f"{CUTLASS}:tools/util/include/cutlass/util/device_rmsnorm.h", None)],
         envoltura=[("src/MBKernels/cutlass/rmsnorm_ext.cu", None), ("src/MBKernels/cutlass/rmsnorm.py", None)],
         ajuste_tipo="fija (heurística de CUTLASS)", configs=1),
]


# --- Fuentes --------------------------------------------------------------------------------

def ruta_fuente(fuente):
    """Ruta local de una fuente del repo o externa (se descarga una vez a .cache/fuentes/)."""
    if ":" not in fuente:
        return os.path.join(RAIZ, fuente)
    repo_rev, ruta = fuente.split(":", 1)
    repo, rev = repo_rev.split("@")
    local = os.path.join(FUENTES, f"{repo.replace('/', '_')}@{rev}", ruta)
    if not os.path.isfile(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        url = f"https://raw.githubusercontent.com/{repo}/{rev}/{ruta}"
        subprocess.run(["curl", "-sSf", "--max-time", "60", url, "-o", local], check=True)
    return local


def url_fuente(fuente):
    if ":" not in fuente:
        return fuente
    repo_rev, ruta = fuente.split(":", 1)
    repo, rev = repo_rev.split("@")
    return f"https://github.com/{repo}/blob/{rev}/{ruta}"


# --- Lineas de codigo logicas -----------------------------------------------------------------

def _lineas_codigo_py(texto):
    """Lineas con codigo (sin comentarios, blancos ni docstrings) de un fuente Python."""
    lineas = set()
    for tok in tokenize.generate_tokens(io.StringIO(texto).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                            tokenize.DEDENT, tokenize.ENDMARKER):
            lineas.update(range(tok.start[0], tok.end[0] + 1))
    # Docstrings y demas cadenas sueltas (expresiones que solo son una cadena).
    for nodo in ast.walk(ast.parse(texto)):
        if isinstance(nodo, ast.Expr) and isinstance(nodo.value, ast.Constant) and isinstance(nodo.value.value, str):
            lineas -= set(range(nodo.lineno, nodo.end_lineno + 1))
    return lineas


def _rango(nodo):
    inicio = min([nodo.lineno] + [d.lineno for d in getattr(nodo, "decorator_list", [])])
    return set(range(inicio, nodo.end_lineno + 1))


def _nombres(nodo):
    if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [nodo.name]
    if isinstance(nodo, ast.Assign):
        return [t.id for t in nodo.targets if isinstance(t, ast.Name)]
    if isinstance(nodo, ast.AnnAssign) and isinstance(nodo.target, ast.Name):
        return [nodo.target.id]
    return []


def _seleccion_py(texto, seleccion):
    """Lineas de la seleccion: nombres de nivel superior, 'Clase.metodo' o rangos 'a-b'."""
    arbol = ast.parse(texto)
    lineas = set()
    for s in seleccion:
        if re.fullmatch(r"\d+-\d+", s):
            a, b = map(int, s.split("-"))
            lineas |= set(range(a, b + 1))
            continue
        clase, _, metodo = s.partition(".")
        nodo = next((n for n in arbol.body if clase in _nombres(n)), None)
        if nodo is None:
            raise KeyError(f"{s!r} no esta en el nivel superior")
        if metodo:
            nodo = next((m for m in nodo.body if getattr(m, "name", None) == metodo), None)
            if nodo is None:
                raise KeyError(f"{s!r}: metodo no encontrado")
        lineas |= _rango(nodo)
    return lineas


def _lineas_codigo_cpp(texto):
    sin = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), texto, flags=re.S)
    sin = re.sub(r"//[^\n]*", "", sin)
    return {i for i, l in enumerate(sin.splitlines(), 1) if l.strip()}


def loc(fuente, seleccion=None):
    """Lineas de codigo logicas de una fuente (toda o la seleccion)."""
    ruta = ruta_fuente(fuente)
    texto = open(ruta, encoding="utf-8").read()
    if ruta.endswith(".py"):
        codigo = _lineas_codigo_py(texto)
        return len(codigo & _seleccion_py(texto, seleccion)) if seleccion else len(codigo)
    if seleccion:
        raise ValueError(f"seleccion por nombre solo en Python: {fuente}")
    return len(_lineas_codigo_cpp(texto))


# --- Medidas en frio --------------------------------------------------------------------------

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def medida_fria(entorno, e):
    """(t_compilacion_s, rendimiento, unidad, pct_pico_mem, reloj, forma, job, error) de la ejecucion en frio."""
    ultimo = os.path.join(RAIZ, "results", entorno, e["exp"], "ultimo")
    if not os.path.isfile(os.path.join(ultimo, "resultados.csv")):
        return None
    with open(os.path.join(ultimo, "resultados.csv")) as f:
        filas = list(csv.DictReader(f))
    with open(os.path.join(ultimo, "meta.json")) as f:
        meta = json.load(f)
    if e["fila_csv"] is None:  # run_matmul: una fila por forma con prefijos triton_/cublas_
        f = max(filas, key=lambda r: int(r["M"]))
        t, valor, reloj = _num(f.get("autotune_s")), _num(f.get("triton_tflops")), _num(f.get("triton_reloj_mhz"))
        forma, unidad, pct_mem = f"{f['M']}³", "TFLOP/s", None
    else:
        cand = [r for r in filas if r.get("variante") == e["fila_csv"]]
        if not cand:
            return None
        f = cand[-1]
        t = _num(f.get("t_compilacion_s")) if "t_compilacion_s" in f else _num(f.get("autotune_s"))
        valor = _num(f.get("tflops")) if _num(f.get("tflops")) else _num(f.get("gbs"))
        unidad = "TFLOP/s" if _num(f.get("tflops")) else "GB/s"
        reloj, pct_mem = _num(f.get("reloj_mhz")), _num(f.get("pct_pico"))
        forma = (f"N={f['N_CTX']} D={f['HEAD_DIM']}" if "N_CTX" in f
                 else f"{f['M']}×{f['N']}" + (f"×{f['K']}" if f.get("K") else ""))
    return {"t_frio_s": t, "valor": valor, "unidad": unidad, "pct_pico_mem": pct_mem, "reloj_mhz": reloj,
            "forma": forma, "job": meta.get("contexto", {}).get("slurm_job"),
            "cache_fria": meta.get("contexto", {}).get("cache_fria"), "error": f.get("error") or ""}


def configs_del_log(entorno, e):
    """Configuraciones evaluadas segun el log del job en frio: Helion ('after searching N configs')
    o el barrido de CuTe (lineas 'candidata ...'). None si no hay log."""
    ultimo = os.path.realpath(os.path.join(RAIZ, "results", entorno, e["exp"], "ultimo"))
    job = re.search(r"job(\d+)", ultimo)
    logs = glob.glob(os.path.join(RAIZ, "logs", f"*-{job.group(1)}.out")) if job else []
    if not logs:
        return None
    texto = open(logs[0], errors="replace").read()
    # Solo la seccion de este experimento (un job puede ejecutar varios benchmarks).
    m = re.search(rf"== Ejecutando \S*{e['exp']}\.py.*?(?=== Ejecutando |\Z)", texto, re.S)
    seccion = m.group(0) if m else texto
    if e["configs"] == "log":
        n = re.findall(r"after searching (\d+) configs", seccion)
        return int(n[-1]) if n else None
    return len(re.findall(r"^\s+candidata ", seccion, re.M)) or None


def pico_mma(entorno):
    """Pico de mma.sync f16->f32 (TFLOP/s) medido por run_pico_mma en ese entorno."""
    ruta = os.path.join(RAIZ, "results", entorno, "run_pico_mma", "ultimo", "meta.json")
    with open(ruta) as f:
        return json.load(f)["resumen"]["pico"]["f16·f32acc"]["tflops"]


# --- Tabla, grafica, informe ------------------------------------------------------------------

def recoger(entorno_frio, entorno_pico):
    pico = pico_mma(entorno_pico)
    filas = []
    for e in ESPEC:
        partes = {p: sum(loc(f, s) for f, s in e.get(p, [])) for p in ("kernel", "envoltura", "ajuste")}
        fila = {"algoritmo": e["columna"], "dsl": e["fila"],
                "loc_kernel": partes["kernel"], "loc_envoltura": partes["envoltura"],
                "loc_ajuste": partes["ajuste"], "loc_propias": partes["envoltura"] + partes["ajuste"]
                + (partes["kernel"] if all(":" not in f for f, _ in e["kernel"]) else 0),
                "loc_total": sum(partes.values()),
                "origen_kernel": "; ".join(url_fuente(f) for f, _ in e["kernel"]),
                "ajuste": e["ajuste_tipo"]}
        m = medida_fria(entorno_frio, e)
        configs = e["configs"]
        if configs in ("log", "candidatas"):
            configs = configs_del_log(entorno_frio, e) if m else None
        fila["configs"] = configs
        fila["configs_nota"] = e.get("configs_nota", "")
        if m:
            fila.update({k: m[k] for k in ("forma", "t_frio_s", "valor", "unidad", "reloj_mhz", "job", "cache_fria")})
            if m["valor"] is not None:
                fila["pct_pico"] = (round(m["pct_pico_mem"], 1) if m["unidad"] == "GB/s" and m["pct_pico_mem"]
                                    else round(100 * m["valor"] / pico, 1) if m["unidad"] == "TFLOP/s" else None)
            fila["error"] = m["error"][:120]
        elif e.get("sin_medida"):
            fila["error"] = e["sin_medida"]
        filas.append(fila)
    return filas, pico


def _fmt_t(t):
    if t is None:
        return "—"
    return f"{t:.1f} s" if t < 120 else f"{t / 60:.1f} min"


def tabla(filas):
    out = []
    for f in filas:
        out.append({
            "algoritmo": f["algoritmo"], "DSL": f["dsl"],
            "LoC kernel": f["loc_kernel"], "LoC propias": f["loc_propias"], "LoC total": f["loc_total"],
            "ajuste": f["ajuste"], "configs": f["configs"] if f.get("configs") is not None else "—",
            "1a llamada en frío": _fmt_t(f.get("t_frio_s")),
            "rendimiento": (f"{f['valor']:g} {f['unidad']}" if f.get("valor") is not None
                            else (f.get("error") or "—")),
            "% del pico": f.get("pct_pico") if f.get("pct_pico") is not None else "—",
        })
    return out


# Color por DSL en orden fijo (sigue a la entidad, no al rango) y marcador por algoritmo.
DSLS = ["Triton", "Triton + TMA", "Gluon", "Helion", "CUTLASS", "Triton + TLX"]
MARCAS = {"Matmul fp16": "o", "FlashAttention fp16": "s", "RMSNorm fp16": "^"}


def grafica(filas, destino, pico, nota):
    """Pequenos multiplos: columnas = algoritmo, filas = eje x (codigo | coste de ajuste)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    color = {d: informe.PALETA[i] for i, d in enumerate(DSLS)}
    algoritmos = list(MARCAS)
    ejes_x = (("loc_total", "líneas de código (kernel + envoltura + ajuste)"),
              ("t_frio_s", "1ª llamada en frío: compilación + autotuning (s)"))
    fig, ejes = plt.subplots(2, len(algoritmos), figsize=(13, 7.6), sharey=True, squeeze=False)
    entero = FuncFormatter(lambda v, _: f"{v:g}")
    for fila_ej, (clave, etiqueta) in enumerate(ejes_x):
        for col, alg in enumerate(algoritmos):
            eje = ejes[fila_ej][col]
            puntos = [(f[clave], f["pct_pico"], f["dsl"]) for f in filas
                      if f["algoritmo"] == alg and f.get(clave) and f.get("pct_pico") is not None]
            colocados = []
            for x, y, dsl in sorted(puntos, key=lambda p: -p[1]):
                eje.scatter([x], [y], s=80, marker=MARCAS[alg], color=color[dsl], edgecolors="white",
                            linewidths=1.5, zorder=3)
                # Etiquetas de puntos casi coincidentes: se apilan hacia abajo.
                k = sum(1 for (cx, cy) in colocados if abs(cy - y) < 4 and 0.4 < cx / x < 2.5)
                colocados.append((x, y))
                eje.annotate(dsl, (x, y), xytext=(7, 3 - 11 * k), textcoords="offset points", fontsize=8,
                             color=informe.TINTA, va="center")
            eje.set_xscale("log")
            xs = [p[0] for p in puntos] or [1]
            eje.set_xlim(min(xs) / 1.8, max(xs) * 4)
            eje.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            eje.xaxis.set_major_formatter(entero)
            eje.xaxis.set_minor_formatter(NullFormatter())
            eje.set_ylim(0, 105)
            eje.grid(color=informe.REJILLA, linewidth=0.8, zorder=0)
            eje.set_axisbelow(True)
            for lado in ("top", "right"):
                eje.spines[lado].set_visible(False)
            for lado in ("left", "bottom"):
                eje.spines[lado].set_color(informe.TINTA_2)
            eje.tick_params(colors=informe.TINTA_2, length=0, labelsize=8)
            eje.set_xlabel(etiqueta, color=informe.TINTA_2, fontsize=9)
            if fila_ej == 0:
                eje.set_title(alg, color=informe.TINTA, fontsize=11, loc="left")
        ejes[fila_ej][0].set_ylabel("% del pico", color=informe.TINTA_2)
    presentes = [d for d in DSLS if any(f["dsl"] == d and f.get("pct_pico") is not None for f in filas)]
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=color[d], label=d) for d in presentes],
               frameon=False, ncol=len(presentes), loc="upper left", bbox_to_anchor=(0.01, 0.95), fontsize=9,
               labelcolor=informe.TINTA)
    fig.suptitle("Productividad frente a rendimiento por DSL (GB10)", color=informe.TINTA, fontsize=12,
                 x=0.02, ha="left")
    fig.text(0.02, 0.012, "% del pico: de mma.sync f16→f32 (" + f"{pico:g}" + " TFLOP/s) en matmul y atención; de "
             "memoria (273 GB/s) en RMSNorm. " + nota, fontsize=7, color=informe.TINTA_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92), h_pad=2.0)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(destino, f"productividad.{ext}"), dpi=200, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frio", default="hennessy-frio", help="entorno de las ejecuciones con TFM_CACHE_FRIA=1")
    p.add_argument("--pico", default="hennessy", help="entorno con run_pico_mma (pico de mma.sync)")
    args = p.parse_args()

    filas, pico = recoger(args.frio, args.pico)
    with open(os.path.join(RAIZ, "results", "productividad.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in filas for k in r)))
        w.writeheader()
        w.writerows(filas)

    t = tabla(filas)
    jobs = sorted({str(f["job"]) for f in filas if f.get("job")})
    pie = (f"Entorno en frío `{args.frio}` (jobs {', '.join(jobs) or '—'}); pico de mma.sync f16→f32 = {pico} "
           f"TFLOP/s (`{args.pico}`/run_pico_mma). LoC = líneas lógicas sin blancos, comentarios ni docstrings.")
    meta = {"contexto": {}}
    md = informe.tabla_md(t, meta, list(t[0])).rsplit("\n\n*", 1)[0] + f"\n\n*{pie}*\n"
    esc = pie.replace("`", "").replace("%", r"\%").replace("_", r"\_").replace("³", r"$^3$").replace("→", r"$\to$")
    tex = re.sub(r"  \\caption\{.*\}\n", lambda _: "  \\caption{Productividad frente a rendimiento. " + esc + "}\n",
                 informe.tabla_tex(t, meta, list(t[0]), "productividad-completa"))
    docs = os.path.join(informe.DOCS, "TFM", "resultados")
    os.makedirs(os.path.join(docs, "figuras"), exist_ok=True)
    grafica(filas, os.path.join(docs, "figuras"), pico,
            f"Una forma por algoritmo: matmul 8192³, FlashAttention B=4 H=32 N=4096 D=128, RMSNorm 16384×8192; "
            f"fp16; GB10. Jobs {', '.join(jobs) or '—'}.")
    with open(os.path.join(docs, "productividad.md"), "w") as f:
        f.write("# Productividad frente a rendimiento (GB10)\n\nGenerado por `benchmarks/productividad.py`. "
                "Análisis en [../productividad.md](../productividad.md).\n\n"
                "![productividad](figuras/productividad.png)\n\n" + md)
    with open(os.path.join(docs, "productividad.tex"), "w") as f:
        f.write(tex)
    for r in t:
        print(" | ".join(str(v) for v in r.values()))
    print("\nresults/productividad.csv y docs/TFM/resultados/productividad.{md,tex} (+ figuras/productividad.png)")


if __name__ == "__main__":
    main()
