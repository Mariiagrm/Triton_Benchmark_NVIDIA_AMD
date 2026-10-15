"""Metricas de Nsight Compute (ncu) de cada kernel medido: POR QUE rinde lo que rinde.

Con TFM_NCU=1, ejecutar.sh lanza cada benchmark bajo ncu (benchmarks/ncu_envoltura.sh) y
comun.medir() perfila una llamada de cada kernel que se guarda (etiqueta NVTX 'tfm#<n>', columna
ncu_id). Este modulo:
    --metricas         imprime la lista de metricas que admite ESTA GPU (las de METRICAS que
                       aparecen en 'ncu --query-metrics'): GB10 no tiene dram__* (memoria
                       unificada LPDDR5X); su trafico a memoria se mide como fallos de L2
    --fusionar <csv>   une el CSV de ncu con los resultados de la ejecucion (columnas ncu_*
                       junto a cada medida, con el mismo prefijo: triton_ncu_*, cublas_ncu_*) y
                       guarda todos los kernels perfilados en ncu.csv; regenera su informe
    (sin argumentos)   docs/TFM/resultados/ncu.md: por arquitectura y algoritmo, las metricas
                       del kernel principal de cada DSL en el mayor tamano medido

Metricas (todas por kernel; el principal de cada medida es el de mayor duracion):
    Tiempo y reloj        gpu__time_duration, sm__cycles_elapsed.avg.per_second
    Speed of Light        sm__throughput (computo) y gpu__compute_memory_throughput (memoria), % del pico
    Tensor cores          sm__pipe_tensor_cycles_active (% de ciclos con el pipe de tensor ocupado)
                          y sm__inst_executed_pipe_tensor (instrucciones MMA)
    Emision               smsp__issue_active (% de ciclos en que cada SMSP emite)
    Ocupacion             lograda (sm__warps_active) y teorica (sm__maximum_warps_per_active_cycle_pct),
                          y el recurso que la limita (launch__occupancy_limit_*)
    Lanzamiento           registros/hilo, memoria compartida/bloque, bloque, rejilla, ondas por SM
    Caches                acierto en L1 y L2, bytes de L2, conflictos de bancos de memoria compartida
    Memoria principal     dram__bytes y dram__throughput (RTX 5090); en GB10, sectores que fallan en L2
    Esperas (stalls)      ciclos de espera por instruccion emitida, por motivo; 'stall_principal'

ncu se ejecuta con --clock-control none (relojes sin fijar, como el resto de medidas) y repite cada
kernel perfilado varias veces (replay) para leer todos los contadores: el tiempo de ncu no sustituye
al de do_bench. Requiere acceso a los contadores (RmProfilingAdminOnly=1 en los nodos: el contenedor
corre como root con --cap-add SYS_ADMIN).

A mano (en el login, sin GPU): python3.11 benchmarks/ncu.py
"""
import csv
import json
import os
import re
import subprocess
import sys

import informe
import resumen

# (columna corta, metrica de ncu, escala). Escala: factor sobre la unidad base de ncu (--print-units base).
METRICAS = [
    ("tiempo_us", "gpu__time_duration.sum", 1e-3),
    ("reloj_mhz", "sm__cycles_elapsed.avg.per_second", 1e-6),
    ("sm_pct", "sm__throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("mem_pct", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("tensor_pct", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 1),
    ("tensor_inst", "sm__inst_executed_pipe_tensor.sum", 1),
    ("emision_pct", "smsp__issue_active.avg.pct_of_peak_sustained_active", 1),
    ("ocupacion_pct", "sm__warps_active.avg.pct_of_peak_sustained_active", 1),
    ("l1_acierto_pct", "l1tex__t_sector_hit_rate.pct", 1),
    ("l2_acierto_pct", "lts__t_sector_hit_rate.pct", 1),
    ("l2_mb", "lts__t_bytes.sum", 1e-6),
    ("l2_fallos_mb", "lts__t_sectors_srcunit_tex_lookup_miss.sum", 32e-6),  # sectores de 32 B
    ("dram_mb", "dram__bytes.sum", 1e-6),
    ("dram_pct", "dram__throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("smem_conflictos", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", 1),
]
STALLS = ["long_scoreboard", "short_scoreboard", "barrier", "mio_throttle", "math_pipe_throttle",
          "wait", "not_selected", "lg_throttle", "dispatch_stall", "no_instruction", "membar"]
METRICAS += [(f"stall_{s}", f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio", 1) for s in STALLS]
# Siempre en la pagina 'raw' de ncu, sin pasarlas en --metrics.
LANZAMIENTO = [
    ("ocupacion_teorica_pct", "sm__maximum_warps_per_active_cycle_pct", 1),
    ("registros", "launch__registers_per_thread", 1),
    ("smem_bloque_kb", "launch__shared_mem_per_block", 1 / 1024),
    ("bloque", "launch__block_size", 1),
    ("rejilla", "launch__grid_size", 1),
    ("ondas", "launch__waves_per_multiprocessor", 1),
]
LIMITES = ["registers", "shared_mem", "warps", "blocks", "barriers"]
NOMBRE_LIMITE = {"registers": "registros", "shared_mem": "mem. compartida", "warps": "warps",
                 "blocks": "bloques", "barriers": "barreras"}


def metricas_disponibles():
    """Lista para --metrics: las de METRICAS que esta GPU admite (por nombre base)."""
    salida = subprocess.run(["ncu", "--query-metrics"], capture_output=True, text=True).stdout
    base = {linea.split()[0] for linea in salida.splitlines() if linea and not linea.startswith(("-", " "))}
    return [m for _, m, _ in METRICAS if m.split(".")[0] in base]


def _num(v):
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def leer_csv_ncu(ruta):
    """{'tfm#n': [kernel, ...]} con las columnas cortas; kernel = dict."""
    with open(ruta, errors="replace") as fh:
        lineas = [l for l in fh if l.startswith('"')]
    if not lineas:
        return {}
    filas = list(csv.reader(lineas))
    cab, datos = filas[0], [f for f in filas[1:] if f and f[0] != ""]  # la 2a fila son las unidades
    col_nvtx = next((i for i, c in enumerate(cab) if "Push/Pop_Range" in c), None)
    idx = {c: i for i, c in enumerate(cab)}
    por_id = {}
    for f in datos:
        m = re.search(r"tfm#\d+", f[col_nvtx]) if col_nvtx is not None else None
        if not m:
            continue
        k = {"kernel": f[idx["Kernel Name"]]}
        for corto, metrica, escala in METRICAS + LANZAMIENTO:
            v = _num(f[idx[metrica]]) if metrica in idx else None
            if v is not None:
                k[corto] = round(v * escala, 3)
        limites = {l: _num(f[idx[f"launch__occupancy_limit_{l}"]]) for l in LIMITES
                   if f"launch__occupancy_limit_{l}" in idx}
        limites = {l: v for l, v in limites.items() if v is not None}
        if limites:
            k["limite_ocupacion"] = NOMBRE_LIMITE[min(limites, key=limites.get)]
        stalls = {s: k[f"stall_{s}"] for s in STALLS if k.get(f"stall_{s}") is not None}
        if stalls:
            k["stall_principal"] = max(stalls, key=stalls.get)
        por_id.setdefault(m.group(0), []).append(k)
    return por_id


def principal(kernels):
    """El kernel que domina la medida (mayor duracion) y cuantos kernels lanza la llamada."""
    k = dict(max(kernels, key=lambda k: k.get("tiempo_us") or 0))
    k["kernels"] = len(kernels)
    return k


def fusionar(ruta_csv):
    por_id = leer_csv_ncu(ruta_csv)
    raiz = informe.RAIZ
    destinos = []
    for meta_json in sorted(_metas_recientes(raiz)):
        with open(meta_json) as fh:
            if json.load(fh).get("contexto", {}).get("ncu_csv") != ruta_csv:
                continue
        ejecucion = os.path.dirname(meta_json)
        with open(os.path.join(ejecucion, "resultados.csv")) as fh:
            filas = list(csv.DictReader(fh))
        todos = []
        for f in filas:
            for col in [c for c in list(f) if c.endswith("ncu_id")]:
                prefijo = col[:-len("ncu_id")]
                kernels = por_id.get(f[col])
                if not kernels:
                    continue
                for k, v in principal(kernels).items():
                    f[f"{prefijo}ncu_{k}"] = v
                todos += [{"ncu_id": f[col], "medida": prefijo.rstrip("_") or f.get("variante", ""), **k}
                          for k in kernels]
        columnas = list(dict.fromkeys(k for f in filas for k in f))
        with open(os.path.join(ejecucion, "resultados.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=columnas)
            w.writeheader()
            w.writerows(filas)
        if todos:
            cols = list(dict.fromkeys(k for t in todos for k in t))
            with open(os.path.join(ejecucion, "ncu.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(todos)
        print(f"ncu: {len(todos)} kernels perfilados unidos a {os.path.relpath(ejecucion, raiz)}/", flush=True)
        try:
            informe.generar(ejecucion)
        except Exception as e:
            print(f"AVISO: no se pudo regenerar el informe ({type(e).__name__}: {e}).", flush=True)
        destinos.append(ejecucion)
    if not destinos:
        print(f"ncu: ninguna ejecucion apunta a {ruta_csv} (¿el benchmark no llego a guardar?).", flush=True)
    else:
        generar()
        import conclusiones  # aqui: conclusiones importa este modulo
        conclusiones.generar()
    return destinos


def _metas_recientes(raiz):
    """meta.json de las ejecuciones de las ultimas 24 h (las que puede haber escrito este proceso)."""
    import time
    limite = time.time() - 86400
    base = os.path.join(raiz, "results")
    for maquina in os.listdir(base):
        dm = os.path.join(base, maquina)
        if not os.path.isdir(dm):
            continue
        for exp in os.listdir(dm):
            de = os.path.join(dm, exp)
            if not os.path.isdir(de):
                continue
            for ej in os.listdir(de):
                m = os.path.join(de, ej, "meta.json")
                if ej != "ultimo" and os.path.isfile(m) and os.path.getmtime(m) > limite:
                    yield m


# --- Resumen: docs/TFM/resultados/ncu.md ------------------------------------------------------
COLUMNAS_DOC = [
    ("kernel", "kernel"), ("tiempo_us", "µs"), ("reloj_mhz", "MHz"), ("sm_pct", "SM %"),
    ("tensor_pct", "tensor %"), ("mem_pct", "memoria %"), ("emision_pct", "emisión %"),
    ("ocupacion_pct", "ocupación %"), ("ocupacion_teorica_pct", "ocup. teórica %"),
    ("limite_ocupacion", "limita"), ("registros", "regs/hilo"), ("smem_bloque_kb", "smem KB/bloque"),
    ("l2_acierto_pct", "acierto L2 %"), ("mem_principal_mb", "memoria principal MB"),
    ("smem_conflictos", "conflictos smem"), ("stall_principal", "espera principal"),
]


def _kernel_corto(n):
    n = re.sub(r"\(.*", "", str(n))
    n = re.sub(r"^void ", "", n)
    return (n[:38] + "…") if len(n) > 39 else n


def celdas_ncu():
    """Por (arquitectura, fila, columna): datos ncu del mayor tamano, de los entornos *-ncu."""
    mejores = {}
    for maquina, exp, ultimo in informe.ultimas_ejecuciones():
        if not maquina.endswith("-ncu"):
            continue
        for c in resumen.celdas_de(maquina, exp, ultimo):
            d = c["datos"]
            if not any(k.startswith("ncu_") for k in d):
                continue
            c["arquitectura"] = resumen.arquitectura(ultimo)[0]
            clave = (c["arquitectura"], c["fila"], c["columna"])
            if clave not in mejores or c["orden"] > mejores[clave]["orden"]:
                mejores[clave] = c
    return mejores


def _valor(c, col):
    d = c["datos"]
    if col == "mem_principal_mb":
        v = d.get("ncu_dram_mb")
        if v in ("", None):
            v = d.get("ncu_l2_fallos_mb")
            return f"{v} (fallos L2)" if v not in ("", None) else "—"
        return v
    v = d.get(f"ncu_{col}")
    if col == "kernel":
        return f"`{_kernel_corto(v)}`" + (f" (+{int(float(d['ncu_kernels'])) - 1})"
                                          if d.get("ncu_kernels") not in ("", None) and float(d["ncu_kernels"]) > 1 else "")
    if col == "stall_principal" and v not in ("", None):
        return str(v).replace("_", " ")
    return "—" if v in ("", None) else v


def tabla_columna(arq, col_res, celdas, nivel="###"):
    """Tabla Markdown de las metricas ncu de un algoritmo (columna del resumen) en una arquitectura,
    o None si no hay ningun perfil de ese algoritmo en esa GPU."""
    filas = [f for f in resumen.FILAS if (arq, f, col_res) in celdas]
    if not filas:
        return None
    cab = ["DSL"] + [t for _, t in COLUMNAS_DOC]
    lineas = [f"{nivel} {col_res}\n", "| " + " | ".join(cab) + " |", "|" + "|".join([":---"] * len(cab)) + "|"]
    for fila in resumen.FILAS:
        c = celdas.get((arq, fila, col_res))
        if c is None:
            clase, texto = resumen.motivo(arq, fila, col_res)
            if clase == "no aplica":
                continue
            if clase == "no medido":
                texto = "sin perfil de ncu en esta GPU"
            lineas.append(f"| **{fila}** | — *{clase}*: {texto} |" + " |" * (len(cab) - 2))
            continue
        lineas.append(f"| **{fila}** | " + " | ".join(str(_valor(c, k)) for k, _ in COLUMNAS_DOC) + " |")
    tam = celdas[(arq, filas[0], col_res)]["tamano"]
    return "\n".join(lineas) + f"\n\n*Tamaño: {tam}.*\n"


DESCRIPCION = ("*Kernel principal de cada medida (el de mayor duración; +n = otros kernels de la misma "
               "llamada) en el mayor tamaño medido. **SM %** y **memoria %**: *Speed of Light* de cómputo y "
               "de memoria; **tensor %**: ciclos con el pipe de tensor ocupado; **emisión %**: ciclos en que "
               "cada SMSP emite una instrucción; **limita**: recurso que fija la ocupación teórica; **memoria "
               "principal**: DRAM en RTX 5090; en GB10 (memoria unificada, sin contadores dram) los bytes que "
               "fallan en L2; **espera principal**: motivo con más ciclos de espera por instrucción emitida. "
               "ncu con relojes sin fijar (`--clock-control none`).*\n")


def generar():
    celdas = celdas_ncu()
    partes = ["# Métricas de Nsight Compute por kernel\n",
              "*Generado por `benchmarks/ncu.py` a partir de las ejecuciones con `TFM_NCU=1` (entornos "
              "`<nodo>-ncu`).*\n", DESCRIPCION]
    if not celdas:
        partes.append("*Aún no hay ejecuciones con TFM_NCU=1: `TFM_NCU=1 bash ejecutar.sh encolar todo`.*\n")
    arqs = sorted({a for a, _, _ in celdas}, key=lambda a: ("GB10" not in a, a))
    for arq in arqs:
        partes.append(f"## {arq}\n")
        for col_res, _ in resumen.COLUMNAS:
            t = tabla_columna(arq, col_res, celdas, nivel="###")
            if t:
                partes.append(t)
    destino = os.path.join(informe.DOCS, "TFM", "resultados", "ncu.md")
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w") as fh:
        fh.write("\n".join(partes))
    print("Métricas ncu en docs/TFM/resultados/ncu.md", flush=True)
    return destino


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--metricas":
        print(",".join(metricas_disponibles()))
    elif len(sys.argv) > 2 and sys.argv[1] == "--fusionar":
        fusionar(sys.argv[2])
    else:
        generar()
