"""Vista unica para sacar conclusiones: cada algoritmo y DSL con las dos arquitecturas juntas.

Reune en una tabla por algoritmo lo que el resto de informes reparte en varios ficheros:
    rendimiento y % del pico     resumen.py   (ultima ejecucion, mayor tamano)
    error numerico               resultados.csv (error_rel) o, si el benchmark no lo guarda, la
                                 validacion que paso (assert_close de validation.py)
    metricas de Nsight Compute   ncu.py       (entornos <nodo>-ncu): SM %, tensor %, memoria %,
                                 ocupacion lograda, recurso que la limita y espera principal
    eficiencia energetica        energia.py   (entornos <nodo>-energia)
    productividad                results/productividad.csv (LoC y 1a llamada en frio, GB10)
Filas = DSL, una columna por arquitectura y la razon entre ellas. Encima de cada tabla, una
lectura automatica: mejor DSL por GPU, distancia a la referencia y, con ncu, que limita.

Salida: results/conclusiones.csv (una fila por algoritmo, DSL y arquitectura) y
docs/TFM/resultados/conclusiones.md. Lo regeneran comun.guardar() tras cada ejecucion y
ncu.py --fusionar al unir las metricas de ncu; a mano (login, sin GPU):
    python3.11 benchmarks/conclusiones.py
"""
import csv
import os
import re

import energia
import informe
import ncu
import resumen

RAIZ = informe.RAIZ

# Validacion de los benchmarks que no guardan error_rel (abortan si no pasa: si hay dato, paso).
_VALIDACION = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation.py")).read()


def _tol(nombre):
    m = re.search(rf"{nombre} = \{{\"atol\": ([\d.e-]+), \"rtol\": ([\d.e-]+)\}}", _VALIDACION)
    return f"atol {m.group(1)}, rtol {m.group(2)}" if m else "tolerancia de validation.py"


VALIDADO = {"Matmul": f"validado frente a cuBLAS ({_tol('TOL_MATMUL')})",
            "RMSNorm": f"validado frente a F.rms_norm fp32 ({_tol('TOL_RMSNORM')})"}


def corta(arq):
    """'NVIDIA GeForce RTX 5090 (sm_120)' -> 'RTX 5090'."""
    return arq.split(" (sm_")[0].replace("NVIDIA ", "").replace("GeForce ", "")


def _orden_arq(arq):
    return ("GB10" not in arq, arq)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def error_de(c):
    """Texto del error numerico de una celda de rendimiento."""
    e = _f(c["datos"].get("error_rel"))
    if e is not None:
        return f"error {e:.2%}" if e >= 1e-3 else f"error {e:.1e}"
    if c["fila"] == "Referencia":
        return "es la referencia"
    return VALIDADO.get(c["columna"].split()[0], "")


def productividad():
    """{(algoritmo, DSL): fila de results/productividad.csv} (GB10, caché fría)."""
    ruta = os.path.join(RAIZ, "results", "productividad.csv")
    if not os.path.isfile(ruta):
        return {}
    with open(ruta) as fh:
        return {(f["algoritmo"], f["dsl"]): f for f in csv.DictReader(fh)}


def cuello(d):
    """Lectura de las metricas ncu (prefijo ncu_): que limita al kernel. Heuristica de Speed of
    Light: >= 60 % en un pipe es estar cerca de su techo; si nada lo esta, manda la latencia."""
    sm, mem, ten = (_f(d.get(f"ncu_{k}")) for k in ("sm_pct", "mem_pct", "tensor_pct"))
    if sm is None and mem is None:
        return None
    if ten is not None and ten >= 60:
        return f"tensor cores ({ten:.0f} % de ciclos)"
    if mem is not None and mem >= 60 and mem >= (sm or 0):
        return f"memoria ({mem:.0f} % SoL)"
    if sm is not None and sm >= 60:
        return f"cómputo ({sm:.0f} % SoL)"
    espera = str(d.get("ncu_stall_principal") or "").replace("_", " ")
    return "latencia" + (f" (espera: {espera})" if espera else "")


def recoger():
    """{(algoritmo, DSL, arquitectura): dict} con todos los datos de esa celda."""
    rend = {(c["columna"], c["fila"], c["arquitectura"]): c for c in resumen.recoger()}
    perf = {(col, fila, arq): c for (arq, fila, col), c in ncu.celdas_ncu().items()}
    ener = {(c["columna"], c["fila"], c["arquitectura"]): c for c in energia.recoger()}
    prod = productividad()
    arqs = sorted({k[2] for k in list(rend) + list(perf) + list(ener)}, key=_orden_arq)
    out = {}
    for col, _ in resumen.COLUMNAS:
        for fila in resumen.FILAS:
            for arq in arqs:
                c, n, e = rend.get((col, fila, arq)), perf.get((col, fila, arq)), ener.get((col, fila, arq))
                x = {"algoritmo": col, "dsl": fila, "arquitectura": arq, "tipo": resumen.TIPO[col]}
                if c:
                    x.update(valor=c["valor"], unidad=c["unidad"], pct_pico=c.get("pct"),
                             ref_pico=c.get("ref_pico", "pico de memoria" if c.get("pct") is not None else ""),
                             error=error_de(c), tamano=c["tamano"], config=c["config"],
                             origen=f"{c['maquina']}, {resumen._job(c)}", extra=c["extra"])
                else:
                    x["motivo"] = resumen.motivo(arq, fila, col)
                if n:
                    d = n["datos"]
                    x.update({k: d.get(f"ncu_{k}") for k, _ in ncu.COLUMNAS_DOC if k != "mem_principal_mb"})
                    x["ncu_tamano"], x["cuello"] = n["tamano"], cuello(d)
                if e:
                    x.update(eficiencia=e["eficiencia"], unidad_eficiencia=e["unidad"], eficiencia_vs_ref=e["vs_ref"])
                p = prod.get((col, fila)) if "GB10" in arq else None
                if p:
                    x.update(loc_total=p["loc_total"], t_frio_s=p["t_frio_s"], ajuste=p["ajuste"])
                if c or n or e:
                    out[(col, fila, arq)] = x
                elif x["motivo"][0] != "no aplica":
                    out[(col, fila, arq)] = x
    return out, arqs


# --- Informe -----------------------------------------------------------------------------------

def _celda(x):
    if x is None:
        return "n/a"
    if "valor" not in x:
        clase, texto = x["motivo"]
        return f"— *{clase}*: {texto}" if clase != "no aplica" else "n/a"
    pct = f" · **{x['pct_pico']:.0f} %**" if x.get("pct_pico") is not None else ""
    lineas = [f"**{x['valor']:g} {x['unidad']}**{pct}" + (f" ({x['extra']})" if x.get("extra") else "")]
    if x.get("error"):
        lineas.append(x["error"])
    if x.get("sm_pct") not in (None, ""):
        partes = [f"SM {x['sm_pct']:g} %", f"tensor {x['tensor_pct']:g} %" if x.get("tensor_pct") not in (None, "") else None,
                  f"mem {x['mem_pct']:g} %" if x.get("mem_pct") not in (None, "") else None,
                  f"ocup. {x['ocupacion_pct']:g} %" if x.get("ocupacion_pct") not in (None, "") else None]
        lineas.append("ncu: " + " · ".join(p for p in partes if p))
        lineas.append(f"limita: **{x['cuello']}**" + (f"; ocupación por {x['limite_ocupacion']}"
                                                      if x.get("limite_ocupacion") else ""))
    else:
        lineas.append("ncu: *sin perfil*")
    if x.get("eficiencia") is not None:
        rel = f" (×{x['eficiencia_vs_ref']:g} ref.)" if x.get("eficiencia_vs_ref") else ""
        lineas.append(f"{x['eficiencia']:g} {x['unidad_eficiencia']}{rel}")
    lineas.append(f"_{x['origen']}_")
    return "<br>".join(lineas)


def _razon(a, b):
    if a and b and a.get("valor") and b.get("valor"):
        return f"×{b['valor'] / a['valor']:.2f}"
    return "—"


def _prod_txt(x):
    if not x or not x.get("loc_total"):
        return "—"
    t = _f(x.get("t_frio_s"))
    t = "" if t is None else (f" · {t:.1f} s" if t < 120 else f" · {t / 60:.1f} min")
    return f"{x['loc_total']} LoC{t}<br>{x['ajuste']}"


def lectura(col, filas, celdas, arqs):
    """Frases de conclusion de un algoritmo, solo con lo que hay medido."""
    out = []
    for arq in arqs:
        med = [celdas[(col, f, arq)] for f in filas if (col, f, arq) in celdas and "valor" in celdas[(col, f, arq)]]
        dsls = [x for x in med if x["dsl"] != "Referencia" and x.get("pct_pico") is not None]
        if not dsls:
            continue
        mejor = max(dsls, key=lambda x: x["valor"])
        peor = min(dsls, key=lambda x: x["valor"])
        ref = next((x for x in med if x["dsl"] == "Referencia"), None)
        frase = (f"**{corta(arq)}**: mejor DSL **{mejor['dsl']}** ({mejor['valor']:g} {mejor['unidad']}, "
                 f"{mejor['pct_pico']:.0f} % del {mejor['ref_pico']})")
        if ref:
            frase += f", {mejor['valor'] / ref['valor']:.0%} de la referencia ({ref['pct_pico']:.0f} %)"
        if peor is not mejor:
            frase += f"; el más lento, {peor['dsl']} ({peor['pct_pico']:.0f} %)"
        if mejor.get("cuello"):
            frase += f"; según ncu, a {mejor['dsl']} le limita {mejor['cuello']}"
        out.append(frase + ".")
    if len(arqs) == 2:
        rz = [(f, celdas[(col, f, arqs[1])]["valor"] / celdas[(col, f, arqs[0])]["valor"]) for f in filas
              if all("valor" in celdas.get((col, f, a), {}) for a in arqs)]
        if rz:
            lo, hi = min(rz, key=lambda r: r[1]), max(rz, key=lambda r: r[1])
            out.append(f"{corta(arqs[1])}/{corta(arqs[0])}: de ×{lo[1]:.2f} ({lo[0]}) a ×{hi[1]:.2f} ({hi[0]}).")
    return out


def tabla_md(col, celdas, arqs):
    filas = [f for f in resumen.FILAS if any((col, f, a) in celdas for a in arqs)]
    if not filas:
        return ""
    cab = ["DSL"] + [corta(a) for a in arqs] + ([f"{corta(arqs[1])}/{corta(arqs[0])}"] if len(arqs) == 2 else []) \
        + ["productividad (GB10)"]
    lineas = ["| " + " | ".join(cab) + " |", "|" + "|".join([":---"] * len(cab)) + "|"]
    for f in filas:
        xs = [celdas.get((col, f, a)) for a in arqs]
        fila = [f"**{f}**"] + [_celda(x) for x in xs]
        if len(arqs) == 2:
            fila.append(_razon(*xs))
        fila.append(_prod_txt(next((x for x in xs if x and x.get("loc_total")), None)))
        lineas.append("| " + " | ".join(fila) + " |")
    tam = "; ".join(f"{corta(a)} {x['tamano']}" for a in arqs
                    for x in [next((celdas[(col, f, a)] for f in filas if "tamano" in celdas.get((col, f, a), {})), None)] if x)
    lect = lectura(col, filas, celdas, arqs)
    return (f"## {col} ({resumen.TIPO[col]})\n\n" + "".join(f"- {l}\n" for l in lect) + ("\n" if lect else "")
            + "\n".join(lineas) + (f"\n\n*Tamaño: {tam}.*" if tam else "") + "\n")


NOTA = ("*Generado por `benchmarks/conclusiones.py`: una sola vista con lo que reparten "
        "[resumen.md](resumen.md), [ncu.md](ncu.md), [energia.md](energia.md) y "
        "[productividad.md](productividad.md). Cada celda: **rendimiento · % del pico** (mayor tamaño "
        "medido) · error numérico (o la validación que pasó, si el benchmark no guarda el error) · "
        "métricas de Nsight Compute (SM y memoria = *Speed of Light*; tensor = ciclos con el pipe de "
        "tensor ocupado; ocup. = ocupación lograda) y **qué limita** al kernel (≥ 60 % en un pipe = cerca "
        "de su techo; si ninguno, latencia y su espera principal) · eficiencia energética · entorno y job. "
        "Productividad: líneas de código y 1ª llamada en frío (compilación + autotuning) en GB10.*\n\n")


def generar():
    celdas, arqs = recoger()
    if not celdas:
        return None
    campos = list(dict.fromkeys(k for x in celdas.values() for k in x if k != "motivo")) + ["motivo"]
    destino_csv = os.path.join(RAIZ, "results", "conclusiones.csv")
    with open(destino_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows({**x, "motivo": ": ".join(x["motivo"]) if "motivo" in x else ""} for x in celdas.values())
    partes = ["# Conclusiones por kernel y DSL: " + " frente a ".join(corta(a) for a in arqs) + "\n\n", NOTA,
              resumen.NOTA_PICOS]
    if not any(x.get("sm_pct") not in (None, "") for x in celdas.values()):
        partes.append("> **Faltan las métricas de Nsight Compute**: aún no hay ejecuciones con `TFM_NCU=1` "
                      "(`TFM_NCU=1 bash ejecutar.sh encolar todo`). Al terminar, este documento se regenera solo.\n\n")
    partes += [tabla_md(col, celdas, arqs) + "\n" for col, _ in resumen.COLUMNAS]
    # Las tablas (tabla_md) van en el documento unico, docs/TFM/comparativa.md (benchmarks/comparativa.py).
    print(f"Conclusiones en {os.path.relpath(destino_csv, RAIZ)}", flush=True)
    return destino_csv


if __name__ == "__main__":
    generar()
