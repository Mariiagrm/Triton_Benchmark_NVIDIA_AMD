"""Migra resultados del formato antiguo (results/<experimento>/...) al separado por maquina.

    results/<experimento>/<ejecucion>/          -> results/<maquina>/<experimento>/<ejecucion>/
    docs/TFM/resultados/<experimento>.{md,tex}  -> docs/TFM/resultados/<maquina>/...
    docs/TFM/resultados/figuras/<experimento>.* -> docs/TFM/resultados/<maquina>/figuras/...

La maquina de cada ejecucion es la de su meta.json ('maquina' o, en las antiguas, 'host').
Se rehace el enlace 'ultimo' de cada <maquina>/<experimento> y se regeneran las tablas
consolidadas. Idempotente: si no queda nada en el formato antiguo, no hace nada.

Uso (en el login o en el nodo, sin GPU; despues revisar con git status y hacer commit):
    python3.11 benchmarks/migrar_resultados.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import informe  # noqa: E402

RAIZ = informe.RAIZ
RESULTS = os.path.join(RAIZ, "results")
DOCS = os.path.join(informe.DOCS, "TFM", "resultados")


def es_ejecucion(ruta):
    return os.path.isdir(ruta) and not os.path.islink(ruta) and os.path.isfile(os.path.join(ruta, "meta.json"))


# Entornos historicos: ejecuciones anteriores a que meta.json guardara 'maquina' y el driver.
# hennessy uso el driver 580.159.03 hasta el 30-09-2026 16:39 (historial de apt) y no hubo
# ejecuciones entre ese cambio y la primera con el driver registrado (02-10-2026): todas las
# de hennessy sin 'driver_nvidia' son de ese entorno. Ver docs/TFM/plataforma.md.
ENTORNO_SIN_DRIVER = {"hennessy": "hennessy-580"}


def maquina_de_meta(ruta):
    with open(os.path.join(ruta, "meta.json")) as f:
        c = json.load(f).get("contexto", {})
    if c.get("maquina"):
        return c["maquina"]
    host = c.get("host") or "desconocida"
    if not c.get("driver_nvidia") and host in ENTORNO_SIN_DRIVER:
        return ENTORNO_SIN_DRIVER[host]
    return host


def migrar_results():
    """Mueve cada ejecucion a results/<maquina>/<exp>/. Devuelve {(exp, ejecucion): maquina}."""
    destinos = {}
    for exp in sorted(os.listdir(RESULTS)):
        dir_exp = os.path.join(RESULTS, exp)
        ejecuciones = [e for e in sorted(os.listdir(dir_exp)) if es_ejecucion(os.path.join(dir_exp, e))] \
            if os.path.isdir(dir_exp) else []
        if not ejecuciones:
            continue  # ya es un directorio de maquina (o no es un experimento)
        for ej in ejecuciones:
            maquina = maquina_de_meta(os.path.join(dir_exp, ej))
            nuevo = os.path.join(RESULTS, maquina, exp, ej)
            os.makedirs(os.path.dirname(nuevo), exist_ok=True)
            os.rename(os.path.join(dir_exp, ej), nuevo)
            destinos[(exp, ej)] = maquina
            print(f"results/{exp}/{ej} -> results/{maquina}/{exp}/{ej}")
        # El 'ultimo' antiguo ya no vale: se rehace por maquina.
        ultimo = os.path.join(dir_exp, "ultimo")
        if os.path.lexists(ultimo):
            os.remove(ultimo)
        if not os.listdir(dir_exp):
            os.rmdir(dir_exp)
    # 'ultimo' de cada <maquina>/<exp> -> su ejecucion mas reciente (los nombres empiezan por la fecha).
    for maquina, exp in {(m, e) for (e, _), m in destinos.items()}:
        dir_exp = os.path.join(RESULTS, maquina, exp)
        ultima = max(e for e in os.listdir(dir_exp) if es_ejecucion(os.path.join(dir_exp, e)))
        ultimo = os.path.join(dir_exp, "ultimo")
        if os.path.lexists(ultimo):
            os.remove(ultimo)
        os.symlink(ultima, ultimo)
    return destinos


def migrar_docs():
    """Mueve las copias de docs/TFM/resultados/ a la carpeta de la maquina que las genero
    (se lee de su linea 'Generado automaticamente desde `results/...`')."""
    if not os.path.isdir(DOCS):
        return
    for md in sorted(f for f in os.listdir(DOCS) if f.endswith(".md")):
        nombre = md[:-3]
        with open(os.path.join(DOCS, md)) as f:
            texto = f.read()
        m = re.search(r"desde `results/([^`]+?)/?`", texto)
        if not m:
            continue
        partes = m.group(1).split("/")
        if len(partes) == 3:   # ya en formato nuevo: results/<maquina>/<exp>/<ejecucion>
            maquina = partes[0]
        else:                  # antiguo: results/<exp>/<ejecucion> (ya migrada en results/)
            exp, ej = partes[0], partes[1]
            ruta = next((os.path.join(RESULTS, d, exp, ej) for d in os.listdir(RESULTS)
                         if es_ejecucion(os.path.join(RESULTS, d, exp, ej))), None)
            if ruta is None:
                print(f"AVISO: no encuentro la ejecucion de {md} ({m.group(1)}); se deja donde esta")
                continue
            maquina = os.path.basename(os.path.dirname(os.path.dirname(ruta)))
            texto = texto.replace(f"results/{exp}/{ej}", f"results/{maquina}/{exp}/{ej}")
        dest = os.path.join(DOCS, maquina)
        os.makedirs(os.path.join(dest, "figuras"), exist_ok=True)
        with open(os.path.join(dest, md), "w") as f:
            f.write(texto)
        os.remove(os.path.join(DOCS, md))
        for origen, destino in [(os.path.join(DOCS, f"{nombre}.tex"), os.path.join(dest, f"{nombre}.tex"))] + \
                [(os.path.join(DOCS, "figuras", f"{nombre}.{e}"), os.path.join(dest, "figuras", f"{nombre}.{e}"))
                 for e in ("png", "pdf")]:
            if os.path.exists(origen):
                os.rename(origen, destino)
        print(f"docs/TFM/resultados/{nombre}.* -> docs/TFM/resultados/{maquina}/")
    figs = os.path.join(DOCS, "figuras")
    if os.path.isdir(figs) and not os.listdir(figs):
        os.rmdir(figs)


if __name__ == "__main__":
    migrar_results()
    migrar_docs()
    for fam in sorted({informe.familia(e) for _, e, _ in informe.ultimas_ejecuciones()} - {None}):
        informe.actualizar_metricas(fam)
