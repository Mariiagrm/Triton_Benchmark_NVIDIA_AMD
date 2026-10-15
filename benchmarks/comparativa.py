"""Documento UNICO de la comparativa: metodologia + procedencia + resultados + perfil + energia.

Une en docs/TFM/comparativa.md todo lo necesario para comparar cada kernel y cada DSL con el
mismo criterio:
    1. Alcance              algoritmos, DSLs y GPUs
    2. Metodologia          plataforma y picos, procedencia de los kernels, cargas, correccion,
                            medida, metricas y % del pico, seleccion del dato, Nsight Compute,
                            energia, productividad y reproducibilidad
    3. Resultados por algoritmo
                            procedencia + rendimiento + % del pico (kernels.py) y, si hay perfil,
                            metricas de Nsight Compute (ncu.py)
    4. Vista global         graficas por arquitectura, eficiencia energetica, productividad
    5. Fuentes

Las tablas salen de los mismos modulos que los informes parciales (kernels, ncu, energia,
resumen), asi que no hay dos versiones del mismo dato. Lo regenera resumen.generar() (y con el,
comun.guardar() tras cada ejecucion). A mano, en el login: python3.11 benchmarks/comparativa.py
"""
import os
import re

import energia
import fichas
import informe
import kernels
import ncu
import resumen

DOCS_TFM = os.path.join(informe.DOCS, "TFM")

METODOLOGIA = """\
### 2.1 Principio de la comparación

Para cada algoritmo, **todos los DSLs pasan por el mismo banco de medida**
(`benchmarks/banco_matmul.py`, `banco_attention.py`, `banco_rmsnorm.py`): mismas formas, misma
validación, misma medida y mismo guardado. Cada DSL solo aporta cómo preparar su kernel. Así, una
diferencia de rendimiento entre DSLs se debe al kernel y no a cómo se midió. En cada GPU, la
**referencia** es la implementación de PyTorch del mismo algoritmo (cuBLAS, cuBLASLt, SDPA,
`F.rms_norm`), medida en la misma ejecución.

### 2.2 Plataforma y picos de referencia

{tabla_plataforma}

Los dos chips son Blackwell GeForce: no tienen `wgmma` ni `tcgen05`, así que todos los DSLs
acaban emitiendo `mma.sync` y **el techo de los compute-bound lo pone esa instrucción**. Por eso,
en GB10 el pico de cómputo es el de `mma.sync` medido con un microbenchmark propio
(`run_pico_mma`, acumulación en FP32, § [pico_mma.md](pico_mma.md)); NVIDIA no publica pico de
fp16 ni de FP8 para GB10. En RTX 5090 se usa la ficha de NVIDIA hasta que se mida
(`run_pico_mma` en pascal). En memory-bound, el pico es el ancho de banda de la ficha. Detalle
del hardware y del software: [plataforma.md](plataforma.md).

### 2.3 Kernels y su procedencia

Cada kernel se clasifica según cuánto código propio lleva (§ [procedencia_kernels.md](procedencia_kernels.md)):
**oficial sin modificar** (el de la fuente, con solo la envoltura para medirlo), **adaptado** (parte
de un ejemplo público, con los cambios documentados) o **propio** (escrito sobre la API documentada).
Siempre que existe, se usa el kernel oficial del proyecto. Así se compara lo que cada DSL ofrece de
serie. Una celda sin kernel dice por qué: no existe en ese DSL, o el oficial no compila en sm_120/121.

### 2.4 Cargas de trabajo

| algoritmo | formas | trabajo (FLOP) / tráfico (bytes) |
|:---|:---|:---|
| Matmul fp16 y FP8 | M = N = K ∈ {{4096, 8192}} | 2·M·N·K FLOP |
| FlashAttention (forward) | B = 4, H = 32, N ∈ {{1024, 4096, 16384}}, D ∈ {{64, 128}}, causal y no causal | 4·B·H·N²·D FLOP (×0.5 si causal), como el tutorial de Triton |
| RMSNorm fp16 y bf16 | M ∈ {{4096, 16384}} filas × N ∈ {{2048, 4096, 8192}} | (2·M·N + N)·bytes por elemento (leer x y w, escribir y) |

Las tablas de resultados dan el **mayor tamaño medido** de cada celda (matmul 8192³, atención
N = 16384 con D = 128, RMSNorm 16384×8192); los informes de cada experimento tienen todos.

### 2.5 Corrección antes de medir

Ningún kernel se mide sin validarlo antes contra PyTorch (`benchmarks/validation.py`): matmul con
atol 0.1 y rtol 0.01 (acumular en otro orden que cuBLAS con K grande da ~0.1 absoluto), atención
con atol 0.01 frente a SDPA, RMSNorm con atol y rtol 0.02 frente a `F.rms_norm` en fp32. En FP8 no
se exige tolerancia: se informa del error relativo de Frobenius. En los kernels Triton se comprueba
además en el PTX que el cálculo va a **tensor cores** (`mma.sync`/`wgmma`/`tcgen05.mma`); si no,
la ejecución aborta.

### 2.6 Medida del rendimiento

- **Tiempo:** `triton.testing.do_bench`, 100 ms de calentamiento y 500 ms de repetición; se toma la
  **mediana** (y los percentiles 20 y 80). `do_bench` vacía la L2 entre repeticiones.
- **Compilación y autotuning:** fuera de la medida. La primera llamada se cronometra aparte
  (`t_compilacion_s`). Los DSLs sin autotuning (Gluon, CuTe) barren unas pocas configuraciones y
  usan la más rápida.
- **Relojes:** no se fijan (GB10 no lo permite); se **registran** el reloj SM y la potencia
  durante cada medida (NVML cada 20 ms, `reloj_mhz` y `potencia_w`).
- **Aislamiento:** cada experimento corre en un contenedor efímero con la GPU entera; la cola
  `-benchmark` de cada nodo ejecuta un trabajo cada vez. Antes de medir, `ejecutar.sh` para los
  contenedores que hayan quedado de intentos anteriores y avisa si hay otros procesos en la GPU.
  Las ejecuciones con la GPU compartida se apartan a `descartados/` con su motivo (job 21058).

### 2.7 Métricas y normalización

| tipo | métrica | % del pico |
|:---|:---|:---|
| **CB** (compute-bound: matmul, atención) | TFLOP/s = FLOP / tiempo | sobre el pico de `mma.sync` medido en esa GPU y precisión (si no hay, el de la ficha) |
| **MB** (memory-bound: RMSNorm) | GB/s = bytes / tiempo | sobre el ancho de banda de la ficha |

El % del pico es la medida principal de la comparación: dice cuánto del hardware aprovecha cada
kernel y permite comparar GPUs de tamaño muy distinto. Un % por encima de 100 de la ficha indica
que la GPU va por encima del reloj boost de la ficha o que no aplica el recorte de la acumulación
en FP32.

### 2.8 Qué dato se publica

Cada celda es la **última ejecución** de ese benchmark en esa arquitectura, en el mayor tamaño
medido (si hay dos nodos con la misma GPU, como hennessy y patterson en GB10, cuenta la más
reciente). No entran en las tablas de rendimiento los entornos especiales: `*-frio` (caché de
compilación vacía; solo si no hay otro dato), `*-energia` (régimen sostenido), `*-ncu` (bajo
Nsight Compute) ni las pruebas en dos nodos (`run_dist_matmul`). Una celda vacía dice su clase:
*no aplica* (la variante no existe en ese algoritmo, «n/a»), *sin kernel*, *no compila* o *no
medido*, con el motivo.

### 2.9 Perfil con Nsight Compute (por qué rinde lo que rinde)

Con `TFM_NCU=1`, cada benchmark se ejecuta bajo `ncu` y `comun.medir()` perfila **una llamada más**
de cada kernel guardado, ya compilado y caliente, marcada con un rango NVTX; la compilación, el
autotuning y las repeticiones de `do_bench` quedan fuera del perfil. Las métricas se unen a la fila
de su medida (`ncu_*`) y van a un entorno aparte (`<nodo>-ncu`), porque `ncu` repite cada kernel
varias veces para leer todos los contadores y su tiempo no sustituye al de `do_bench`. Se usan
las mismas métricas en las dos GPUs, salvo la memoria principal: GB10 no tiene contadores de DRAM
(memoria unificada LPDDR5X) y se mide como bytes que fallan en L2.

| métrica | contador de ncu | qué indica |
|:---|:---|:---|
| SM % | `sm__throughput` | *Speed of Light* de cómputo: uso de la unidad más cargada del SM |
| tensor % | `sm__pipe_tensor_cycles_active` | ciclos con los tensor cores ocupados: lo que separa un kernel CB bueno de uno malo |
| memoria % | `gpu__compute_memory_throughput` | *Speed of Light* de memoria (L1, L2 y DRAM) |
| emisión % | `smsp__issue_active` | ciclos en que el planificador emite una instrucción |
| ocupación % (lograda / teórica) | `sm__warps_active` / `sm__maximum_warps_per_active_cycle_pct` | warps activos frente al máximo; «limita» = recurso que la fija (registros, memoria compartida, warps, bloques) |
| regs/hilo, smem KB/bloque | `launch__registers_per_thread`, `launch__shared_mem_per_block` | recursos que pide el kernel |
| acierto L2 % | `lts__t_sector_hit_rate` | reutilización en L2 |
| memoria principal MB | `dram__bytes` (RTX 5090) / fallos de L2 (GB10) | tráfico real a memoria |
| conflictos smem | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` | conflictos de bancos de memoria compartida |
| espera principal | `smsp__average_warps_issue_stalled_*_per_issue_active` | motivo con más ciclos de espera por instrucción (long scoreboard = memoria global, barrier = sincronización, mio throttle = memoria compartida, math pipe throttle = tensor cores saturados...) |

### 2.10 Energía y productividad

- **Eficiencia energética** (`TFM_ENERGIA=1`): potencia instantánea de la GPU en régimen sostenido
  (6 s en bucle, descartando los 2 primeros) → GFLOP/J en CB y GB/J en MB, también descontando la
  potencia en reposo. Método completo: [resultados/energia.md](resultados/energia.md).
- **Productividad** (`TFM_CACHE_FRIA=1`): líneas de código lógicas del kernel, coste del ajuste
  (tamaño del espacio y primera llamada con las cachés vacías) y rendimiento de esa misma
  ejecución. Método: [productividad.md](productividad.md).

### 2.11 Reproducibilidad

Cada resultado guarda su contexto en `meta.json`: GPU, driver, kernel de Linux, imagen (con su
id), versiones de torch/triton/helion/CUDA, job de Slurm y comando. Desde el login (ibsen):

```bash
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo                 # todos los benchmarks en hennessy
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar ambos todo           # hennessy y patterson a la vez
TFM_NCU=1 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo       # con métricas de Nsight Compute
TFM_ENERGIA=1 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo   # con eficiencia energética
python3.11 ~/hennessy/tfm_entorno/benchmarks/comparativa.py          # regenerar este documento
```
"""


def tabla_plataforma():
    picos = resumen.picos_mma()
    res = kernels.leer_resultados()
    arqs = sorted({a for a, _, _ in res}, key=lambda a: ("GB10" not in a, a))
    nodos = {}
    for (a, _, _), c in res.items():
        nodos.setdefault(a, set()).add(c["maquina"].split("-")[0])
    lineas = ["| GPU | nodos | pico CB fp16 | pico CB FP8 | pico MB (memoria) |",
              "|:---|:---|:---|:---|:---|"]
    for arq in arqs:
        celdas = []
        for prec in ("fp16", "fp8"):
            medido = picos.get(arq, {}).get(prec)
            if medido:
                celdas.append(f"{medido[0]:g} TFLOP/s (mma.sync medido, {medido[1]})")
            else:
                p = fichas.pico(arq, "TFLOP/s", prec)
                celdas.append(f"{p[0]:g} TFLOP/s (ficha NVIDIA)" if p else "—")
        mem = fichas.pico(arq, "GB/s")
        lineas.append(f"| {arq} | {', '.join(sorted(nodos.get(arq, [])))} | {celdas[0]} | {celdas[1]} | "
                      f"{f'{mem[0]:g} GB/s (ficha NVIDIA)' if mem else '—'} |")
    return "\n".join(lineas)


def _figuras(arq):
    slug = resumen.slug(arq)
    out = []
    for nombre, titulo in (("resumen_compute_bound", "Compute-bound"), ("resumen_memory_bound", "Memory-bound")):
        if os.path.isfile(os.path.join(DOCS_TFM, "resultados", "figuras", f"{nombre}_{slug}.png")):
            out.append(f"![{titulo} — {arq}](resultados/figuras/{nombre}_{slug}.png)")
    return out


def _tabla_productividad():
    ruta = os.path.join(DOCS_TFM, "resultados", "productividad.md")
    if not os.path.isfile(ruta):
        return None
    texto = open(ruta).read()
    tabla = [l for l in texto.splitlines() if l.startswith("|")]
    pie = [l for l in texto.splitlines() if l.startswith("*Entorno")]
    return "\n".join(tabla) + ("\n\n" + pie[0] if pie else "")


def generar():
    proc, fuentes = kernels.leer_procedencia()
    res = kernels.leer_resultados()
    arqs = sorted({a for a, _, _ in res}, key=lambda a: ("GB10" not in a, a))
    celdas_ncu = ncu.celdas_ncu()
    try:  # vista de conclusiones (benchmarks/conclusiones.py): GB10 frente a RTX 5090, que limita, energia...
        import conclusiones
        cc, ca = conclusiones.recoger()
    except Exception as e:
        print(f"AVISO: comparativa sin conclusiones ({type(e).__name__}: {e}).", flush=True)
        conclusiones, cc, ca = None, {}, []

    p = ["# Comparativa de kernels y DSLs en Blackwell GeForce\n",
         "*Documento único generado por `benchmarks/comparativa.py` a partir de los resultados "
         "(`results/`) y de [procedencia_kernels.md](procedencia_kernels.md); se regenera tras cada "
         "ejecución. Reúne la metodología, la procedencia de cada kernel, su rendimiento y % del pico en "
         "cada GPU, su perfil de Nsight Compute, la eficiencia energética y la productividad.*\n",
         "## Índice\n\n1. [Alcance](#1-alcance)\n2. [Metodología](#2-metodología)\n"
         "3. [Resultados por algoritmo](#3-resultados-por-algoritmo) (procedencia, comparación y "
         "conclusiones, perfil de Nsight Compute)\n4. [Vista global](#4-vista-global)\n"
         "5. [Fuentes](#5-fuentes)\n",
         "## 1. Alcance\n",
         "| | |\n|:---|:---|\n"
         "| **Algoritmos** | Matmul (fp16 y FP8), FlashAttention forward (fp16, causal y no causal), "
         "RMSNorm (fp16 y bf16) |\n"
         "| **DSLs** | Triton (con sus variantes block-ptr y TMA), Triton + TLX, Gluon, Helion, CUTLASS "
         "(CuTe DSL y C++); referencia: PyTorch (cuBLAS, cuBLASLt, SDPA, `F.rms_norm`) |\n"
         f"| **GPUs** | {', '.join(arqs)} |\n",
         "## 2. Metodología\n",
         METODOLOGIA.format(tabla_plataforma=tabla_plataforma()),
         "## 3. Resultados por algoritmo\n",
         "*Cada resultado: rendimiento en el mayor tamaño medido · **% del pico** · entorno y job. Una "
         "celda sin dato dice por qué falta; n/a = la variante no existe para ese algoritmo. Configuración "
         "aplicada en cada celda (bloques, warps, etapas...): [resultados/resumen.md](resultados/resumen.md).*\n"]
    for i, (grupo, cols) in enumerate(kernels.GRUPOS, 1):
        tabla = kernels.tabla(grupo, cols, proc, res, arqs)
        p.append(re.sub(r"^## ", f"### 3.{i} ", tabla, count=1))
        if conclusiones and cc:
            p.append(f"#### Comparación y conclusiones: {grupo}\n")
            p.append("*Cada celda: rendimiento · % del pico · error numérico o validación · qué limita al "
                     "kernel según ncu · eficiencia energética · entorno y job; productividad en GB10 "
                     "(líneas de código y 1ª llamada en frío). Generado con `benchmarks/conclusiones.py`.*\n")
            for col in cols:
                t = conclusiones.tabla_md(col, cc, ca)
                if t:
                    p.append(re.sub(r"^## ", "##### ", t, count=1))
        perfiles = [t for arq in arqs for col in cols
                    if (t := ncu.tabla_columna(arq, col, celdas_ncu, nivel=f"##### {arq} —"))]
        p.append(f"#### Perfil de Nsight Compute: {grupo}\n")
        if perfiles:
            p.append(ncu.DESCRIPCION)
            p.extend(perfiles)
        else:
            p.append("*Sin perfil todavía: `TFM_NCU=1 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo` "
                     "(y lo mismo en pascal para la RTX 5090).*\n")

    p.append("## 4. Vista global\n")
    p.append("### 4.1 Rendimiento por arquitectura\n")
    p.append("*Barras: rendimiento de cada DSL; encima, % del pico; línea: pico de referencia. Los "
             "huecos llevan «✕ (n)» y su motivo debajo de la figura.*\n")
    for arq in arqs:
        p.extend(f + "\n" for f in _figuras(arq))
    p.append("### 4.2 Eficiencia energética\n")
    try:
        ce = energia.recoger()
        arqs_e = sorted({c["arquitectura"] for c in ce})
        if ce:
            p.append("*GFLOP/J (CB) y GB/J (MB) en régimen sostenido; «×» = relativo a la referencia de la "
                     "misma GPU. Método: [resultados/energia.md](resultados/energia.md).*\n")
            p.extend(energia.tabla_md(a, [c for c in ce if c["arquitectura"] == a]).replace("## ", "#### ", 1)
                     for a in arqs_e)
        else:
            p.append("*Sin medidas de energía (`TFM_ENERGIA=1`).*\n")
    except Exception as e:
        p.append(f"*No se pudo leer la eficiencia energética ({type(e).__name__}: {e}).*\n")
    p.append("### 4.3 Productividad frente a rendimiento\n")
    prod = _tabla_productividad()
    p.append((prod + "\n") if prod else "*Sin datos de productividad (`benchmarks/productividad.py`).*\n")
    p.append("*Análisis: [productividad.md](productividad.md).*\n")
    p.append("## 5. Fuentes\n")
    p.append((fuentes + "\n") if fuentes else "*Ver [procedencia_kernels.md](procedencia_kernels.md).*\n")

    destino = os.path.join(DOCS_TFM, "comparativa.md")
    with open(destino, "w") as fh:
        fh.write("\n".join(p))
    print("Comparativa unificada en docs/TFM/comparativa.md", flush=True)
    return destino


if __name__ == "__main__":
    generar()
