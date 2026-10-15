"""DOCUMENTO UNICO del TFM: docs/TFM/comparativa.md.

Es el unico Markdown de la documentacion: reune la plataforma, la metodologia, la procedencia de
cada kernel, los resultados por algoritmo y DSL (rendimiento, % del pico, conclusiones, perfil de
Nsight Compute), el pico de mma.sync, las tecnicas de optimizacion del matmul (TMA, FP8, 2:4), la
eficiencia energetica, la productividad, las incidencias, las conclusiones y las fuentes.

Las tablas de resultados salen de los mismos modulos que los calculan (kernels, conclusiones, ncu,
energia, resumen) y de results/*.csv; el analisis escrito a mano esta en las constantes de este
fichero y la procedencia en benchmarks/procedencia.py. Los demas scripts solo escriben CSV y
figuras (docs/TFM/resultados/**/figuras/). Lo regenera resumen.generar() (y con el,
comun.guardar() tras cada ejecucion). A mano, en el login: python3.11 benchmarks/comparativa.py
"""
import csv
import os
import re

import energia
import fichas
import informe
import kernels
import ncu
import procedencia
import resumen

DOCS_TFM = os.path.join(informe.DOCS, "TFM")

# Enlaces a los antiguos documentos -> anclas de este documento.
ANCLAS = {"tma": "tma", "fp8": "fp8", "sparsity": "sparsity", "pico_mma": "pico-mma", "plataforma": "plataforma",
          "productividad": "productividad", "procedencia_kernels": "procedencia", "energia": "energia",
          "resumen": "resultados", "bitacora": "incidencias", "kernels": "resultados", "ncu": "resultados",
          "conclusiones": "resultados"}


def ancla(id_, titulo):
    return f'<a id="{id_}"></a>\n\n{titulo}\n'


def _enlaces_internos(texto):
    """[x](tma.md), [x](resultados/energia.md#y), [x](../pico_mma.md) -> [x](#ancla)."""
    def cambio(m):
        nombre = m.group(2)
        return f"(#{ANCLAS[nombre]})" if nombre in ANCLAS else m.group(0)
    return re.sub(r"\(((?:\.\./|[\w./-]*/)?)([\w-]+)\.md(?:#[^)]*)?\)", cambio, texto)


# ---------------------------------------------------------------------------------------------
RESUMEN_EJECUTIVO = """\
- **Qué se compara.** Tres algoritmos —matmul (fp16 y FP8), FlashAttention forward y RMSNorm—
  escritos en cinco DSLs (Triton y sus variantes, Triton + TLX, Gluon, Helion, CUTLASS/CuTe) y
  frente a PyTorch, en dos GPUs Blackwell GeForce: **GB10** (sm_121, DGX Spark) y **RTX 5090**
  (sm_120). Todos los DSLs pasan por el mismo banco de medida y se expresan como **% del pico**.
- **El techo es `mma.sync`.** Blackwell GeForce no tiene `wgmma` ni `tcgen05`: todos los DSLs acaban
  en la MMA de sm_80. En GB10 se ha medido: 122.7 TFLOP/s en FP16 y 243 en FP8 (ráfaga), 4×
  en MXFP4/NVFP4. Ningún kernel real pasa del ~77 % de ese pico: la mitad del hueco es el reloj
  que el chip sostiene bajo un GEMM y la otra mitad, alimentar la MMA.
- **Rendimiento por DSL.** Triton (y su variante TMA) da el mejor o casi el mejor resultado en los
  tres algoritmos con poco código; CUTLASS/CuTe es el más rápido en atención pero su matmul oficial
  no es portable a sm_121 (19 % del pico); Helion es el más conciso pero paga minutos de autotuning
  y queda atrás en matmul; Gluon no compensa sin autotuner. En un memory-bound (RMSNorm) todos
  rondan el 89–90 % del ancho de banda: el DSL apenas importa.
- **Técnicas.** FP8 es la única palanca que sube el techo (+62 % en Triton, +108 % con cuBLASLt;
  error 3.75 %) y casi duplica los GFLOP/J. TMA se activa pero apenas acelera (+2–5 %); la
  sparsity 2:4 es una regresión (kernel de Ampère).
"""

PLATAFORMA = """\
| | NVIDIA GB10 (DGX Spark) | NVIDIA GeForce RTX 5090 |
|:---|:---|:---|
| nodos | hennessy y patterson (mismo hardware) | pascal |
| arquitectura | Blackwell GeForce, sm_121, 48 SMs | Blackwell GeForce, sm_120, 170 SMs |
| por SM | 1 536 hilos, 65 536 registros, 100 KiB de memoria compartida (99 KiB por bloque con *opt-in*) | — |
| L2 | 24 MiB | — |
| memoria | LPDDR5X unificada CPU/GPU, 256 bits a 8 533 MT/s = **273 GB/s**, 119.7 GiB visibles | GDDR7, 512 bits a 28 Gbps = **1 792 GB/s** |
| reloj SM | máx. 3 003 MHz; bajo carga ~2.5 GHz en ráfaga, 2.42 GHz sostenido con MMA pura, 2.08–2.18 GHz con un matmul 8192³ | boost de la ficha 2 407 MHz |
| picos de la ficha | solo «1 PFLOP FP4 con sparsity», 273 GB/s y 140 W de TDP del SoC | 209.5 TFLOP/s FP16 y 419 FP8 (acumulación FP32), 1 676 FP4, 575 W |
| CPU | 20 núcleos Arm (10 Cortex-X925 + 10 Cortex-A725), aarch64 | — |

Fuentes de las fichas: *DGX Spark User Guide* («Hardware Overview») y *NVIDIA RTX Blackwell GPU
Architecture*, tabla 3 (págs. 46–47).

**Software.** Imágenes Docker (una por DSL, mismo `Dockerfile`) sobre NGC PyTorch 25.10: PyTorch
2.9.0a0, CUDA 13.0, cuDNN 9.14, Python 3.12, **Triton 3.8.0** en todas, triton-utlx 3.8.0.post1
(TLX), Helion 1.4.1.dev (commit `0f3b242`), CUTLASS v4.8.0 y CuTe DSL 4.8.0. Ubuntu 24.04.

**Drivers y entornos.** Los resultados se guardan por entorno (`results/<entorno>/`), y cada
`meta.json` registra driver y kernel de Linux:

| entorno | qué es |
|:---|:---|
| `hennessy-580` | hennessy con driver 580.159 / Linux 6.17 (29–30 sep 2026) |
| `hennessy` | hennessy con driver 610.57.04 / Linux 7.0.0-1019, desde el 2 oct |
| `patterson` | patterson; driver 580.126 hasta el 13 oct, 610.57.04 / Linux 7.0.0-1019 desde entonces |
| `pascal` | RTX 5090; driver 610.43.02 desde el 4 oct |
| `hennessy-frio`, `hennessy-energia`, `<nodo>-ncu` | modos especiales (cachés vacías, energía, Nsight Compute) que el resumen de rendimiento no usa |

Los dos nodos GB10 están unidos por un cable QSFP (ConnectX-7, RoCE 200 Gb/s) y comparten la cola
Slurm `hennessy-patterson`; la prueba en dos nodos (`run_dist_matmul`) no entra en las tablas.
"""

METODOLOGIA = """\
### Principio de la comparación

Para cada algoritmo, **todos los DSLs pasan por el mismo banco de medida**
(`benchmarks/banco_matmul.py`, `banco_attention.py`, `banco_rmsnorm.py`): mismas formas, misma
validación, misma medida y mismo guardado. Cada DSL solo aporta cómo preparar su kernel. Así, una
diferencia entre DSLs se debe al kernel y no a cómo se midió. En cada GPU, la **referencia** es
la implementación de PyTorch del mismo algoritmo (cuBLAS, cuBLASLt, SDPA, `F.rms_norm`), medida
en la misma ejecución.

### Picos de referencia

{tabla_plataforma}

En GB10 el pico de cómputo es el de `mma.sync` **medido** con un microbenchmark propio
(acumulación FP32, § [Pico de mma.sync](#pico-mma)), porque NVIDIA no publica pico de fp16 ni de
FP8 para GB10. En RTX 5090 se usa la ficha hasta medirlo. En memory-bound, el pico es el ancho de
banda de la ficha.

### Cargas de trabajo

| algoritmo | formas | trabajo (FLOP) / tráfico (bytes) |
|:---|:---|:---|
| Matmul fp16 y FP8 | M = N = K ∈ {{4096, 8192}} | 2·M·N·K FLOP |
| FlashAttention (forward) | B = 4, H = 32, N ∈ {{1024, 4096, 16384}}, D ∈ {{64, 128}}, causal y no causal | 4·B·H·N²·D FLOP (×0.5 si causal), como el tutorial de Triton |
| RMSNorm fp16 y bf16 | M ∈ {{4096, 16384}} filas × N ∈ {{2048, 4096, 8192}} | (2·M·N + N)·bytes por elemento (leer x y w, escribir y) |

Las tablas dan el **mayor tamaño medido** de cada celda (matmul 8192³, atención N = 16384 con
D = 128, RMSNorm 16384×8192); `results/<entorno>/<experimento>/` tiene todos.

### Corrección antes de medir

Ningún kernel se mide sin validarlo contra PyTorch (`benchmarks/validation.py`): matmul con atol
0.1 y rtol 0.01 (acumular en otro orden que cuBLAS con K grande da ~0.1 absoluto), atención con
atol 0.01 frente a SDPA (la referencia ingenua necesitaría 137 GB con N = 16384), RMSNorm con atol
y rtol 0.02 frente a `F.rms_norm` en fp32. En FP8 se informa del error relativo de Frobenius. Se
comprueba además que el cálculo va a **tensor cores**: en el PTX de Triton (`mma.sync`), en el
SASS de Helion (`HMMA`, que guarda solo el binario al autotunear) y en el PTX de CuTe
(`CUTE_DSL_KEEP=ptx`). Las tres emiten `mma.sync.aligned.m16n8k16...f32.f16.f16.f32`.

### Medida del rendimiento

- **Tiempo:** `triton.testing.do_bench`, 100 ms de calentamiento y 500 ms de repetición; se toma la
  **mediana** (y los percentiles 20 y 80). `do_bench` vacía la L2 entre repeticiones.
- **Compilación y autotuning:** fuera de la medida; la primera llamada se cronometra aparte. Los
  DSLs sin autotuning (Gluon, CuTe) barren unas pocas configuraciones y usan la más rápida.
- **Relojes:** no se pueden fijar en GB10; se **registran** el reloj SM y la potencia instantánea
  durante cada medida (NVML cada 20 ms). NVML solo actualiza cada ~0.5 s y el reloj tarda ~0.5 s en
  estabilizarse, así que las comparaciones corregidas por reloj se hacen en régimen sostenido
  (`run_sostenido`: 6 s seguidos, contando los 4 últimos).
- **Aislamiento:** contenedor efímero con la GPU entera, un trabajo cada vez por nodo. Antes de
  medir se paran los contenedores que hayan quedado de intentos anteriores y se avisa si hay otros
  procesos en la GPU; las ejecuciones con la GPU compartida se apartan a `descartados/`.

### Métricas y normalización

| tipo | métrica | % del pico |
|:---|:---|:---|
| **CB** (compute-bound: matmul, atención) | TFLOP/s = FLOP / tiempo | sobre el pico de `mma.sync` medido en esa GPU y precisión (si no hay, el de la ficha) |
| **MB** (memory-bound: RMSNorm) | GB/s = bytes / tiempo | sobre el ancho de banda de la ficha |

El % del pico es la medida principal de la comparación: dice cuánto del hardware aprovecha cada
kernel y permite comparar GPUs de tamaño muy distinto. Por encima del 100 % de la ficha, la GPU va
por encima del boost de la ficha o no aplica el recorte de la acumulación en FP32.

### Qué dato se publica

Cada celda es la **última ejecución** de ese benchmark en esa arquitectura, en el mayor tamaño
medido (con dos nodos de la misma GPU, como hennessy y patterson, cuenta la más reciente). No
entran los entornos especiales `*-frio` (solo si no hay otro dato), `*-energia` y `*-ncu`, ni la
prueba en dos nodos. Una celda vacía dice su clase —*no aplica* («n/a»), *sin kernel*, *no compila*
o *no medido*— y el motivo.

### Perfil con Nsight Compute

Con `TFM_NCU=1` cada benchmark corre bajo `ncu` y `comun.medir()` perfila **una llamada más** de
cada kernel guardado, ya compilado y caliente, marcada con un rango NVTX; compilación, autotuning y
repeticiones de `do_bench` quedan fuera. Las métricas se unen a la fila de su medida (`ncu_*`) en
un entorno aparte, porque `ncu` repite el kernel para leer los contadores. Relojes sin fijar
(`--clock-control none`). En GB10 no hay contadores de DRAM (memoria unificada): el tráfico a
memoria se mide como bytes que fallan en L2. Los contadores exigen administrador
(`RmProfilingAdminOnly=1`): el contenedor corre como root con `SYS_ADMIN`.

| métrica | contador de ncu | qué indica |
|:---|:---|:---|
| SM % | `sm__throughput` | *Speed of Light* de cómputo |
| tensor % | `sm__pipe_tensor_cycles_active` | ciclos con los tensor cores ocupados |
| memoria % | `gpu__compute_memory_throughput` | *Speed of Light* de memoria (L1, L2 y DRAM) |
| emisión % | `smsp__issue_active` | ciclos en que el planificador emite una instrucción |
| ocupación % (lograda / teórica) | `sm__warps_active` / `sm__maximum_warps_per_active_cycle_pct` | warps activos frente al máximo; «limita» = recurso que fija la teórica |
| regs/hilo, smem KB/bloque | `launch__registers_per_thread`, `launch__shared_mem_per_block` | recursos del kernel |
| acierto L2 % | `lts__t_sector_hit_rate` | reutilización en L2 |
| memoria principal MB | `dram__bytes` (RTX 5090) / fallos de L2 (GB10) | tráfico real a memoria |
| conflictos smem | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` | conflictos de bancos de memoria compartida |
| espera principal | `smsp__average_warps_issue_stalled_*_per_issue_active` | motivo con más ciclos de espera (long scoreboard = memoria global, barrier = sincronización, mio throttle = memoria compartida, math pipe throttle = tensor cores saturados) |
"""

PICO_MMA = """\
**Por qué.** Todo el TFM se apoya en que en Blackwell GeForce el techo lo pone `mma.sync`, pero
NVIDIA no publica su ritmo para sm_121. Se mide directamente para expresar cada kernel como % del
pico alcanzable y no como «% de cuBLAS».

**Método** (`src/Microbench/pico_mma_ext.cu`, `run_pico_mma`). Cada warp carga una vez sus
fragmentos A y B en registros (datos aleatorios) y encadena `iters × ILP` instrucciones `mma.sync`
sobre ILP acumuladores independientes, sin accesos a memoria; al final escribe un valor por hilo
para que el compilador no elimine el bucle. Un bloque por SM; se barren 1–32 warps/SM e ILP 1–8.
Cada instrucción se compila para `sm_121a` y se comprueba en el SASS que es de tensor core.

![Pico de mma.sync en GB10](resultados/hennessy/figuras/run_pico_mma.png)

| instrucción PTX | SASS | pico (ráfaga) | FLOP/ciclo/SM | relativo a f16 |
|:---|:---|---:|---:|---:|
| `m16n8k8` tf32 → f32 | `HMMA.1688.F32.TF32` | 61.4 TFLOP/s | 511 | 0.5× |
| `m16n8k16` f16 → f32 | `HMMA.16816.F32` | **122.7 TFLOP/s** | 1 014–1 024 | 1× |
| `m16n8k16` f16 → f16 / bf16 → f32 | `HMMA.16816.F16` / `.BF16` | 122.8 TFLOP/s | 1 020 | 1× |
| `m16n8k32` e4m3 → f32 | `QMMA.16832.F32.E4M3.E4M3` | **243.3 TFLOP/s** | 2 041 | 2× |
| `m16n8k32` s8 → s32 | `IMMA.16832.S8.S8` | 243.9 TOP/s | 2 031 | 2× |
| `kind::mxf8f6f4` MXFP8 | `QMMA.SF.16832...E8` | 242.9 TFLOP/s | 2 038 | 2× |
| `kind::mxf4nvf4` MXFP4 / NVFP4 | `OMMA.SF.16864...` | **487.3 TFLOP/s** | 4 089 | 4× |

- **1 024 FLOP/ciclo/SM en FP16/BF16, 2 048 en FP8/INT8, 4 096 en MXFP4/NVFP4**, sin penalización
  por acumular en FP32 (a diferencia de la ficha de la RTX 5090, que la recorta a la mitad) ni por
  el escalado por bloques. Las MMA block-scaled solo existen en `sm_121a`.
- **Saturarla exige ≥ 8 warps/SM** (≥ 2 por subpartición): un warp no emite más de una MMA cada
  ~17 ciclos, y la unidad acepta una cada 16. Con 4 warps/SM el techo es el ~94 %.
- **Validación del pico** por cuatro caminos: recálculo desde los datos crudos (12 configuraciones
  con ≥ 8 warps/SM dan 122.2–122.7), cuenta por ciclo (122.7 / (48 × 1 024) = 2.497 GHz, el reloj
  registrado), las cifras por ciclo de la ficha de la RTX 5090 (mismo SM) y la única cifra oficial
  de GB10 (1 PFLOP FP4 con sparsity ⇒ 2.54 GHz; los 122.7 son el 98 % de lo que daría FP16 a ese
  reloj). Es un pico **medido en ráfaga** (~2.5 GHz), no teórico al reloj máximo (3 003 MHz).

**Régimen sostenido** (`run_sostenido`, job 20848; 6 s seguidos, cuentan los 4 últimos), matmul 8192³:

![Rendimiento sostenido frente al pico](resultados/hennessy/figuras/run_sostenido.png)

| kernel | TFLOP/s | reloj (MHz) | potencia (W) | % del pico al mismo reloj | % del pico nominal |
|:---|---:|---:|---:|---:|---:|
| `mma.sync` f16→f32 (microbenchmark) | 118.7 | 2 424 | 100 | 99.6 % | 97 % |
| cuBLAS fp16 | 94.4 | 2 138 | 90 | **89.8 %** | 77 % |
| Triton fp16 | 90.6 | 2 177 | 90 | 84.6 % | 74 % |
| Triton + TMA fp16 | 93.8 | 2 080 | 89 | **91.7 %** | 76 % |
| cuBLASLt FP8 | 182.9 | 2 145 | 91 | 86.7 % | 75 % |
| Triton FP8 | 144.0 | 2 158 | 91 | 67.9 % | 59 % |

**Lectura.** Ningún kernel real pasa del ~77 % del pico nominal, y el hueco tiene dos partes de
tamaño parecido: **reloj** (~12–15 puntos: la MMA pura sostiene 2 424 MHz, un GEMM real baja a
2 080–2 180 MHz aunque NVML lea menos potencia; hipótesis pendiente: el presupuesto de todo el SoC,
que comparte potencia y memoria con la CPU) y **por ciclo** (~8–15 puntos: a su propio reloj los
mejores rinden el 87–92 %; se pierde alimentando la MMA: `ldmatrix`, esperas de `cp.async`/TMA,
sincronización, epílogo). FP4 block-scaled (487 TFLOP/s) es la palanca que ningún kernel usa aún.
"""

TECNICAS = """\
Punto de partida: el matmul denso fp16 en Triton se estanca en ~90–100 TFLOP/s en GB10. Se
probaron las técnicas propias de Hopper/Blackwell (datos: entorno `hennessy-580`, 8192³).

| técnica | TFLOP/s | vs FP16 | ¿sube el techo? | evidencia clave |
|:---|---:|---:|:---:|:---|
| **FP16 baseline** (Triton) | ~92 | — | referencia | `mma.sync.m16n8k16`, carga `cp.async` |
| TMA en FP16 (descriptores) | ~93 | +0–2 % | no | activa `cp.async.bulk.tensor` pero no acelera |
| **FP8 denso** (Triton) | **150** | **+62 %** | sí | `mma.sync.m16n8k32.e4m3`, error 3.75 % |
| FP8 denso (cuBLASLt) | **192** | **+108 %** | sí | `torch._scaled_mm` |
| FP8 + TMA (Triton) | 150 | +62 % | no (vs FP8) | TMA real, solo +1.5 % sobre FP8 |
| Sparsity 2:4 (cuSPARSELt) | 32 | **−65 %** | no | kernel `sm80_xmma_sparse` (Ampère) |

{ancla_tma}

- `tl.make_block_ptr` **no** activa TMA (baja a `cp.async`) y está deprecado en Triton 3.8; solo
  `tl.make_tensor_descriptor` emite `cp.async.bulk.tensor` (verificado en el PTX). Activar el
  acelerador es cuestión de sintaxis, y hay que comprobarlo en el PTX.
- Con TMA activo, el matmul fp16 gana un ~2 % (dentro del ruido): el matmul denso **no es
  memory-bound** (~34 GB/s efectivos, los datos se reutilizan en SRAM).
- En régimen sostenido TMA sí mejora la eficiencia **por ciclo** (del 84.6 % al 91.7 % del pico de
  `mma.sync`, al nivel de cuBLAS), pero su kernel sostiene un reloj algo menor (2 080 frente a
  2 177 MHz) y el neto es +3.5 %. Una primera lectura («TMA ahorra un tercio de potencia») era un
  artefacto de la potencia promedio de NVML.

![TMA en matmul fp16](resultados/hennessy-580/figuras/run_matmul_tma.png)

{ancla_fp8}

- FP8 e4m3 rompe la barrera: **150 TFLOP/s** en Triton y **192** con cuBLASLt (+57–62 % y +108 %).
  El throughput viene de la instrucción: `m16n8k32` (K doble que `m16n8k16`).
- El coste es de precisión, no de estabilidad: error relativo **3.75 %** (0.02 % en fp16), igual en
  Triton y cuBLASLt (límite del formato).
- **FP8 + TMA** solo aporta +1.5 %: FP8 tampoco es memory-bound. El hueco de Triton frente a
  cuBLASLt (~17 %) es de *scheduling* (kernels persistentes, especialización de warps), no de datos.
- En energía, cuBLASLt FP8 hace 2 037 GFLOP/J frente a 1 051 de cuBLAS fp16 con la misma potencia.
- En la RTX 5090, cuBLASLt FP8 da 599–624 TFLOP/s, por encima de los 419 de la ficha con
  acumulación FP32; el error idéntico a Triton apunta a que en sm_120 la acumulación FP32 va a
  ritmo completo (pendiente de `run_pico_mma` en pascal).
- CUTLASS no tiene un matmul FP8 comparable: el `dense_gemm` de CuTe es fp16/bf16 y su FP8 en
  Blackwell GeForce es *block-scaled* (MXFP8), otro formato.

![FP8 frente a FP16](resultados/hennessy-580/figuras/run_matmul_fp8.png)

{ancla_sparsity}

- Triton no admite 2:4 en `tl.dot`; se usó PyTorch (`to_sparse_semi_structured`, cuSPARSELt).
- El 2:4 rinde el **34–62 % del denso**: cuSPARSELt despacha a `sm80_xmma_sparse_gemm` (Ampère),
  mientras el denso usa un kernel nativo `nvjet_sm121_..._tmaAB`. FP8 + 2:4 no está soportado por
  la API. La cifra de catálogo (sparse + baja precisión) no es reproducible con este stack.
"""

INCIDENCIAS = """\
- **Blackwell GeForce no es el Blackwell de los ejemplos.** Casi todos los kernels oficiales
  optimizados usan `wgmma` (Hopper) o `tcgen05` (Blackwell de centro de datos). El ejemplo de
  atención de Gluon aborta (`LLVM ERROR: Cannot select: intrinsic tcgen05.wait.st`), los
  tutoriales de atención de TLX usan `wgmma`/`tcgen05`, y en CuTe solo sirve el FlashAttention-2
  de Ampere.
- **Versiones.** triton-utlx 3.8.0.post1 genera IR inválido en GB10 (`operandSegmentSizes`) y el
  tutorial actual de triton-ext no compila con él. Para usar código oficial sin tocarlo se fijaron
  commits exactos (Triton `c01b677`, CUTLASS `098de2a`, Helion `0f3b242`).
- **Comparación justa.** CuTe no tiene autotuning: con su configuración por defecto se quedaba en
  el ~60 % de SDPA; se barren sus parámetros públicos. El causal de Helion no se salta el triángulo
  superior (hace el trabajo del no causal). `warp_specialize` de Triton va desactivado, como en el
  propio tutorial para capability 12.
- **CUTLASS matmul en GB10.** El ejemplo oficial `Sm120GemmKernel` da el 99 % de cuBLAS en la RTX
  5090 pero el 19–24 % en GB10; ni el número de clusters ni el tile (máx. 37.8 TFLOP/s) lo
  arreglan: problema abierto de portabilidad sm_120 → sm_121.
- **Variabilidad en la RTX 5090.** En el job 20638 cuBLAS dio 168 TFLOP/s frente a 209 en el 20499
  con el mismo kernel: las comparaciones fiables son dentro del mismo job.
- **Medir reloj y potencia en GB10.** La potencia de `nvmlDeviceGetPowerUsage` va retrasada ~1 s;
  se usa la instantánea. Las cifras de reloj de los jobs 20835–20837 no sirven para corregir el pico.
- **Drivers.** hennessy pasó de 580 a 610 el 30 sep y patterson el 13 oct: los entornos separan
  esas ejecuciones.
- **GPU compartida (job 21058, 14 oct).** Tras caerse patterson, Slurm reencoló el trabajo y los
  contenedores del intento anterior siguieron usando la GPU de hennessy: el intento de las 18:00
  midió ~40 % de lo normal. Esos resultados están en `descartados/` y el ejecutor ya para los
  contenedores huérfanos.
- **Patterson inestable.** El 14 oct se colgó seis veces bajo carga de GPU (4–20 min) tras pasar
  al driver 610 / Linux 7.0; causa sin confirmar (sin acceso a los registros del sistema).
"""

CONCLUSIONES = """\
1. **En Blackwell GeForce el techo lo pone `mma.sync`**, y los mejores kernels (cuBLAS, Triton +
   TMA, cuBLASLt FP8, CuTe FlashAttention) se quedan en el 74–77 % de su pico nominal en GB10; a su
   propio reloj, en el 87–92 %.
2. **No hay un DSL dominante.** Triton es el mejor compromiso (código corto, autotuning de segundos,
   el mejor o casi el mejor rendimiento en los tres algoritmos). Helion minimiza el código pero
   traslada el esfuerzo a minutos de autotuning y en matmul no alcanza a Triton. CUTLASS/CuTe da el
   máximo control con un orden de magnitud más de código; su atención es la más rápida, pero su
   matmul no es portable a sm_121. Gluon solo compensa cuando hace falta controlar el layout.
3. **En memory-bound el DSL apenas importa**: todos al 89–90 % del ancho de banda en GB10; cambia el
   coste (de 16 líneas y 77 s de autotuning a 203 líneas y 55 s de nvcc).
4. **FP8 es la única técnica que sube el techo** y casi duplica la eficiencia energética; TMA y 2:4
   no rinden en este hardware y stack.
5. **El ejemplo oficial no es garantía de rendimiento ni de portabilidad**: la mitad de los kernels
   oficiales optimizados no corren en sm_120/121, y el que corre puede rendir 4 veces menos en una
   GPU de la misma familia.

**Líneas abiertas:** kernel FP8 persistente; matmul MXFP4/NVFP4 (`tl.dot_scaled`) frente a 487
TFLOP/s; `run_pico_mma` en la RTX 5090; perfiles de Nsight Compute en las dos GPUs para atribuir la
pérdida por ciclo; atribuir la bajada de reloj bajo GEMM (*throttle reasons*); revisar 2:4 cuando
haya kernels sparse para sm_121.
"""

REPRODUCIBILIDAD = """\
Cada resultado guarda su contexto en `meta.json`: GPU, driver, kernel de Linux, imagen (con su
id), versiones, job de Slurm y comando. Desde el login (ibsen):

```bash
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo                 # todos los benchmarks en hennessy
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar ambos todo           # hennessy y patterson a la vez
TFM_NCU=1 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo       # con métricas de Nsight Compute
TFM_ENERGIA=1 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo   # con eficiencia energética
TFM_CACHE_FRIA=1 TFM_MAQUINA=hennessy-frio bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp matmul --sizes 8192
python3.11 ~/hennessy/tfm_entorno/benchmarks/productividad.py        # LoC y tabla de productividad (sin GPU)
python3.11 ~/hennessy/tfm_entorno/benchmarks/comparativa.py          # regenerar este documento (sin GPU)
```
"""


# ---------------------------------------------------------------------------------------------
def tabla_plataforma():
    picos = resumen.picos_mma()
    res = kernels.leer_resultados()
    arqs = sorted({a for a, _, _ in res}, key=lambda a: ("GB10" not in a, a))
    lineas = ["| GPU | pico CB fp16 | pico CB FP8 | pico MB (memoria) |", "|:---|:---|:---|:---|"]
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
        lineas.append(f"| {arq} | {celdas[0]} | {celdas[1]} | {f'{mem[0]:g} GB/s (ficha NVIDIA)' if mem else '—'} |")
    return "\n".join(lineas)


def _figura(ruta_rel, titulo):
    return f"![{titulo}]({ruta_rel})\n" if os.path.isfile(os.path.join(DOCS_TFM, ruta_rel)) else ""


def notas_procedencia():
    """Texto de procedencia.TEXTO sin titulo, tablas ni fuentes: criterios y notas por kernel."""
    out, seccion = [], ""
    for linea in procedencia.TEXTO.splitlines():
        if linea.startswith("# "):
            continue
        if linea.startswith("## "):
            seccion = linea[3:]
            if seccion != "Fuentes":
                out.append(f"#### {seccion}")
            continue
        if seccion == "Fuentes" or linea.startswith("|"):
            continue
        out.append(linea)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def tabla_productividad():
    ruta = os.path.join(informe.RAIZ, "results", "productividad_tabla.csv")
    if not os.path.isfile(ruta):
        return "*Sin datos: `python3.11 benchmarks/productividad.py`.*\n"
    with open(ruta) as fh:
        filas = list(csv.reader(fh))
    lineas = ["| " + " | ".join(filas[0]) + " |", "|" + "|".join([":---"] * len(filas[0])) + "|"]
    lineas += ["| " + " | ".join(f) + " |" for f in filas[1:]]
    pie = os.path.join(informe.RAIZ, "results", "productividad_pie.txt")
    nota = f"\n\n*{open(pie).read().strip()}*" if os.path.isfile(pie) else ""
    return "\n".join(lineas) + nota + "\n"


PRODUCTIVIDAD = """\
**Qué se mide** (`benchmarks/productividad.py`, `TFM_CACHE_FRIA=1`, una forma por algoritmo en
fp16 en GB10: matmul 8192³, FlashAttention B=4 H=32 N=4096 D=128, RMSNorm 16384×8192):

1. **Líneas de código lógicas** (sin blancos, comentarios ni docstrings) del kernel, de la envoltura
   propia y del ajuste propio. Si se carga un ejemplo oficial de fuera, se cuenta ese código en la
   versión fijada, solo la parte del algoritmo.
2. **Coste del ajuste**: modo de ajuste, tamaño del espacio de configuraciones y tiempo de la primera
   llamada (compilación + autotuning + primera ejecución) con **todas las cachés vacías**.
3. **Rendimiento** de esa misma ejecución, como % del pico.

![Productividad frente a rendimiento](resultados/figuras/productividad.png)

{tabla}

**Lectura.** Helion es el más conciso (16–45 líneas, de 2 a 4 veces menos que Triton) pero busca
134–251 configuraciones y tarda de 1.3 a 4 min por forma (hasta 31 min en atención con N = 16384);
iguala al resto en RMSNorm, se acerca en atención y queda atrás en matmul. Triton es el equilibrio:
38–220 líneas, autotuning de 15–96 s y el mejor o casi el mejor rendimiento. CUTLASS/CuTe exige un
orden de magnitud más de código (670–710 líneas el kernel oficial) y su matmul oficial rinde el 19 %
en GB10. Gluon, sin autotuner, cuesta más que Triton y rinde menos en matmul. El tiempo de ajuste
es una dimensión propia: de 0.6 s a 4 min por forma con rendimientos parecidos. Limitaciones: las
LoC miden tamaño, no esfuerzo; una sola forma y una sola ejecución por algoritmo.
"""


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
         "*Documento único del TFM, generado por `benchmarks/comparativa.py` a partir de `results/` y "
         "regenerado tras cada ejecución. Tablas en CSV en `results/` y figuras en "
         "`docs/TFM/resultados/**/figuras/`.*\n",
         "## Índice\n\n1. [Resumen](#resumen)\n2. [Plataforma](#plataforma)\n3. [Metodología](#metodologia)\n"
         "4. [Procedencia de los kernels](#procedencia)\n5. [Resultados por algoritmo y DSL](#resultados)\n"
         "6. [Pico de `mma.sync`: el techo del hardware](#pico-mma)\n"
         "7. [Técnicas de optimización del matmul en GB10](#tecnicas)\n8. [Eficiencia energética](#energia)\n"
         "9. [Productividad frente a rendimiento](#productividad)\n10. [Incidencias y validez de los datos](#incidencias)\n"
         "11. [Conclusiones y líneas abiertas](#conclusiones)\n12. [Reproducibilidad](#reproducibilidad)\n"
         "13. [Fuentes](#fuentes)\n",
         ancla("resumen", "## 1. Resumen"), RESUMEN_EJECUTIVO,
         ancla("plataforma", "## 2. Plataforma"), PLATAFORMA,
         ancla("metodologia", "## 3. Metodología"), METODOLOGIA.format(tabla_plataforma=tabla_plataforma()),
         ancla("procedencia", "## 4. Procedencia de los kernels"),
         "*Tabla de procedencia de cada kernel junto a su resultado: § [5](#resultados). Aquí, el criterio "
         "y las notas de cada kernel (fuente única: `benchmarks/procedencia.py`).*\n",
         notas_procedencia(),
         ancla("resultados", "## 5. Resultados por algoritmo y DSL"),
         "*Cada resultado: rendimiento en el mayor tamaño medido · **% del pico** · entorno y job. Una "
         "celda sin dato dice por qué falta; n/a = la variante no existe para ese algoritmo. Configuración "
         "aplicada en cada celda (bloques, warps, etapas): `results/resumen.csv`.*\n"]
    for i, (grupo, cols) in enumerate(kernels.GRUPOS, 1):
        p.append(re.sub(r"^## ", f"### 5.{i} ", kernels.tabla(grupo, cols, proc, res, arqs), count=1))
        if conclusiones and cc:
            p.append(f"#### Comparación y conclusiones: {grupo}\n")
            p.append("*Cada celda: rendimiento · % del pico · error numérico o validación · qué limita al "
                     "kernel según ncu · eficiencia energética · entorno y job; productividad en GB10.*\n")
            for col in cols:
                t = conclusiones.tabla_md(col, cc, ca)
                if t:
                    p.append(re.sub(r"^## ", "##### ", t, count=1))
        p.append(f"#### Perfil de Nsight Compute: {grupo}\n")
        perfiles = [t for arq in arqs for col in cols
                    if (t := ncu.tabla_columna(arq, col, celdas_ncu, nivel=f"##### {arq} —"))]
        if perfiles:
            p.append(ncu.DESCRIPCION)
            p.extend(perfiles)
        else:
            p.append("*Sin perfil todavía: `TFM_NCU=1 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar todo` "
                     "(y lo mismo en pascal para la RTX 5090).*\n")
    p.append("### 5.4 Vista por arquitectura\n")
    p.append("*Barras: rendimiento de cada DSL; encima, % del pico; línea: pico de referencia; los huecos "
             "llevan «✕ (n)» y su motivo debajo.*\n")
    for arq in arqs:
        s = resumen.slug(arq)
        p.append(_figura(f"resultados/figuras/resumen_compute_bound_{s}.png", f"Compute-bound — {arq}"))
        p.append(_figura(f"resultados/figuras/resumen_memory_bound_{s}.png", f"Memory-bound — {arq}"))

    p += [ancla("pico-mma", "## 6. Pico de `mma.sync`: el techo del hardware"), PICO_MMA,
          ancla("tecnicas", "## 7. Técnicas de optimización del matmul en GB10"),
          TECNICAS.format(ancla_tma=ancla("tma", "### 7.1 TMA: se activa pero no acelera"),
                          ancla_fp8=ancla("fp8", "### 7.2 FP8: la palanca que rompe el techo"),
                          ancla_sparsity=ancla("sparsity", "### 7.3 Sparsity 2:4: regresión en este stack")),
          ancla("energia", "## 8. Eficiencia energética")]
    p.append(re.sub(r"^## Cómo se calcula", "### Cómo se calcula", energia.METODO_MD.strip(), count=1) + "\n")
    try:
        ce = energia.recoger()
        for a in sorted({c["arquitectura"] for c in ce}):
            p.append(energia.tabla_md(a, [c for c in ce if c["arquitectura"] == a]).replace("## ", "### ", 1))
            s = resumen.slug(a)
            p.append(_figura(f"resultados/figuras/energia_compute_bound_{s}.png", f"Energía compute-bound — {a}"))
            p.append(_figura(f"resultados/figuras/energia_memory_bound_{s}.png", f"Energía memory-bound — {a}"))
        if not ce:
            p.append("*Sin medidas de energía (`TFM_ENERGIA=1`).*\n")
    except Exception as e:
        p.append(f"*No se pudo leer la eficiencia energética ({type(e).__name__}: {e}).*\n")
    p += [ancla("productividad", "## 9. Productividad frente a rendimiento"),
          PRODUCTIVIDAD.format(tabla=tabla_productividad()),
          ancla("incidencias", "## 10. Incidencias y validez de los datos"), INCIDENCIAS,
          ancla("conclusiones", "## 11. Conclusiones y líneas abiertas"), CONCLUSIONES,
          ancla("reproducibilidad", "## 12. Reproducibilidad"), REPRODUCIBILIDAD,
          ancla("fuentes", "## 13. Fuentes"),
          (fuentes + "\n") if fuentes else ""]

    texto = _enlaces_internos("\n".join(x for x in p if x))
    destino = os.path.join(DOCS_TFM, "comparativa.md")
    with open(destino, "w") as fh:
        fh.write(texto)
    print("Documento unico en docs/TFM/comparativa.md", flush=True)
    return destino


if __name__ == "__main__":
    generar()
