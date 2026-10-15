# Comparativa de kernels y DSLs en Blackwell GeForce

*Documento único del TFM, generado por `benchmarks/comparativa.py` a partir de `results/` y regenerado tras cada ejecución. Tablas en CSV en `results/` y figuras en `docs/TFM/resultados/**/figuras/`.*

## Índice

1. [Resumen](#resumen)
2. [Plataforma](#plataforma)
3. [Metodología](#metodologia)
4. [Procedencia de los kernels](#procedencia)
5. [Resultados por algoritmo y DSL](#resultados)
6. [Pico de `mma.sync`: el techo del hardware](#pico-mma)
7. [Técnicas de optimización del matmul en GB10](#tecnicas)
8. [Eficiencia energética](#energia)
9. [Productividad frente a rendimiento](#productividad)
10. [Incidencias y validez de los datos](#incidencias)
11. [Conclusiones y líneas abiertas](#conclusiones)
12. [Reproducibilidad](#reproducibilidad)
13. [Fuentes](#fuentes)

<a id="resumen"></a>

## 1. Resumen

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

<a id="plataforma"></a>

## 2. Plataforma

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

<a id="metodologia"></a>

## 3. Metodología

### Principio de la comparación

Para cada algoritmo, **todos los DSLs pasan por el mismo banco de medida**
(`benchmarks/banco_matmul.py`, `banco_attention.py`, `banco_rmsnorm.py`): mismas formas, misma
validación, misma medida y mismo guardado. Cada DSL solo aporta cómo preparar su kernel. Así, una
diferencia entre DSLs se debe al kernel y no a cómo se midió. En cada GPU, la **referencia** es
la implementación de PyTorch del mismo algoritmo (cuBLAS, cuBLASLt, SDPA, `F.rms_norm`), medida
en la misma ejecución.

### Picos de referencia

| GPU | pico CB fp16 | pico CB FP8 | pico MB (memoria) |
|:---|:---|:---|:---|
| NVIDIA GB10 (sm_121) | 122.83 TFLOP/s (mma.sync medido, hennessy, job 21058) | 243.31 TFLOP/s (mma.sync medido, hennessy, job 21058) | 273 GB/s (ficha NVIDIA) |
| NVIDIA GeForce RTX 5090 (sm_120) | 209.5 TFLOP/s (ficha NVIDIA) | 419 TFLOP/s (ficha NVIDIA) | 1792 GB/s (ficha NVIDIA) |

En GB10 el pico de cómputo es el de `mma.sync` **medido** con un microbenchmark propio
(acumulación FP32, § [Pico de mma.sync](#pico-mma)), porque NVIDIA no publica pico de fp16 ni de
FP8 para GB10. En RTX 5090 se usa la ficha hasta medirlo. En memory-bound, el pico es el ancho de
banda de la ficha.

### Cargas de trabajo

| algoritmo | formas | trabajo (FLOP) / tráfico (bytes) |
|:---|:---|:---|
| Matmul fp16 y FP8 | M = N = K ∈ {4096, 8192} | 2·M·N·K FLOP |
| FlashAttention (forward) | B = 4, H = 32, N ∈ {1024, 4096, 16384}, D ∈ {64, 128}, causal y no causal | 4·B·H·N²·D FLOP (×0.5 si causal), como el tutorial de Triton |
| RMSNorm fp16 y bf16 | M ∈ {4096, 16384} filas × N ∈ {2048, 4096, 8192} | (2·M·N + N)·bytes por elemento (leer x y w, escribir y) |

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

<a id="procedencia"></a>

## 4. Procedencia de los kernels

*Tabla de procedencia de cada kernel junto a su resultado: § [5](#resultados). Aquí, el criterio y las notas de cada kernel (fuente única: `benchmarks/procedencia.py`).*

*Origen de cada kernel del TFM: si es código oficial sin modificar, una adaptación de un ejemplo público o una implementación propia sobre una API documentada. La procedencia se ha verificado comparando el código con la fuente citada, en la versión indicada.*

**Tipo de elaboración:**

- **Oficial sin modificar**: el algoritmo es el de la fuente, que se llama tal cual; solo se añade la envoltura necesaria para medirlo.
- **Adaptado**: parte de un ejemplo público, con cambios que se detallan.
- **Propio**: diseño propio, escrito a partir de la documentación y los ejemplos de la API.

#### Kernels compute-bound (`src/CBKernels/`)

**`matmul.py`.** 27 de las 36 líneas distintas del kernel son idénticas a `matmul_kernel` del tutorial 03. Coinciden el orden de programas agrupado para la L2 (`GROUP_SIZE_M`), la aritmética de punteros y el bucle en K con `tl.dot` y acumulación en fp32. Es propio el espacio de autotuning (producto cartesiano de BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps y num_stages). También lo es la comprobación en el PTX de que se usan tensor cores (`comun.exigir_tensor_cores`).

**`matmul_blockptr.py`.** Es el mismo kernel con las cargas expresadas con *block pointers*. Se conserva como intento documentado: en sm_121 no activa TMA, sino `cp.async` clásico (§ [TMA](#tma)).

**`matmul_tma.py`.** Toma del tutorial 09 los descriptores de tensor creados dentro del kernel (`tl.make_tensor_descriptor`), las cargas y escrituras `desc.load` / `desc.store` y el *allocator* obligatorio (`triton.set_allocator`). A diferencia del tutorial, **no es persistente**: mantiene la rejilla y el orden agrupado del baseline para aislar el efecto de TMA.

**`matmul_fp8.py`.** Mismo esqueleto que el baseline, con A y B en FP8 e4m3 (`torch.float8_e4m3fn`). El tutorial 03 usa e5m2 como ejemplo.

**`matmul_fp8_tma.py`.** Une las dos variantes anteriores y amplía el espacio de autotuning (num_stages 3–5, BLOCK_K hasta 256).

**`gluon/matmul.py`.** Gluon no tiene un matmul oficial para sm_120/121. Sus tutoriales de matmul usan `wgmma` (Hopper) o `tcgen05` (Blackwell de centro de datos), dos instrucciones que no existen en esta arquitectura, que usa `mma.sync`. El kernel es **propio**, montado con piezas oficiales que sí funcionan aquí:
- **MMA de tensor core:** `mma_v2` con `NVMMADistributedLayout(version=[2,0])` y `DotOperandLayout`, como en `test_core.py::test_mma_v2`.
- **Pipeline de copias:** un anillo de `STAGES` buffers en memoria compartida que se llena con `cp.async_load` y `commit_group` / `wait_group`, como en `03-async-copy.py`.
- **Esqueleto:** el orden de programas agrupado para la L2 del baseline Triton.

Como Gluon no tiene autotuning, el banco mide unas pocas configuraciones de bloque, etapas y warps y usa la más rápida.

**`helion/matmul.py`.** Es el `matmul` del ejemplo oficial con el mismo decorador (`static_shapes=True` y los `autotune_config_overrides` del ejemplo), pero sin el parámetro `epilogue`, que aquí es la identidad. El ejemplo tiene test en el CI de Helion (`test_examples.py::test_matmul`).

**`cutlass/matmul_cute.py`.** Carga el ejemplo oficial de CuTe DSL para Blackwell GeForce desde `$CUTLASS_DIR` **sin modificarlo**: fp16 con acumulación en fp32, TMA y pipeline de varias etapas. Solo la envoltura es propia: pasa las matrices de PyTorch como tensores CuTe sin copiarlas (A con K contigua, B y C con N contigua) y compila el kernel con `cute.compile`, siguiendo la función `run()` del propio ejemplo. El kernel no admite bf16. Paquete `nvidia-cutlass-dsl` 4.8.0, la misma versión que CUTLASS.

#### Kernels compute-bound (`src/CBKernels/`): FlashAttention (forward)

Solo se usan implementaciones **oficiales** de cada proyecto. Antes de medir se comprobó en
las dos GPUs (RTX 5090 sm_120 y GB10 sm_121) cuáles se ejecutan y dan el mismo resultado que
`torch.nn.functional.scaled_dot_product_attention` (atol 1e-2): jobs 20639 (pascal) y 20640
(hennessy), con D ∈ {64, 128}, causal y no causal.

**`triton/fused_attention.py`.** Es el fichero del tutorial **byte a byte** (`cmp` idéntico),
copiado al repositorio porque los tutoriales no se instalan con el paquete de Triton.
FlashAttention-2 (Dao, 2023) con autotuning de `BLOCK_M`, `BLOCK_N`, `num_stages` y `num_warps` y descriptores TMA
de host (en capability ≥ 9). Se llama con `warp_specialize=False`: el benchmark del propio
tutorial solo lo activa en Hopper y Blackwell de centro de datos (capability 9 y 10).

**`helion/attention.py`.** Carga sin modificar `examples/attention.py` del Helion instalado en
la imagen (`0f3b242`; el fichero no cambia hasta `b2dbaee`, 2026-10-04). Usa los kernels que
devuelven solo la salida, ambos con test en el CI de Helion
(`test_examples.py::test_attention_output` y `::test_causal_attention_output`). Escala
1/√D fija, igual que el banco. El kernel causal **recorre todos los bloques** de K/V y enmascara
con −∞: no se salta el triángulo superior, así que hace el trabajo del no causal.

**`cutlass/attention_fa2.py`.** Carga sin modificar el ejemplo de FlashAttention-2 de CuTe DSL
para Ampere (`mma.sync` 16×8×16 vía `warp.MmaF16BF16Op`, `cp.async` y `ldmatrix`). Es el único
FMHA de CuTe DSL que puede correr en sm_120/121: los de `hopper/` y `blackwell/` usan `wgmma` y
`tcgen05`. La envoltura pasa los tensores como la función `run()` del ejemplo, que trabaja en
(B, N, H, D): se hace una copia transpuesta de Q, K y V **fuera de la medida**. Como el ejemplo
no tiene autotuning, el banco barre sus parámetros públicos (`m_block_size`, `n_block_size` ∈
{64, 128}, `num_threads` ∈ {128, 256}) que acepta su `can_implement` y usa el más rápido.

**Gluon.** El único ejemplo oficial de atención (`examples/gluon/01-attention-forward.py`,
v3.8.0) está escrito para Blackwell de centro de datos (`tcgen05_mma`, memoria de tensor). En
sm_120 y sm_121 el compilador aborta (`LLVM ERROR: Cannot select: intrinsic
%llvm.nvvm.tcgen05.wait.st`): esa instrucción no existe en Blackwell GeForce.

**Triton + TLX.** Los tutoriales de FlashAttention de µTLX son solo para Hopper (`hopper_fa_ws*`)
y Blackwell de centro de datos (`blackwell_fa_ws*`). Su MMA, `tlx.async_dot`, genera `wgmma`
(capability < 100) o `tcgen05` (≥ 100) según `utlx_plugin/mma_ops.py`: ninguna de las dos
existe en sm_120/121. Además, el tutorial de `main` (`7d2619e`) no compila con el paquete
instalado `triton-utlx` 3.8.0.post1 (`TypeError: async_tasks.__init__() got an unexpected
keyword argument '_semantic'`, job 20644). Con D=64 lo rechaza su propia comprobación
`tl.static_assert(BLOCK_N <= HEAD_DIM)`.
El fork de Meta con TLX integrado (`facebookexperimental/triton`, `a05314c`, 2026-10-02) tampoco
tiene tutoriales de atención para sm_120/121 (`third_party/tlx/tutorials/`: solo `blackwell_*`,
`hopper_*` y `amd_*`). Por ejemplo, `blackwell_fa_ws_pipelined_persistent.py` llama a
`tlx.tcgen05_commit` y usa memoria de tensor (TMEM), exclusivas de sm_100, y su `tlx.async_dot`
elige `tcgen05` o `wgmma` por capability (`third_party/tlx/language/tlx/mma_ops.py`). *No se ha
ejecutado*: requiere compilar ese fork de Triton; la conclusión sale de su código fuente.

#### Kernels memory-bound (`src/MBKernels/`): RMSNorm

**`triton/rmsnorm_baseline.py`.** Adapta el LayerNorm del tutorial 05: un programa por fila, fila entera en registros, `MAX_FUSED_SIZE = 65536` y la heurística de warps `BLOCK_SIZE // 256`. El tutorial limita a 8 warps y aquí se llega a 16. Se elimina la resta de la media, porque RMSNorm no centra, y la reducción es de x² en fp32.

**`triton_tlx/rmsnorm.py`.** Es un diseño propio: kernel persistente con *prefetch* de filas a memoria compartida en un anillo de `NUM_STAGES` buffers (`cp.async`). Las llamadas (`tlx.local_alloc`, `local_view`, `async_load`, `async_load_commit_group` / `wait_group`, `local_load`) siguen el patrón de `test/test_tlx.py::test_local_load` y del tutorial `blackwell-multi-cta-layernorm` de triton-ext. Con triton-utlx 3.8.0.post1 **no compila**: el plugin crea `ttg.async_copy_global_to_local` sin `operandSegmentSizes` y el verificador de MLIR lo rechaza. Falla incluso el ejemplo mínimo del propio repositorio, así que no es un error del kernel.

**`gluon/rmsnorm.py`.** Mismo algoritmo que el baseline Triton, pero con el reparto de la fila entre hilos fijado a mano mediante un `BlockedLayout` explícito (16 B por hilo, para cargas vectorizadas). La API sigue el tutorial de layouts de Gluon de Triton v3.8.0.

**`helion/rmsnorm.py`.** Es el `rms_norm_fwd` del ejemplo oficial sin la salida auxiliar `inv_rms`, que solo necesita el *backward*. Así mueve los mismos bytes que las demás implementaciones. El kernel Triton lo genera y autotunea Helion.

**`cutlass/rmsnorm_ext.cu`.** Llama a la función oficial `cutlass::rmsnorm` **sin modificarla**. En fp16 con N % 8 == 0 usa el kernel vectorizado con `float4` (`rmsnorm_twoPassAlgo_e8`); en bf16 usa el escalar (`_e1`). Solo es propia la envoltura como extensión de PyTorch (`torch.utils.cpp_extension`), para medirlo con el mismo banco que el resto.

#### Microbenchmark de hardware (`src/Microbench/`)

**`pico_mma_ext.cu`.** No es un kernel de ningún DSL, sino código de caracterización del hardware: cadenas de `mma.sync` en PTX en línea sobre registros, sin accesos a memoria, con ILP acumuladores independientes por warp. Usa la técnica habitual de los microbenchmarks de *throughput* de instrucciones: dependencias controladas, un único volcado final para que el compilador no elimine el bucle y barrido de la ocupación. La sintaxis de cada instrucción (incluidas las *block-scaled* `kind::mxf8f6f4` y `kind::mxf4nvf4`, con sus operandos de escala) sigue la PTX ISA de NVIDIA. Que sea correcta lo confirman ptxas, que compila cada instrucción por separado para `sm_121a`, y el SASS (`cuobjdump`), que muestra la instrucción de tensor core esperada (`HMMA`, `QMMA`, `IMMA`, `OMMA`). `pico_mma.py` lo compila con nvcc a una biblioteca compartida y lo carga con ctypes.

#### Referencias de comparación (no son kernels propios)

<a id="resultados"></a>

## 5. Resultados por algoritmo y DSL

*Cada resultado: rendimiento en el mayor tamaño medido · **% del pico** · entorno y job. Una celda sin dato dice por qué falta; n/a = la variante no existe para ese algoritmo. Configuración aplicada en cada celda (bloques, warps, etapas): `results/resumen.csv`.*

### 5.1 Matmul

| DSL | Kernel | Elaboración | Fuente | GB10 fp16 | GB10 FP8 | RTX 5090 fp16 | RTX 5090 FP8 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | *fp16*: `triton/matmul.py` (baseline)<br>*FP8*: `triton/matmul_fp8.py` | Adaptado | *fp16*: Tutorial oficial de Triton `03-matrix-multiplication.py` [1]<br>*FP8*: Esqueleto de `matmul.py`; el tutorial 03 ya muestra entradas FP8 [1] | **89.89 TFLOP/s** · **73 %**<br>_hennessy, job 21058_ | **143.4 TFLOP/s** · **59 %**<br>_hennessy, job 21058_ | **217.89 TFLOP/s** · **104 %**<br>_pascal, job 20499_ | **394.76 TFLOP/s** · **94 %**<br>_pascal, job 20499_ |
| **Triton (block-ptr)** | `triton/matmul_blockptr.py` | Adaptado | Esqueleto de `matmul.py` + API `tl.make_block_ptr` / `tl.advance` de Triton [1] | **91.71 TFLOP/s** · **75 %**<br>_hennessy, job 21058_ | n/a | **217.6 TFLOP/s** · **104 %**<br>_pascal, job 20499_ | n/a |
| **Triton + TMA** | *fp16*: `triton/matmul_tma.py`<br>*FP8*: `triton/matmul_fp8_tma.py` | *fp16*: Adaptado<br>*FP8*: Propio (combinación) | *fp16*: Tutorial oficial de Triton `09-persistent-matmul.py`, variantes con TMA [1]<br>*FP8*: `matmul_fp8.py` + descriptores TMA de `matmul_tma.py` | **94.62 TFLOP/s** · **77 %**<br>_hennessy, job 21058_ | **150.05 TFLOP/s** · **62 %**<br>_hennessy, job 21058_ | **223.6 TFLOP/s** · **107 %**<br>_pascal, job 20499_ | **394.47 TFLOP/s** · **94 %**<br>_pascal, job 20499_ |
| **Triton + TLX** | — | sin kernel | — | — *sin kernel*: no hay matmul en TLX | — *sin kernel*: no hay matmul en TLX | — *sin kernel*: no hay matmul en TLX | — *sin kernel*: no hay matmul en TLX |
| **Gluon** | `gluon/matmul.py` | Propio | Piezas oficiales de Triton v3.8.0: `mma_v2` (`test_core.py::test_mma_v2`) y pipeline `cp.async` (`gluon/03-async-copy.py`) [1] | **77.45 TFLOP/s** · **63 %**<br>_hennessy, job 21058_ | — *sin kernel*: no hay matmul FP8 en este DSL | — *no medido*: no se ha ejecutado en esta GPU | — *sin kernel*: no hay matmul FP8 en este DSL |
| **Helion** | `helion/matmul.py` | Adaptado | Ejemplo oficial de Helion `examples/matmul.py` (`matmul`) [3] | **61.94 TFLOP/s** · **50 %** (solo medido con caché fría)<br>_hennessy-frio, job 20841_ | — *sin kernel*: no hay matmul FP8 en este DSL | — *no medido*: no se ha ejecutado en esta GPU | — *sin kernel*: no hay matmul FP8 en este DSL |
| **CUTLASS** | `cutlass/matmul_cute.py` | Oficial sin modificar | `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py` (`Sm120GemmKernel`) [2] | **23.59 TFLOP/s** · **19 %** (solo medido con caché fría)<br>_hennessy-frio, job 20841_ | — *sin kernel*: el dense_gemm de CuTe solo admite fp16/bf16 (su FP8 en GeForce es MXFP8) | **165.83 TFLOP/s** · **79 %**<br>_pascal, job 20638_ | — *sin kernel*: el dense_gemm de CuTe solo admite fp16/bf16 (su FP8 en GeForce es MXFP8) |
| **Referencia** | *fp16*: `torch.matmul`<br>*FP8*: `torch._scaled_mm` | Referencia (no es kernel propio) | *fp16*: cuBLAS (fp16/bf16; fp32 en TF32)<br>*FP8*: cuBLASLt en FP8 | **92.96 TFLOP/s** · **76 %**<br>_hennessy, job 21058_ | **184.97 TFLOP/s** · **76 %**<br>_hennessy, job 21058_ | **209.14 TFLOP/s** · **100 %**<br>_pascal, job 20499_ | **598.67 TFLOP/s** · **143 %**<br>_pascal, job 20499_ |

#### Comparación y conclusiones: Matmul

*Cada celda: rendimiento · % del pico · error numérico o validación · qué limita al kernel según ncu · eficiencia energética · entorno y job; productividad en GB10.*

##### Matmul fp16 (CB)

- **GB10**: mejor DSL **Triton + TMA** (94.62 TFLOP/s, 77 % del pico mma.sync medido), 102% de la referencia (76 %); el más lento, CUTLASS (19 %); según ncu, a Triton + TMA le limita tensor cores (90 % de ciclos).
- **RTX 5090**: mejor DSL **Triton + TMA** (223.6 TFLOP/s, 107 % del pico ficha NVIDIA), 107% de la referencia (100 %); el más lento, CUTLASS (79 %).
- RTX 5090/GB10: de ×2.25 (Referencia) a ×7.03 (CUTLASS).

| DSL | GB10 | RTX 5090 | RTX 5090/GB10 | productividad (GB10) |
|:---|:---|:---|:---|:---|
| **Triton** | **89.89 TFLOP/s** · **73 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: SM 83.01 % · tensor 83.01 % · mem 51.76 % · ocup. 16.63 %<br>limita: **tensor cores (83 % de ciclos)**; ocupación por mem. compartida<br>1019.6 GFLOP/J (×0.97 ref.)<br>_hennessy, job 21058_ | **217.89 TFLOP/s** · **104 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×2.42 | 67 LoC · 34.1 s<br>autotune (triton.autotune) |
| **Triton (block-ptr)** | **91.71 TFLOP/s** · **75 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: SM 85.27 % · tensor 85.27 % · mem 52.98 % · ocup. 15.65 %<br>limita: **tensor cores (85 % de ciclos)**; ocupación por mem. compartida<br>1036 GFLOP/J (×0.99 ref.)<br>_hennessy, job 21058_ | **217.6 TFLOP/s** · **104 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×2.37 | — |
| **Triton + TMA** | **94.62 TFLOP/s** · **77 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: SM 89.71 % · tensor 89.71 % · mem 73.36 % · ocup. 15.46 %<br>limita: **tensor cores (90 % de ciclos)**; ocupación por mem. compartida<br>1064.1 GFLOP/J (×1.01 ref.)<br>_hennessy, job 21058_ | **223.6 TFLOP/s** · **107 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×2.36 | 68 LoC · 96.1 s<br>autotune (triton.autotune) |
| **Triton + TLX** | — *sin kernel*: no hay matmul en TLX | — *sin kernel*: no hay matmul en TLX | — | — |
| **Gluon** | **77.45 TFLOP/s** · **63 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: SM 67.33 % · tensor 67.33 % · mem 56.11 % · ocup. 16.63 %<br>limita: **tensor cores (67 % de ciclos)**; ocupación por registros<br>860.6 GFLOP/J (×0.82 ref.)<br>_hennessy, job 21058_ | — *no medido*: no se ha ejecutado en esta GPU | — | 101 LoC · 3.9 s<br>barrido propio (sin autotuning) |
| **Helion** | **61.94 TFLOP/s** · **50 %** (solo medido con caché fría)<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>_hennessy-frio, job 20841_ | — *no medido*: no se ha ejecutado en esta GPU | — | 25 LoC · 4.0 min<br>autotune Helion (quick) |
| **CUTLASS** | **23.59 TFLOP/s** · **19 %** (solo medido con caché fría)<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>573.2 GFLOP/J (×0.55 ref.)<br>_hennessy-frio, job 20841_ | **165.83 TFLOP/s** · **79 %**<br>validado frente a cuBLAS (atol 1e-1, rtol 1e-2)<br>ncu: *sin perfil*<br>_pascal, job 20638_ | ×7.03 | 713 LoC · 1.0 s<br>fija (tile del ejemplo) |
| **Referencia** | **92.96 TFLOP/s** · **76 %**<br>es la referencia<br>ncu: SM 85.65 % · tensor 85.65 % · mem 57.41 % · ocup. 15.56 %<br>limita: **tensor cores (86 % de ciclos)**; ocupación por registros<br>1051.1 GFLOP/J<br>_hennessy, job 21058_ | **209.14 TFLOP/s** · **100 %**<br>es la referencia<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×2.25 | — |

*Tamaño: GB10 8192×8192×8192; RTX 5090 8192×8192×8192.*

##### Matmul FP8 (CB)

- **GB10**: mejor DSL **Triton + TMA** (150.05 TFLOP/s, 62 % del pico mma.sync medido), 81% de la referencia (76 %); el más lento, Triton (59 %); según ncu, a Triton + TMA le limita tensor cores (68 % de ciclos).
- **RTX 5090**: mejor DSL **Triton** (394.76 TFLOP/s, 94 % del pico ficha NVIDIA), 66% de la referencia (143 %); el más lento, Triton + TMA (94 %).
- RTX 5090/GB10: de ×2.63 (Triton + TMA) a ×3.24 (Referencia).

| DSL | GB10 | RTX 5090 | RTX 5090/GB10 | productividad (GB10) |
|:---|:---|:---|:---|:---|
| **Triton** | **143.4 TFLOP/s** · **59 %**<br>error 3.75%<br>ncu: SM 76.38 % · tensor 66.41 % · mem 76.38 % · ocup. 15.77 %<br>limita: **tensor cores (66 % de ciclos)**; ocupación por mem. compartida<br>1585.8 GFLOP/J (×0.78 ref.)<br>_hennessy, job 21058_ | **394.76 TFLOP/s** · **94 %**<br>error 3.75%<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×2.75 | — |
| **Triton + TMA** | **150.05 TFLOP/s** · **62 %**<br>error 3.75%<br>ncu: SM 73.01 % · tensor 68.05 % · mem 77.14 % · ocup. 15.57 %<br>limita: **tensor cores (68 % de ciclos)**; ocupación por mem. compartida<br>1629.2 GFLOP/J (×0.8 ref.)<br>_hennessy, job 21058_ | **394.47 TFLOP/s** · **94 %**<br>error 3.75%<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×2.63 | — |
| **Triton + TLX** | — *sin kernel*: no hay matmul en TLX | — *sin kernel*: no hay matmul en TLX | — | — |
| **Gluon** | — *sin kernel*: no hay matmul FP8 en este DSL | — *sin kernel*: no hay matmul FP8 en este DSL | — | — |
| **Helion** | — *sin kernel*: no hay matmul FP8 en este DSL | — *sin kernel*: no hay matmul FP8 en este DSL | — | — |
| **CUTLASS** | — *sin kernel*: el dense_gemm de CuTe solo admite fp16/bf16 (su FP8 en GeForce es MXFP8) | — *sin kernel*: el dense_gemm de CuTe solo admite fp16/bf16 (su FP8 en GeForce es MXFP8) | — | — |
| **Referencia** | **184.97 TFLOP/s** · **76 %**<br>error 3.75%<br>ncu: SM 86.2 % · tensor 86.2 % · mem 54.5 % · ocup. 16.29 %<br>limita: **tensor cores (86 % de ciclos)**; ocupación por registros<br>2036.5 GFLOP/J<br>_hennessy, job 21058_ | **598.67 TFLOP/s** · **143 %**<br>error 3.75%<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×3.24 | — |

*Tamaño: GB10 8192×8192×8192; RTX 5090 8192×8192×8192.*

#### Perfil de Nsight Compute: Matmul

*Kernel principal de cada medida (el de mayor duración; +n = otros kernels de la misma llamada) en el mayor tamaño medido. **SM %** y **memoria %**: *Speed of Light* de cómputo y de memoria; **tensor %**: ciclos con el pipe de tensor ocupado; **emisión %**: ciclos en que cada SMSP emite una instrucción; **limita**: recurso que fija la ocupación teórica; **memoria principal**: DRAM en RTX 5090; en GB10 (memoria unificada, sin contadores dram) los bytes que fallan en L2; **espera principal**: motivo con más ciclos de espera por instrucción emitida. ncu con relojes sin fijar (`--clock-control none`).*

##### NVIDIA GB10 (sm_121) — Matmul fp16

| DSL | kernel | µs | MHz | SM % | tensor % | memoria % | emisión % | ocupación % | ocup. teórica % | limita | regs/hilo | smem KB/bloque | acierto L2 % | memoria principal MB | conflictos smem | espera principal |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | `matmul_kernel` | 11405.216 | 2362.875 | 83.01 | 83.01 | 51.76 | 9.65 | 16.63 | 0 | mem. compartida | 226 | 97 | 84.53 | 2013.266 (fallos L2) | 5625 | math pipe throttle |
| **Triton (block-ptr)** | `matmul_blockptr_kernel` | 11182.656 | 2345.961 | 85.27 | 85.27 | 52.98 | 10.72 | 15.65 | 0 | mem. compartida | 204 | 97 | 84.53 | 2013.046 (fallos L2) | 10907 | math pipe throttle |
| **Triton + TMA** | `matmul_tma_kernel` | 10845.696 | 2298.985 | 89.71 | 89.71 | 73.36 | 11.03 | 15.46 | 0 | mem. compartida | 207 | 49.023 | 88.36 | 2014.553 (fallos L2) | 113562 | math pipe throttle |
| **Triton + TLX** | — *sin kernel*: no hay matmul en TLX | | | | | | | | | | | | | | | |
| **Gluon** | `matmul_kernel` | 13938.176 | 2383.825 | 67.33 | 67.33 | 56.11 | 15.67 | 16.63 | 16.67 | registros | 220 | 49 | 87.62 | 2116.555 (fallos L2) | 402933648 | math pipe throttle |
| **Helion** | — *no medido*: sin perfil de ncu en esta GPU | | | | | | | | | | | | | | | |
| **CUTLASS** | — *no medido*: sin perfil de ncu en esta GPU | | | | | | | | | | | | | | | |
| **Referencia** | `nvjet_sm121_hsh_mma_128x208x64_2_32x10…` | 11352.032 | 2336.718 | 85.65 | 85.65 | 57.41 | 9.19 | 15.56 | 16.67 | registros | 255 | 86 | 85.51 | 2020.086 (fallos L2) | 0 | math pipe throttle |

*Tamaño: 8192×8192×8192.*

##### NVIDIA GB10 (sm_121) — Matmul FP8

| DSL | kernel | µs | MHz | SM % | tensor % | memoria % | emisión % | ocupación % | ocup. teórica % | limita | regs/hilo | smem KB/bloque | acierto L2 % | memoria principal MB | conflictos smem | espera principal |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | `matmul_fp8_kernel` | 7194.848 | 2340.792 | 76.38 | 66.41 | 76.38 | 22.22 | 15.77 | 0 | mem. compartida | 248 | 97 | 85.24 | 966.005 (fallos L2) | 5159729 | math pipe throttle |
| **Triton + TMA** | `matmul_fp8_tma_kernel` | 6966.08 | 2359.596 | 73.01 | 68.05 | 77.14 | 21.99 | 15.57 | 0 | mem. compartida | 230 | 97.016 | 86.42 | 884.254 (fallos L2) | 268435456 | math pipe throttle |
| **Triton + TLX** | — *sin kernel*: no hay matmul en TLX | | | | | | | | | | | | | | | |
| **Gluon** | — *sin kernel*: no hay matmul FP8 en este DSL | | | | | | | | | | | | | | | |
| **Helion** | — *sin kernel*: no hay matmul FP8 en este DSL | | | | | | | | | | | | | | | |
| **CUTLASS** | — *sin kernel*: el dense_gemm de CuTe solo admite fp16/bf16 (su FP8 en GeForce es MXFP8) | | | | | | | | | | | | | | | |
| **Referencia** | `nvjet_sm121_qqhsh_mma_128x256x128_2_32…` | 5575.072 | 2327.481 | 86.2 | 86.2 | 54.5 | 8.43 | 16.29 | 16.67 | registros | 255 | 98 | 89.79 | 680.355 (fallos L2) | 0 | math pipe throttle |

*Tamaño: 8192×8192×8192.*

### 5.2 FlashAttention (forward)

| DSL | Kernel | Elaboración | Fuente | GB10 fp16 | GB10 causal | RTX 5090 fp16 | RTX 5090 causal |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | `triton/fused_attention.py` | Oficial sin modificar (copia literal) | Tutorial oficial `06-fused-attention.py` [1] · ¿corre en sm_120/121? SI | **87.96 TFLOP/s** · **72 %**<br>_hennessy, job 21058_ | **84.81 TFLOP/s** · **69 %**<br>_hennessy, job 21058_ | **162.82 TFLOP/s** · **78 %**<br>_pascal, job 20648_ | **157.2 TFLOP/s** · **75 %**<br>_pascal, job 20648_ |
| **Triton + TLX** | — | no hay oficial ejecutable | tutoriales `tlx/tutorials/{hopper,blackwell}_fa_ws*.py` [4] · ¿corre en sm_120/121? NO ver notas de procedencia | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) |
| **Gluon** | — | no hay oficial ejecutable | `python/examples/gluon/01-attention-forward.py` [1] usa `tcgen05` · ¿corre en sm_120/121? NO `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st` | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st |
| **Helion** | `helion/attention.py` → `examples/attention.py` | Oficial sin modificar | `attention_output` / `causal_attention_output` [3] · ¿corre en sm_120/121? SI | **67.48 TFLOP/s** · **55 %**<br>_hennessy, job 21058_ | **43.9 TFLOP/s** · **36 %**<br>_hennessy, job 21058_ | **149.88 TFLOP/s** · **72 %**<br>_pascal, job 20648_ | **75.85 TFLOP/s** · **36 %**<br>_pascal, job 20648_ |
| **CUTLASS** | `cutlass/attention_fa2.py` → `flash_attention_v2.py` | Oficial sin modificar | `examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py` (`FlashAttentionForwardAmpere`) [2] · ¿corre en sm_120/121? SI | **91.39 TFLOP/s** · **74 %**<br>_hennessy, job 20649_ | **90.8 TFLOP/s** · **74 %**<br>_hennessy, job 20649_ | **164.64 TFLOP/s** · **79 %**<br>_pascal, job 20648_ | **162.18 TFLOP/s** · **77 %**<br>_pascal, job 20648_ |
| **Referencia** | `torch.nn.functional.scaled_dot_product_attention` | Referencia (no es kernel propio) | FlashAttention-2 incluido en PyTorch (`pytorch_flash::flash_fwd_kernel`); también es la referencia de corrección | **82.19 TFLOP/s** · **67 %**<br>_hennessy, job 21058_ | **84.21 TFLOP/s** · **69 %**<br>_hennessy, job 21058_ | **152.93 TFLOP/s** · **73 %**<br>_pascal, job 20648_ | **152.36 TFLOP/s** · **73 %**<br>_pascal, job 20648_ |

#### Comparación y conclusiones: FlashAttention (forward)

*Cada celda: rendimiento · % del pico · error numérico o validación · qué limita al kernel según ncu · eficiencia energética · entorno y job; productividad en GB10.*

##### FlashAttention fp16 (CB)

- **GB10**: mejor DSL **CUTLASS** (91.39 TFLOP/s, 74 % del pico mma.sync medido), 111% de la referencia (67 %); el más lento, Helion (55 %).
- **RTX 5090**: mejor DSL **CUTLASS** (164.64 TFLOP/s, 79 % del pico ficha NVIDIA), 108% de la referencia (73 %); el más lento, Helion (72 %).
- RTX 5090/GB10: de ×1.80 (CUTLASS) a ×2.22 (Helion).

| DSL | GB10 | RTX 5090 | RTX 5090/GB10 | productividad (GB10) |
|:---|:---|:---|:---|:---|
| **Triton** | **87.96 TFLOP/s** · **72 %**<br>error 3.9e-04<br>ncu: SM 82.38 % · tensor 82.38 % · mem 34.48 % · ocup. 16.72 %<br>limita: **tensor cores (82 % de ciclos)**; ocupación por mem. compartida<br>913.6 GFLOP/J (×1.08 ref.)<br>_hennessy, job 21058_ | **162.82 TFLOP/s** · **78 %**<br>error 3.9e-04<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.85 | 220 LoC · 15.2 s<br>autotune (triton.autotune) |
| **Triton + TLX** | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — | — |
| **Gluon** | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — | — |
| **Helion** | **67.48 TFLOP/s** · **55 %**<br>error 4.5e-04<br>ncu: *sin perfil*<br>875.2 GFLOP/J (×1.03 ref.)<br>_hennessy, job 21058_ | **149.88 TFLOP/s** · **72 %**<br>error 4.5e-04<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×2.22 | 63 LoC · 2.1 min<br>autotune Helion (quick) |
| **CUTLASS** | **91.39 TFLOP/s** · **74 %**<br>error 1.1e-04<br>ncu: *sin perfil*<br>930.3 GFLOP/J (×1.1 ref.)<br>_hennessy, job 20649_ | **164.64 TFLOP/s** · **79 %**<br>error 1.1e-04<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.80 | 778 LoC · 7.7 s<br>barrido propio (sin autotuning) |
| **Referencia** | **82.19 TFLOP/s** · **67 %**<br>es la referencia<br>ncu: SM 75.42 % · tensor 75.42 % · mem 36.37 % · ocup. 8.37 %<br>limita: **tensor cores (75 % de ciclos)**; ocupación por mem. compartida<br>846.6 GFLOP/J<br>_hennessy, job 21058_ | **152.93 TFLOP/s** · **73 %**<br>es la referencia<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.86 | — |

*Tamaño: GB10 B=4 H=32 N=16384 D=128; RTX 5090 B=4 H=32 N=16384 D=128.*

##### FlashAttention causal fp16 (CB)

- **GB10**: mejor DSL **CUTLASS** (90.8 TFLOP/s, 74 % del pico mma.sync medido), 108% de la referencia (69 %); el más lento, Helion (36 %).
- **RTX 5090**: mejor DSL **CUTLASS** (162.18 TFLOP/s, 77 % del pico ficha NVIDIA), 106% de la referencia (73 %); el más lento, Helion (36 %).
- RTX 5090/GB10: de ×1.73 (Helion) a ×1.85 (Triton).

| DSL | GB10 | RTX 5090 | RTX 5090/GB10 | productividad (GB10) |
|:---|:---|:---|:---|:---|
| **Triton** | **84.81 TFLOP/s** · **69 %**<br>error 2.8e-04<br>ncu: SM 78.28 % · tensor 78.28 % · mem 37.4 % · ocup. 16.69 %<br>limita: **tensor cores (78 % de ciclos)**; ocupación por registros<br>853.6 GFLOP/J (×1 ref.)<br>_hennessy, job 21058_ | **157.2 TFLOP/s** · **75 %**<br>error 2.8e-04<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.85 | — |
| **Triton + TLX** | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | — | — |
| **Gluon** | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — | — |
| **Helion** | **43.9 TFLOP/s** · **36 %**<br>error 3.9e-04<br>ncu: *sin perfil*<br>425.2 GFLOP/J (×0.5 ref.)<br>_hennessy, job 21058_ | **75.85 TFLOP/s** · **36 %**<br>error 3.9e-04<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.73 | — |
| **CUTLASS** | **90.8 TFLOP/s** · **74 %**<br>error 1.3e-04<br>ncu: *sin perfil*<br>899.8 GFLOP/J (×1.06 ref.)<br>_hennessy, job 20649_ | **162.18 TFLOP/s** · **77 %**<br>error 2.0e-05<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.79 | — |
| **Referencia** | **84.21 TFLOP/s** · **69 %**<br>es la referencia<br>ncu: SM 77.19 % · tensor 77.19 % · mem 32.42 % · ocup. 8.32 %<br>limita: **tensor cores (77 % de ciclos)**; ocupación por mem. compartida<br>851 GFLOP/J<br>_hennessy, job 21058_ | **152.36 TFLOP/s** · **73 %**<br>es la referencia<br>ncu: *sin perfil*<br>_pascal, job 20648_ | ×1.81 | — |

*Tamaño: GB10 B=4 H=32 N=16384 D=128; RTX 5090 B=4 H=32 N=16384 D=128.*

#### Perfil de Nsight Compute: FlashAttention (forward)

*Kernel principal de cada medida (el de mayor duración; +n = otros kernels de la misma llamada) en el mayor tamaño medido. **SM %** y **memoria %**: *Speed of Light* de cómputo y de memoria; **tensor %**: ciclos con el pipe de tensor ocupado; **emisión %**: ciclos en que cada SMSP emite una instrucción; **limita**: recurso que fija la ocupación teórica; **memoria principal**: DRAM en RTX 5090; en GB10 (memoria unificada, sin contadores dram) los bytes que fallan en L2; **espera principal**: motivo con más ciclos de espera por instrucción emitida. ncu con relojes sin fijar (`--clock-control none`).*

##### NVIDIA GB10 (sm_121) — FlashAttention fp16

| DSL | kernel | µs | MHz | SM % | tensor % | memoria % | emisión % | ocupación % | ocup. teórica % | limita | regs/hilo | smem KB/bloque | acierto L2 % | memoria principal MB | conflictos smem | espera principal |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | `_attn_fwd` | 194022.176 | 2239.275 | 82.38 | 82.38 | 34.48 | 22.34 | 16.72 | 0 | mem. compartida | 255 | 49.023 | 98.43 | 2156.036 (fallos L2) | 4003775 | math pipe throttle |
| **Triton + TLX** | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | | | | | | | | | | | | | | | |
| **Gluon** | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | | | | | | | | | | | | | | | |
| **Helion** | — *no medido*: sin perfil de ncu en esta GPU | | | | | | | | | | | | | | | |
| **CUTLASS** | — *no medido*: sin perfil de ncu en esta GPU | | | | | | | | | | | | | | | |
| **Referencia** | `pytorch_flash::flash_fwd_kernel<Flash_…` | 209586.4 | 2264.221 | 75.42 | 75.42 | 36.37 | 25.93 | 8.37 | 8.33 | mem. compartida | 255 | 65 | 98.66 | 2165.502 (fallos L2) | 61107 | wait |

*Tamaño: B=4 H=32 N=16384 D=128.*

##### NVIDIA GB10 (sm_121) — FlashAttention causal fp16

| DSL | kernel | µs | MHz | SM % | tensor % | memoria % | emisión % | ocupación % | ocup. teórica % | limita | regs/hilo | smem KB/bloque | acierto L2 % | memoria principal MB | conflictos smem | espera principal |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | `_attn_fwd` | 98236.192 | 2345.311 | 78.28 | 78.28 | 37.4 | 22.06 | 16.69 | 16.67 | registros | 255 | 33.008 | 97.27 | 2183.442 (fallos L2) | 24675768 | wait |
| **Triton + TLX** | — *no compila*: los tutoriales de TLX usan wgmma/tcgen05 (Hopper, Blackwell de servidor) | | | | | | | | | | | | | | | |
| **Gluon** | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | | | | | | | | | | | | | | | |
| **Helion** | — *no medido*: sin perfil de ncu en esta GPU | | | | | | | | | | | | | | | |
| **CUTLASS** | — *no medido*: sin perfil de ncu en esta GPU | | | | | | | | | | | | | | | |
| **Referencia** | `pytorch_flash::flash_fwd_kernel<Flash_…` | 99250.464 | 2354.154 | 77.19 | 77.19 | 32.42 | 26.06 | 8.32 | 8.33 | mem. compartida | 255 | 65 | 96.93 | 2152.131 (fallos L2) | 16465 | wait |

*Tamaño: B=4 H=32 N=16384 D=128.*

### 5.3 RMSNorm

| DSL | Kernel | Elaboración | Fuente | GB10 fp16 | GB10 bf16 | RTX 5090 fp16 | RTX 5090 bf16 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | `triton/rmsnorm_baseline.py` | Adaptado | Tutorial oficial de Triton `05-layer-norm.py` (`_layer_norm_fwd_fused`) [1] | **242.7 GB/s** · **89 %**<br>_hennessy, job 21058_ | **242.8 GB/s** · **89 %**<br>_hennessy, job 21058_ | **1521.4 GB/s** · **85 %**<br>_pascal, job 20499_ | **1513.4 GB/s** · **84 %**<br>_pascal, job 20499_ |
| **Triton + TLX** | `triton_tlx/rmsnorm.py` | Propio | API de µTLX (`triton-lang/triton-ext`, `extensions/utlx`) [4] | — *no compila*: triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes) | — *no compila*: triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes) | **1456.4 GB/s** · **81 %**<br>_pascal, job 20499_ | **1456.4 GB/s** · **81 %**<br>_pascal, job 20499_ |
| **Gluon** | `gluon/rmsnorm.py` | Propio | API de Gluon, tutorial oficial `gluon/02-layouts.py` [1] | **242.8 GB/s** · **89 %**<br>_hennessy, job 21058_ | **242.8 GB/s** · **89 %**<br>_hennessy, job 21058_ | **1521.2 GB/s** · **85 %**<br>_pascal, job 20499_ | **1513.4 GB/s** · **84 %**<br>_pascal, job 20499_ |
| **Helion** | `helion/rmsnorm.py` | Adaptado | Ejemplo oficial de Helion `examples/rms_norm.py` (`rms_norm_fwd`) [3] | **230.3 GB/s** · **84 %**<br>_patterson, job 20435_ | **226.6 GB/s** · **83 %**<br>_patterson, job 20435_ | — *no medido*: no se ha ejecutado en esta GPU | — *no medido*: no se ha ejecutado en esta GPU |
| **CUTLASS** | `cutlass/rmsnorm_ext.cu` | Oficial sin modificar | `cutlass::rmsnorm`, `tools/util/include/cutlass/util/device_rmsnorm.h` [2] | **230.5 GB/s** · **84 %**<br>_patterson, job 20435_ | **211.6 GB/s** · **78 %**<br>_patterson, job 20435_ | **1438.7 GB/s** · **80 %**<br>_pascal, job 20638_ | **1400.1 GB/s** · **78 %**<br>_pascal, job 20638_ |
| **Referencia** | `torch.nn.functional.rms_norm` | Referencia (no es kernel propio) | kernel RMSNorm de PyTorch; también es la referencia de corrección | **217.1 GB/s** · **80 %**<br>_hennessy, job 21058_ | **216.8 GB/s** · **79 %**<br>_hennessy, job 21058_ | **1299.3 GB/s** · **72 %**<br>_pascal, job 20638_ | **1302.8 GB/s** · **73 %**<br>_pascal, job 20638_ |

#### Comparación y conclusiones: RMSNorm

*Cada celda: rendimiento · % del pico · error numérico o validación · qué limita al kernel según ncu · eficiencia energética · entorno y job; productividad en GB10.*

##### RMSNorm fp16 (MB)

- **GB10**: mejor DSL **Gluon** (242.8 GB/s, 89 % del pico de memoria), 112% de la referencia (80 %); el más lento, Helion (84 %); según ncu, a Gluon le limita latencia (espera: long scoreboard).
- **RTX 5090**: mejor DSL **Triton** (1521.4 GB/s, 85 % del pico de memoria), 117% de la referencia (72 %); el más lento, CUTLASS (80 %).
- RTX 5090/GB10: de ×5.98 (Referencia) a ×6.27 (Triton).

| DSL | GB10 | RTX 5090 | RTX 5090/GB10 | productividad (GB10) |
|:---|:---|:---|:---|:---|
| **Triton** | **242.7 GB/s** · **89 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: SM 4.68 % · tensor 0 % · mem 12.52 % · ocup. 88.61 %<br>limita: **latencia (espera: long scoreboard)**; ocupación por registros<br>6.23 GB/J (×1.05 ref.)<br>_hennessy, job 21058_ | **1521.4 GB/s** · **85 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×6.27 | 38 LoC · 0.6 s<br>heurística (warps por N) |
| **Triton + TLX** | — *no compila*: triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes) | **1456.4 GB/s** · **81 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | — | 52 LoC<br>heurística + --stages a mano |
| **Gluon** | **242.8 GB/s** · **89 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: SM 4.51 % · tensor 0 % · mem 12.65 % · ocup. 88.94 %<br>limita: **latencia (espera: long scoreboard)**; ocupación por registros<br>6.37 GB/J (×1.07 ref.)<br>_hennessy, job 21058_ | **1521.2 GB/s** · **85 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×6.27 | 35 LoC · 0.6 s<br>heurística (warps y layout por N) |
| **Helion** | **230.3 GB/s** · **84 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>6.5 GB/J (×1.09 ref.)<br>_patterson, job 20435_ | — *no medido*: no se ha ejecutado en esta GPU | — | 16 LoC · 77.3 s<br>autotune Helion (quick) |
| **CUTLASS** | **230.5 GB/s** · **84 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>6 GB/J (×1.01 ref.)<br>_patterson, job 20435_ | **1438.7 GB/s** · **80 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20638_ | ×6.24 | 203 LoC · 55.5 s<br>fija (heurística de CUTLASS) |
| **Referencia** | **217.1 GB/s** · **80 %**<br>es la referencia<br>ncu: SM 5.49 % · tensor 0 % · mem 14.57 % · ocup. 95.53 %<br>limita: **latencia (espera: long scoreboard)**; ocupación por registros<br>5.95 GB/J<br>_hennessy, job 21058_ | **1299.3 GB/s** · **72 %**<br>es la referencia<br>ncu: *sin perfil*<br>_pascal, job 20638_ | ×5.98 | — |

*Tamaño: GB10 16384×8192; RTX 5090 16384×8192.*

##### RMSNorm bf16 (MB)

- **GB10**: mejor DSL **Triton** (242.8 GB/s, 89 % del pico de memoria), 112% de la referencia (79 %); el más lento, CUTLASS (78 %); según ncu, a Triton le limita latencia (espera: long scoreboard).
- **RTX 5090**: mejor DSL **Triton** (1513.4 GB/s, 84 % del pico de memoria), 116% de la referencia (73 %); el más lento, CUTLASS (78 %).
- RTX 5090/GB10: de ×6.01 (Referencia) a ×6.62 (CUTLASS).

| DSL | GB10 | RTX 5090 | RTX 5090/GB10 | productividad (GB10) |
|:---|:---|:---|:---|:---|
| **Triton** | **242.8 GB/s** · **89 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: SM 5.06 % · tensor 0 % · mem 12.49 % · ocup. 89.03 %<br>limita: **latencia (espera: long scoreboard)**; ocupación por registros<br>6.27 GB/J (×1.04 ref.)<br>_hennessy, job 21058_ | **1513.4 GB/s** · **84 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×6.23 | — |
| **Triton + TLX** | — *no compila*: triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes) | **1456.4 GB/s** · **81 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | — | — |
| **Gluon** | **242.8 GB/s** · **89 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: SM 4.89 % · tensor 0 % · mem 12.63 % · ocup. 88.85 %<br>limita: **latencia (espera: long scoreboard)**; ocupación por registros<br>6.37 GB/J (×1.06 ref.)<br>_hennessy, job 21058_ | **1513.4 GB/s** · **84 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20499_ | ×6.23 | — |
| **Helion** | **226.6 GB/s** · **83 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>6.47 GB/J (×1.07 ref.)<br>_patterson, job 20435_ | — *no medido*: no se ha ejecutado en esta GPU | — | — |
| **CUTLASS** | **211.6 GB/s** · **78 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>4.68 GB/J (×0.78 ref.)<br>_patterson, job 20435_ | **1400.1 GB/s** · **78 %**<br>validado frente a F.rms_norm fp32 (atol 2e-2, rtol 2e-2)<br>ncu: *sin perfil*<br>_pascal, job 20638_ | ×6.62 | — |
| **Referencia** | **216.8 GB/s** · **79 %**<br>es la referencia<br>ncu: SM 5.59 % · tensor 0 % · mem 14.66 % · ocup. 96.91 %<br>limita: **latencia (espera: long scoreboard)**; ocupación por registros<br>6.02 GB/J<br>_hennessy, job 21058_ | **1302.8 GB/s** · **73 %**<br>es la referencia<br>ncu: *sin perfil*<br>_pascal, job 20638_ | ×6.01 | — |

*Tamaño: GB10 16384×8192; RTX 5090 16384×8192.*

#### Perfil de Nsight Compute: RMSNorm

*Kernel principal de cada medida (el de mayor duración; +n = otros kernels de la misma llamada) en el mayor tamaño medido. **SM %** y **memoria %**: *Speed of Light* de cómputo y de memoria; **tensor %**: ciclos con el pipe de tensor ocupado; **emisión %**: ciclos en que cada SMSP emite una instrucción; **limita**: recurso que fija la ocupación teórica; **memoria principal**: DRAM en RTX 5090; en GB10 (memoria unificada, sin contadores dram) los bytes que fallan en L2; **espera principal**: motivo con más ciclos de espera por instrucción emitida. ncu con relojes sin fijar (`--clock-control none`).*

##### NVIDIA GB10 (sm_121) — RMSNorm fp16

| DSL | kernel | µs | MHz | SM % | tensor % | memoria % | emisión % | ocupación % | ocup. teórica % | limita | regs/hilo | smem KB/bloque | acierto L2 % | memoria principal MB | conflictos smem | espera principal |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | `rmsnorm_kernel` | 2212.352 | 2502.926 | 4.68 | 0 | 12.52 | 4.69 | 88.61 | 100 | registros | 40 | 1.062 | 0.16 | 536.887 (fallos L2) | 67092 | long scoreboard |
| **Triton + TLX** | — *no compila*: triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes) | | | | | | | | | | | | | | | |
| **Gluon** | `rmsnorm_kernel` | 2189.792 | 2501.573 | 4.51 | 0 | 12.65 | 4.51 | 88.94 | 100 | registros | 40 | 1.062 | 0.16 | 536.887 (fallos L2) | 67293 | long scoreboard |
| **Helion** | — *no medido*: sin perfil de ncu en esta GPU | | | | | | | | | | | | | | | |
| **CUTLASS** | — *no medido*: sin perfil de ncu en esta GPU | | | | | | | | | | | | | | | |
| **Referencia** | `at::<unnamed>::vectorized_layer_norm_k…` | 2461.152 | 2504.352 | 5.49 | 0 | 14.57 | 4.66 | 95.53 | 100 | registros | 40 | 1.023 | 30.04 | 573.799 (fallos L2) | 4734 | long scoreboard |

*Tamaño: 16384×8192.*

##### NVIDIA GB10 (sm_121) — RMSNorm bf16

| DSL | kernel | µs | MHz | SM % | tensor % | memoria % | emisión % | ocupación % | ocup. teórica % | limita | regs/hilo | smem KB/bloque | acierto L2 % | memoria principal MB | conflictos smem | espera principal |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | `rmsnorm_kernel` | 2217.728 | 2504.974 | 5.06 | 0 | 12.49 | 5.1 | 89.03 | 100 | registros | 39 | 1.062 | 0.16 | 536.887 (fallos L2) | 64710 | long scoreboard |
| **Triton + TLX** | — *no compila*: triton-utlx 3.8.0.post1 genera IR inválido (operandSegmentSizes) | | | | | | | | | | | | | | | |
| **Gluon** | `rmsnorm_kernel` | 2195.232 | 2503.573 | 4.89 | 0 | 12.63 | 4.9 | 88.85 | 100 | registros | 39 | 1.062 | 0.16 | 536.887 (fallos L2) | 64806 | long scoreboard |
| **Helion** | — *no medido*: sin perfil de ncu en esta GPU | | | | | | | | | | | | | | | |
| **CUTLASS** | — *no medido*: sin perfil de ncu en esta GPU | | | | | | | | | | | | | | | |
| **Referencia** | `at::<unnamed>::vectorized_layer_norm_k…` | 2486.848 | 2436.464 | 5.59 | 0 | 14.66 | 5.04 | 96.91 | 100 | registros | 38 | 1.023 | 29.78 | 573.129 (fallos L2) | 5361 | long scoreboard |

*Tamaño: 16384×8192.*

### 5.4 Vista por arquitectura

*Barras: rendimiento de cada DSL; encima, % del pico; línea: pico de referencia; los huecos llevan «✕ (n)» y su motivo debajo.*

![Compute-bound — NVIDIA GB10 (sm_121)](resultados/figuras/resumen_compute_bound_nvidia_gb10_sm_121.png)

![Memory-bound — NVIDIA GB10 (sm_121)](resultados/figuras/resumen_memory_bound_nvidia_gb10_sm_121.png)

![Compute-bound — NVIDIA GeForce RTX 5090 (sm_120)](resultados/figuras/resumen_compute_bound_nvidia_geforce_rtx_5090_sm_120.png)

![Memory-bound — NVIDIA GeForce RTX 5090 (sm_120)](resultados/figuras/resumen_memory_bound_nvidia_geforce_rtx_5090_sm_120.png)

<a id="pico-mma"></a>

## 6. Pico de `mma.sync`: el techo del hardware

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

<a id="tecnicas"></a>

## 7. Técnicas de optimización del matmul en GB10

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

<a id="tma"></a>

### 7.1 TMA: se activa pero no acelera


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

<a id="fp8"></a>

### 7.2 FP8: la palanca que rompe el techo


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

<a id="sparsity"></a>

### 7.3 Sparsity 2:4: regresión en este stack


- Triton no admite 2:4 en `tl.dot`; se usó PyTorch (`to_sparse_semi_structured`, cuSPARSELt).
- El 2:4 rinde el **34–62 % del denso**: cuSPARSELt despacha a `sm80_xmma_sparse_gemm` (Ampère),
  mientras el denso usa un kernel nativo `nvjet_sm121_..._tmaAB`. FP8 + 2:4 no está soportado por
  la API. La cifra de catálogo (sparse + baja precisión) no es reproducible con este stack.

<a id="energia"></a>

## 8. Eficiencia energética

### Cómo se calcula

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

### NVIDIA GB10 (sm_121)

| DSL | Matmul fp16 (CB) | Matmul FP8 (CB) | FlashAttention fp16 (CB) | FlashAttention causal fp16 (CB) | RMSNorm fp16 (MB) | RMSNorm bf16 (MB) |
|:---|:---|:---|:---|:---|:---|:---|
| **Triton** | **1019.6 GFLOP/J (×0.97 ref.)**<br>88.4 W (reposo 16.9 W) · 1260.6 GFLOP/J dinámica · 90.13 TFLOP/s sost. (73 % del pico) · 2158 MHz · 1078 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | **1585.8 GFLOP/J (×0.78 ref.)**<br>90.1 W (reposo 16.1 W) · 1930.8 GFLOP/J dinámica · 142.88 TFLOP/s sost. (59 % del pico) · 2138 MHz · 693.4 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | **913.6 GFLOP/J (×1.08 ref.)**<br>94.7 W (reposo 16.6 W) · 1107.8 GFLOP/J dinámica · 86.52 TFLOP/s sost. (70 % del pico) · 2171 MHz · 1204 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **853.6 GFLOP/J (×1 ref.)**<br>92.2 W (reposo 16.6 W) · 1041 GFLOP/J dinámica · 78.7 TFLOP/s sost. (64 % del pico) · 2203 MHz · 644.1 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **6.23 GB/J (×1.05 ref.)**<br>39.2 W (reposo 14.9 W) · 10.04 GB/J dinámica · 244 GB/s sost. (89 % del pico) · 2502 MHz · 86.24 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ | **6.27 GB/J (×1.04 ref.)**<br>38.9 W (reposo 14.9 W) · 10.16 GB/J dinámica · 243.9 GB/s sost. (89 % del pico) · 2483 MHz · 85.61 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ |
| **Triton (block-ptr)** | **1036 GFLOP/J (×0.99 ref.)**<br>87.9 W (reposo 17 W) · 1284.4 GFLOP/J dinámica · 91.07 TFLOP/s sost. (74 % del pico) · 2125 MHz · 1061 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | n/a | n/a | n/a | n/a | n/a |
| **Triton + TMA** | **1064.1 GFLOP/J (×1.01 ref.)**<br>88 W (reposo 17 W) · 1318.8 GFLOP/J dinámica · 93.64 TFLOP/s sost. (76 % del pico) · 2047 MHz · 1033 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | **1629.2 GFLOP/J (×0.8 ref.)**<br>90.4 W (reposo 16.1 W) · 1982.2 GFLOP/J dinámica · 147.28 TFLOP/s sost. (61 % del pico) · 2158 MHz · 674.9 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | n/a | n/a | n/a | n/a |
| **Gluon** | **860.6 GFLOP/J (×0.82 ref.)**<br>88.3 W (reposo 12.7 W) · 1005.1 GFLOP/J dinámica · 75.99 TFLOP/s sost. (62 % del pico) · 2158 MHz · 1278 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | — *sin kernel*: no hay matmul FP8 en este DSL | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | — *no compila*: el ejemplo oficial usa tcgen05: LLVM ERROR Cannot select tcgen05.wait.st | **6.37 GB/J (×1.07 ref.)**<br>38.3 W (reposo 14.3 W) · 10.16 GB/J dinámica · 243.9 GB/s sost. (89 % del pico) · 2502 MHz · 84.32 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ | **6.37 GB/J (×1.06 ref.)**<br>38.3 W (reposo 14.3 W) · 10.16 GB/J dinámica · 243.9 GB/s sost. (89 % del pico) · 2502 MHz · 84.32 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ |
| **Helion** | — *no medido*: sin medida de energía (TFM_ENERGIA=1) en esta GPU | — *sin kernel*: no hay matmul FP8 en este DSL | **875.2 GFLOP/J (×1.03 ref.)**<br>94.7 W (reposo 16.1 W) · 1054.5 GFLOP/J dinámica · 82.88 TFLOP/s sost. (67 % del pico) · 2132 MHz · 1256 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **425.2 GFLOP/J (×0.5 ref.)**<br>94 W (reposo 16.1 W) · 513.1 GFLOP/J dinámica · 39.97 TFLOP/s sost. (33 % del pico) · 2164 MHz · 1293 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **6.5 GB/J (×1.09 ref.)**<br>38.3 W (reposo 15.8 W) · 11.07 GB/J dinámica · 249 GB/s sost. (91 % del pico) · 2483 MHz · 82.59 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ | **6.47 GB/J (×1.07 ref.)**<br>38.5 W (reposo 15.8 W) · 10.97 GB/J dinámica · 249.1 GB/s sost. (91 % del pico) · 2483 MHz · 82.99 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ |
| **CUTLASS** | **573.2 GFLOP/J (×0.55 ref.)**<br>42.7 W (reposo 11.9 W) · 794.6 GFLOP/J dinámica · 24.47 TFLOP/s sost. (20 % del pico) · 2427.5 MHz · 1918 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | — *sin kernel*: el dense_gemm de CuTe solo admite fp16/bf16 (su FP8 en GeForce es MXFP8) | **930.3 GFLOP/J (×1.1 ref.)**<br>94.3 W (reposo 14.7 W) · 1102.1 GFLOP/J dinámica · 87.73 TFLOP/s sost. (71 % del pico) · 2125 MHz · 1182 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **899.8 GFLOP/J (×1.06 ref.)**<br>92.1 W (reposo 14.7 W) · 1070.7 GFLOP/J dinámica · 82.87 TFLOP/s sost. (67 % del pico) · 2119 MHz · 611 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **6 GB/J (×1.01 ref.)**<br>41.4 W (reposo 13.7 W) · 8.97 GB/J dinámica · 248.6 GB/s sost. (91 % del pico) · 2502 MHz · 89.41 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ | **4.68 GB/J (×0.78 ref.)**<br>49 W (reposo 13.7 W) · 6.5 GB/J dinámica · 229.3 GB/s sost. (84 % del pico) · 2502 MHz · 114.7 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ |
| **Referencia** | **1051.1 GFLOP/J**<br>87.6 W (reposo 16.9 W) · 1302.4 GFLOP/J dinámica · 92.08 TFLOP/s sost. (75 % del pico) · 2086 MHz · 1046 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | **2036.5 GFLOP/J**<br>89.5 W (reposo 16.1 W) · 2483.2 GFLOP/J dinámica · 182.27 TFLOP/s sost. (75 % del pico) · 2112 MHz · 539.9 mJ/llamada · 8192×8192×8192<br>_hennessy-energia, job 20849_ | **846.6 GFLOP/J**<br>94.6 W (reposo 16.6 W) · 1026.8 GFLOP/J dinámica · 80.09 TFLOP/s sost. (65 % del pico) · 2216 MHz · 1299 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **851 GFLOP/J**<br>92.7 W (reposo 16.6 W) · 1036.6 GFLOP/J dinámica · 78.88 TFLOP/s sost. (64 % del pico) · 2216 MHz · 646 mJ/llamada · B=4 H=32 N=4096 D=128<br>_hennessy-energia, job 20850_ | **5.95 GB/J**<br>36.5 W (reposo 14.9 W) · 10.06 GB/J dinámica · 217.2 GB/s sost. (80 % del pico) · 2437 MHz · 90.23 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ | **6.02 GB/J**<br>36.2 W (reposo 14.9 W) · 10.23 GB/J dinámica · 217.8 GB/s sost. (80 % del pico) · 2431 MHz · 89.22 mJ/llamada · 16384×8192<br>_hennessy-energia, job 20851_ |

![Energía compute-bound — NVIDIA GB10 (sm_121)](resultados/figuras/energia_compute_bound_nvidia_gb10_sm_121.png)

![Energía memory-bound — NVIDIA GB10 (sm_121)](resultados/figuras/energia_memory_bound_nvidia_gb10_sm_121.png)

<a id="productividad"></a>

## 9. Productividad frente a rendimiento

**Qué se mide** (`benchmarks/productividad.py`, `TFM_CACHE_FRIA=1`, una forma por algoritmo en
fp16 en GB10: matmul 8192³, FlashAttention B=4 H=32 N=4096 D=128, RMSNorm 16384×8192):

1. **Líneas de código lógicas** (sin blancos, comentarios ni docstrings) del kernel, de la envoltura
   propia y del ajuste propio. Si se carga un ejemplo oficial de fuera, se cuenta ese código en la
   versión fijada, solo la parte del algoritmo.
2. **Coste del ajuste**: modo de ajuste, tamaño del espacio de configuraciones y tiempo de la primera
   llamada (compilación + autotuning + primera ejecución) con **todas las cachés vacías**.
3. **Rendimiento** de esa misma ejecución, como % del pico.

![Productividad frente a rendimiento](resultados/figuras/productividad.png)

| algoritmo | DSL | LoC kernel | LoC propias | LoC total | ajuste | configs | 1a llamada en frío | rendimiento | % del pico |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Matmul fp16 | Triton | 67 | 67 | 67 | autotune (triton.autotune) | 64 | 34.1 s | 92.96 TFLOP/s | 75.7 |
| Matmul fp16 | Triton + TMA | 68 | 68 | 68 | autotune (triton.autotune) | 128 | 96.1 s | 93.31 TFLOP/s | 76.0 |
| Matmul fp16 | Gluon | 85 | 101 | 101 | barrido propio (sin autotuning) | 7 | 3.9 s | 79.26 TFLOP/s | 64.5 |
| Matmul fp16 | Helion | 25 | 25 | 25 | autotune Helion (quick) | 251 | 4.0 min | 61.94 TFLOP/s | 50.4 |
| Matmul fp16 | CUTLASS | 672 | 41 | 713 | fija (tile del ejemplo) | 1 | 1.0 s | 23.59 TFLOP/s | 19.2 |
| FlashAttention fp16 | Triton | 220 | 220 | 220 | autotune (triton.autotune) | 36 | 15.2 s | 88.97 TFLOP/s | 72.4 |
| FlashAttention fp16 | Helion | 45 | 18 | 63 | autotune Helion (quick) | 135 | 2.1 min | 84.76 TFLOP/s | 69.0 |
| FlashAttention fp16 | CUTLASS | 713 | 65 | 778 | barrido propio (sin autotuning) | 6 | 7.7 s | 90.92 TFLOP/s | 74.0 |
| RMSNorm fp16 | Triton | 38 | 38 | 38 | heurística (warps por N) | 1 | 0.6 s | 242.3 GB/s | 88.8 |
| RMSNorm fp16 | Triton + TLX | 52 | 52 | 52 | heurística + --stages a mano | 1 | — | no compila en GB10 (triton-utlx 3.8.0.post1, ver la sección de procedencia) | — |
| RMSNorm fp16 | Gluon | 35 | 35 | 35 | heurística (warps y layout por N) | 1 | 0.6 s | 242.1 GB/s | 88.7 |
| RMSNorm fp16 | Helion | 16 | 16 | 16 | autotune Helion (quick) | 134 | 77.3 s | 247.2 GB/s | 90.5 |
| RMSNorm fp16 | CUTLASS | 135 | 68 | 203 | fija (heurística de CUTLASS) | 1 | 55.5 s | 247 GB/s | 90.5 |

*Entorno en frío `hennessy-frio` (jobs 20841, 20842, 20843); pico de mma.sync f16→f32 = 122.83 TFLOP/s (`hennessy`/run_pico_mma). LoC = líneas lógicas sin blancos, comentarios ni docstrings.*


**Lectura.** Helion es el más conciso (16–45 líneas, de 2 a 4 veces menos que Triton) pero busca
134–251 configuraciones y tarda de 1.3 a 4 min por forma (hasta 31 min en atención con N = 16384);
iguala al resto en RMSNorm, se acerca en atención y queda atrás en matmul. Triton es el equilibrio:
38–220 líneas, autotuning de 15–96 s y el mejor o casi el mejor rendimiento. CUTLASS/CuTe exige un
orden de magnitud más de código (670–710 líneas el kernel oficial) y su matmul oficial rinde el 19 %
en GB10. Gluon, sin autotuner, cuesta más que Triton y rinde menos en matmul. El tiempo de ajuste
es una dimensión propia: de 0.6 s a 4 min por forma con rendimientos parecidos. Limitaciones: las
LoC miden tamaño, no esfuerzo; una sola forma y una sola ejecución por algoritmo.

<a id="incidencias"></a>

## 10. Incidencias y validez de los datos

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

<a id="conclusiones"></a>


<a id="fuentes"></a>

## 13. Fuentes

| # | proyecto | versión | licencia |
|:---|:---|:---|:---|
| [1] | Triton, `triton-lang/triton`, `python/tutorials/` | tag v3.8.0 (`c01b677`) | MIT |
| [2] | CUTLASS, `NVIDIA/cutlass` | tag v4.8.0 (`098de2a`) | BSD-3-Clause |
| [3] | Helion, `pytorch/helion`, `examples/rms_norm.py`, `examples/attention.py` | `main` (`0f3b242`, 2026-09-30) | BSD-3-Clause |
| [4] | µTLX, `triton-lang/triton-ext`, `extensions/utlx` (paquete `triton-utlx` 3.8.0.post1) | `main` (`3c2bb3c`) | MIT |
| [5] | T. Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, arXiv:2307.08691 (ICLR 2024) | — | — |

Las cuatro licencias permiten reutilizar y modificar el código conservando el aviso de copyright.
