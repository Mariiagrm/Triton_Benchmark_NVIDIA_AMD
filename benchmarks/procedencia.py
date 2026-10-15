"""Procedencia de cada kernel del TFM (fuente unica; antes docs/TFM/procedencia_kernels.md).

Texto escrito a mano, con el formato de tablas Markdown que lee benchmarks/kernels.py
(leer_procedencia) y que docs/TFM/comparativa.md reproduce (tablas, notas por kernel y fuentes).
Para cambiar la procedencia de un kernel, editar TEXTO.
"""

TEXTO = r"""# Procedencia de los kernels

*Origen de cada kernel del TFM: si es código oficial sin modificar, una adaptación de un ejemplo público o una implementación propia sobre una API documentada. La procedencia se ha verificado comparando el código con la fuente citada, en la versión indicada.*

**Tipo de elaboración:**

- **Oficial sin modificar**: el algoritmo es el de la fuente, que se llama tal cual; solo se añade la envoltura necesaria para medirlo.
- **Adaptado**: parte de un ejemplo público, con cambios que se detallan.
- **Propio**: diseño propio, escrito a partir de la documentación y los ejemplos de la API.

## Kernels compute-bound (`src/CBKernels/`)

| kernel | DSL | elaboración | fuente |
|:---|:---|:---|:---|
| `triton/matmul.py` (baseline) | Triton | Adaptado | Tutorial oficial de Triton `03-matrix-multiplication.py` [1] |
| `triton/matmul_blockptr.py` | Triton | Adaptado | Esqueleto de `matmul.py` + API `tl.make_block_ptr` / `tl.advance` de Triton [1] |
| `triton/matmul_tma.py` | Triton | Adaptado | Tutorial oficial de Triton `09-persistent-matmul.py`, variantes con TMA [1] |
| `triton/matmul_fp8.py` | Triton | Adaptado | Esqueleto de `matmul.py`; el tutorial 03 ya muestra entradas FP8 [1] |
| `triton/matmul_fp8_tma.py` | Triton | Propio (combinación) | `matmul_fp8.py` + descriptores TMA de `matmul_tma.py` |
| `gluon/matmul.py` | Gluon | Propio | Piezas oficiales de Triton v3.8.0: `mma_v2` (`test_core.py::test_mma_v2`) y pipeline `cp.async` (`gluon/03-async-copy.py`) [1] |
| `helion/matmul.py` | Helion | Adaptado | Ejemplo oficial de Helion `examples/matmul.py` (`matmul`) [3] |
| `cutlass/matmul_cute.py` | CUTLASS (CuTe DSL) | Oficial sin modificar | `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py` (`Sm120GemmKernel`) [2] |

**`matmul.py`.** 27 de las 36 líneas distintas del kernel son idénticas a `matmul_kernel` del tutorial 03. Coinciden el orden de programas agrupado para la L2 (`GROUP_SIZE_M`), la aritmética de punteros y el bucle en K con `tl.dot` y acumulación en fp32. Es propio el espacio de autotuning (producto cartesiano de BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps y num_stages). También lo es la comprobación en el PTX de que se usan tensor cores (`comun.exigir_tensor_cores`).

**`matmul_blockptr.py`.** Es el mismo kernel con las cargas expresadas con *block pointers*. Se conserva como intento documentado: en sm_121 no activa TMA, sino `cp.async` clásico (§ [TMA](tma.md)).

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

## Kernels compute-bound (`src/CBKernels/`): FlashAttention (forward)

Solo se usan implementaciones **oficiales** de cada proyecto. Antes de medir se comprobó en
las dos GPUs (RTX 5090 sm_120 y GB10 sm_121) cuáles se ejecutan y dan el mismo resultado que
`torch.nn.functional.scaled_dot_product_attention` (atol 1e-2): jobs 20639 (pascal) y 20640
(hennessy), con D ∈ {64, 128}, causal y no causal.

| kernel | DSL | elaboración | fuente | ¿corre en sm_120/121? |
|:---|:---|:---|:---|:---|
| `triton/fused_attention.py` | Triton | Oficial sin modificar (copia literal) | Tutorial oficial `06-fused-attention.py` [1] | ✅ |
| `helion/attention.py` → `examples/attention.py` | Helion | Oficial sin modificar | `attention_output` / `causal_attention_output` [3] | ✅ |
| `cutlass/attention_fa2.py` → `flash_attention_v2.py` | CUTLASS (CuTe DSL) | Oficial sin modificar | `examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py` (`FlashAttentionForwardAmpere`) [2] | ✅ |
| — | Gluon | no hay oficial ejecutable | `python/examples/gluon/01-attention-forward.py` [1] usa `tcgen05` | ❌ `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st` |
| — | Triton + TLX | no hay oficial ejecutable | tutoriales `tlx/tutorials/{hopper,blackwell}_fa_ws*.py` [4] | ❌ ver abajo |

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

## Kernels memory-bound (`src/MBKernels/`): RMSNorm

| kernel | DSL | elaboración | fuente |
|:---|:---|:---|:---|
| `triton/rmsnorm_baseline.py` | Triton | Adaptado | Tutorial oficial de Triton `05-layer-norm.py` (`_layer_norm_fwd_fused`) [1] |
| `triton_tlx/rmsnorm.py` | Triton + TLX | Propio | API de µTLX (`triton-lang/triton-ext`, `extensions/utlx`) [4] |
| `gluon/rmsnorm.py` | Gluon | Propio | API de Gluon, tutorial oficial `gluon/02-layouts.py` [1] |
| `helion/rmsnorm.py` | Helion | Adaptado | Ejemplo oficial de Helion `examples/rms_norm.py` (`rms_norm_fwd`) [3] |
| `cutlass/rmsnorm_ext.cu` | CUTLASS (C++/CUDA) | Oficial sin modificar | `cutlass::rmsnorm`, `tools/util/include/cutlass/util/device_rmsnorm.h` [2] |

**`triton/rmsnorm_baseline.py`.** Adapta el LayerNorm del tutorial 05: un programa por fila, fila entera en registros, `MAX_FUSED_SIZE = 65536` y la heurística de warps `BLOCK_SIZE // 256`. El tutorial limita a 8 warps y aquí se llega a 16. Se elimina la resta de la media, porque RMSNorm no centra, y la reducción es de x² en fp32.

**`triton_tlx/rmsnorm.py`.** Es un diseño propio: kernel persistente con *prefetch* de filas a memoria compartida en un anillo de `NUM_STAGES` buffers (`cp.async`). Las llamadas (`tlx.local_alloc`, `local_view`, `async_load`, `async_load_commit_group` / `wait_group`, `local_load`) siguen el patrón de `test/test_tlx.py::test_local_load` y del tutorial `blackwell-multi-cta-layernorm` de triton-ext. Con triton-utlx 3.8.0.post1 **no compila**: el plugin crea `ttg.async_copy_global_to_local` sin `operandSegmentSizes` y el verificador de MLIR lo rechaza. Falla incluso el ejemplo mínimo del propio repositorio, así que no es un error del kernel.

**`gluon/rmsnorm.py`.** Mismo algoritmo que el baseline Triton, pero con el reparto de la fila entre hilos fijado a mano mediante un `BlockedLayout` explícito (16 B por hilo, para cargas vectorizadas). La API sigue el tutorial de layouts de Gluon de Triton v3.8.0.

**`helion/rmsnorm.py`.** Es el `rms_norm_fwd` del ejemplo oficial sin la salida auxiliar `inv_rms`, que solo necesita el *backward*. Así mueve los mismos bytes que las demás implementaciones. El kernel Triton lo genera y autotunea Helion.

**`cutlass/rmsnorm_ext.cu`.** Llama a la función oficial `cutlass::rmsnorm` **sin modificarla**. En fp16 con N % 8 == 0 usa el kernel vectorizado con `float4` (`rmsnorm_twoPassAlgo_e8`); en bf16 usa el escalar (`_e1`). Solo es propia la envoltura como extensión de PyTorch (`torch.utils.cpp_extension`), para medirlo con el mismo banco que el resto.

## Microbenchmark de hardware (`src/Microbench/`)

| fichero | qué mide | elaboración |
|:---|:---|:---|
| `pico_mma_ext.cu` + `pico_mma.py` | pico de cada instrucción `mma.sync` (§ [pico_mma.md](pico_mma.md)) | **Propio** |

**`pico_mma_ext.cu`.** No es un kernel de ningún DSL, sino código de caracterización del hardware: cadenas de `mma.sync` en PTX en línea sobre registros, sin accesos a memoria, con ILP acumuladores independientes por warp. Usa la técnica habitual de los microbenchmarks de *throughput* de instrucciones: dependencias controladas, un único volcado final para que el compilador no elimine el bucle y barrido de la ocupación. La sintaxis de cada instrucción (incluidas las *block-scaled* `kind::mxf8f6f4` y `kind::mxf4nvf4`, con sus operandos de escala) sigue la PTX ISA de NVIDIA. Que sea correcta lo confirman ptxas, que compila cada instrucción por separado para `sm_121a`, y el SASS (`cuobjdump`), que muestra la instrucción de tensor core esperada (`HMMA`, `QMMA`, `IMMA`, `OMMA`). `pico_mma.py` lo compila con nvcc a una biblioteca compartida y lo carga con ctypes.

## Referencias de comparación (no son kernels propios)

| referencia | se usa en | qué ejecuta |
|:---|:---|:---|
| `torch.matmul` | `run_matmul*` | cuBLAS (fp16/bf16; fp32 en TF32) |
| `torch._scaled_mm` | `run_matmul_fp8` | cuBLASLt en FP8 |
| `torch.nn.functional.scaled_dot_product_attention` | `run_attention_*` | FlashAttention-2 incluido en PyTorch (`pytorch_flash::flash_fwd_kernel`); también es la referencia de corrección |
| `torch.nn.functional.rms_norm` | `run_rmsnorm_*` | kernel RMSNorm de PyTorch; también es la referencia de corrección |

## Fuentes

| # | proyecto | versión | licencia |
|:---|:---|:---|:---|
| [1] | Triton, `triton-lang/triton`, `python/tutorials/` | tag v3.8.0 (`c01b677`) | MIT |
| [2] | CUTLASS, `NVIDIA/cutlass` | tag v4.8.0 (`098de2a`) | BSD-3-Clause |
| [3] | Helion, `pytorch/helion`, `examples/rms_norm.py`, `examples/attention.py` | `main` (`0f3b242`, 2026-09-30) | BSD-3-Clause |
| [4] | µTLX, `triton-lang/triton-ext`, `extensions/utlx` (paquete `triton-utlx` 3.8.0.post1) | `main` (`3c2bb3c`) | MIT |
| [5] | T. Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, arXiv:2307.08691 (ICLR 2024) | — | — |

Las cuatro licencias permiten reutilizar y modificar el código conservando el aviso de copyright.
"""
