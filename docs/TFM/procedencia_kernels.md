# Procedencia de los kernels

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
| `helion/matmul.py` | Helion | Adaptado | Ejemplo oficial de Helion `examples/matmul.py` (`matmul`) [3] |
| `cutlass/matmul_cute.py` | CUTLASS (CuTe DSL) | Oficial sin modificar | `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py` (`Sm120GemmKernel`) [2] |

**`matmul.py`.** 27 de las 36 líneas distintas del kernel son idénticas a `matmul_kernel` del tutorial 03. Coinciden el orden de programas agrupado para la L2 (`GROUP_SIZE_M`), la aritmética de punteros y el bucle en K con `tl.dot` y acumulación en fp32. Es propio el espacio de autotuning (producto cartesiano de BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps y num_stages). También lo es la comprobación en el PTX de que se usan tensor cores (`comun.exigir_tensor_cores`).

**`matmul_blockptr.py`.** Es el mismo kernel con las cargas expresadas con *block pointers*. Se conserva como intento documentado: en sm_121 no activa TMA, sino `cp.async` clásico (§ [tma.md](tma.md)).

**`matmul_tma.py`.** Toma del tutorial 09 los descriptores de tensor creados dentro del kernel (`tl.make_tensor_descriptor`), las cargas y escrituras `desc.load` / `desc.store` y el *allocator* obligatorio (`triton.set_allocator`). A diferencia del tutorial, **no es persistente**: mantiene la rejilla y el orden agrupado del baseline para aislar el efecto de TMA.

**`matmul_fp8.py`.** Mismo esqueleto que el baseline, con A y B en FP8 e4m3 (`torch.float8_e4m3fn`). El tutorial 03 usa e5m2 como ejemplo.

**`matmul_fp8_tma.py`.** Une las dos variantes anteriores y amplía el espacio de autotuning (num_stages 3–5, BLOCK_K hasta 256).

**`helion/matmul.py`.** Es el `matmul` del ejemplo oficial con el mismo decorador (`static_shapes=True` y los `autotune_config_overrides` del ejemplo), pero sin el parámetro `epilogue`, que aquí es la identidad. El ejemplo tiene test en el CI de Helion (`test_examples.py::test_matmul`).

**`cutlass/matmul_cute.py`.** Carga el ejemplo oficial de CuTe DSL para Blackwell GeForce desde `$CUTLASS_DIR` **sin modificarlo**: fp16 con acumulación en fp32, TMA y pipeline de varias etapas. Solo la envoltura es propia: pasa las matrices de PyTorch como tensores CuTe sin copiarlas (A con K contigua, B y C con N contigua) y compila el kernel con `cute.compile`, siguiendo la función `run()` del propio ejemplo. El kernel no admite bf16. Paquete `nvidia-cutlass-dsl` 4.8.0, la misma versión que CUTLASS.

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

## Referencias de comparación (no son kernels propios)

| referencia | se usa en | qué ejecuta |
|:---|:---|:---|
| `torch.matmul` | `run_matmul*` | cuBLAS (fp16/bf16; fp32 en TF32) |
| `torch._scaled_mm` | `run_matmul_fp8` | cuBLASLt en FP8 |
| `torch.sparse.to_sparse_semi_structured` | `run_matmul_sparsity` | cuSPARSELt 2:4 (la poda 2:4 es propia) |
| `torch.nn.functional.rms_norm` | `run_rmsnorm_*` | kernel RMSNorm de PyTorch; también es la referencia de corrección |

## Fuentes

| # | proyecto | versión | licencia |
|:---|:---|:---|:---|
| [1] | Triton, `triton-lang/triton`, `python/tutorials/` | tag v3.8.0 (`c01b677`) | MIT |
| [2] | CUTLASS, `NVIDIA/cutlass` | tag v4.8.0 (`098de2a`) | BSD-3-Clause |
| [3] | Helion, `pytorch/helion`, `examples/rms_norm.py` | `main` (`0f3b242`, 2026-09-30) | BSD-3-Clause |
| [4] | µTLX, `triton-lang/triton-ext`, `extensions/utlx` (paquete `triton-utlx` 3.8.0.post1) | `main` (`3c2bb3c`) | MIT |

Las cuatro licencias permiten reutilizar y modificar el código conservando el aviso de copyright.
