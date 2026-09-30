# Resumen: optimización de matmul en tensor cores (GB10)

*Resumen consolidado de todos los experimentos del TFM. Hardware: **NVIDIA GB10 (sm_121)**, memoria LPDDR5X unificada; torch 2.9, triton 3.8.0, CUDA 13.0. Metodología: `triton.testing.do_bench` (mediana); tensor cores verificados en el PTX. Detalle por técnica en `docs/TFM/{tma,fp8,sparsity}.md`.*

## Objetivo

Partir del matmul denso en Triton (baseline, ~90–100 TFLOP/s en FP16) e intentar **superar la barrera de los ~100 TFLOP/s** acercándose a los límites del silicio, aplicando las técnicas específicas de la arquitectura Blackwell/Hopper: TMA, pipelining, swizzling L2, FP8 y sparsity 2:4.

## Tabla resumen (matmul 8192³)

| técnica | TFLOP/s | vs FP16 | ¿sube el techo? | evidencia clave |
|:---|---:|---:|:---:|:---|
| **FP16 baseline** (Triton) | ~92 | — | referencia | `mma.sync.m16n8k16`, `carga=cp.async` |
| TMA en FP16 (descriptores) | ~93 | +0 % | ❌ | activa `cp.async.bulk.tensor` pero no acelera |
| **FP8 denso** (Triton) | **150** | **+62 %** | ✅ | `mma.sync.m16n8k32.e4m3`, error 3.75 % |
| FP8 denso (cuBLASLt) | **192** | **+108 %** | ✅ | `torch._scaled_mm` |
| FP8 + TMA (Triton) | 150 | +62 % | ❌ (vs FP8) | TMA real, solo +1.5 % sobre FP8 |
| Sparsity 2:4 (cuSPARSELt) | 32 | **−65 %** | ❌ | kernel `sm80_xmma_sparse` (Ampère) |

## Qué se descubrió, técnica a técnica

### 1. El baseline está limitado por la MMA, no por la memoria
El matmul denso FP16 se estanca en ~90–100 TFLOP/s. El ancho de banda efectivo es bajísimo (~34 GB/s en 8192³) porque los datos se reutilizan en SRAM: **no es memory-bound**. El cuello es la unidad de cómputo (MMA).

### 2. TMA: se activa pero no acelera → *ver [tma.md](tma.md)*
- `tl.make_block_ptr` **no** activa TMA (baja a `cp.async`) y está **deprecado** en Triton 3.8.
- `tl.make_tensor_descriptor` **sí** activa TMA (`cp.async.bulk.tensor`, verificado en PTX).
- **Pero no da aceleración** (~0 %): coherente con que el matmul denso no sea memory-bound. *Lección: activar el acelerador de hardware correctamente no implica ganar rendimiento si el cuello está en otro sitio.*
- En sm_121 Triton emite `mma.sync` (no `wgmma` ni `tcgen05.mma`): la MMA es la de sm_80.

### 3. FP8: la única palanca que rompe la barrera → *ver [fp8.md](fp8.md)*
- FP8 e4m3 alcanza **150 TFLOP/s** (Triton) y **192** (cuBLASLt), superando con holgura los ~100 de FP16.
- El throughput viene de la MMA: pasa a `m16n8k32` (K doble → doble ritmo).
- **Coste:** error relativo **3.75 %** (vs 0.02 % en FP16), acotado y estable.
- El kernel Triton llega a ~83 % de cuBLASLt; el hueco restante es de **scheduling** (kernels persistentes), no de datos: **TMA sobre FP8 solo aporta +1.5 %**.

### 4. Sparsity 2:4: regresión en este hardware/stack → *ver [sparsity.md](sparsity.md)*
- El 2:4 (cuSPARSELt vía PyTorch) rinde el **34–62 % del denso**: va **más lento**.
- Causa: cuSPARSELt despacha a `sm80_xmma_sparse_gemm` (**kernel de Ampère**), mientras el denso usa un kernel nativo `nvjet_sm121_..._tmaAB` (Blackwell + TMA). No hay kernel sparse nativo para sm_121.
- **FP8 + 2:4 no está soportado** por la API. La cifra de catálogo (~838 TFLOPS, sparse + baja precisión) **no es reproducible** con este stack.

## Conclusión

En GB10 con el *stack* actual (torch 2.9 / triton 3.8 / cuSPARSELt), la única técnica que **efectivamente sube el techo** de los tensor cores es la **reducción de precisión a FP8** (+62 % en Triton, hasta +108 % con cuBLASLt), a cambio de un error acotado del ~3.75 %. Las dos técnicas "de catálogo" —TMA y sparsity 2:4— **no rinden aquí**: TMA porque el matmul no es memory-bound, y 2:4 porque falta un kernel sparse nativo de Blackwell. El valor del trabajo está tanto en el resultado positivo (FP8) como en documentar, con evidencia de bajo nivel (PTX, nombres de kernel), *por qué* las abstracciones y técnicas anunciadas no siempre se traducen en aceleración real.

**Líneas abiertas:** (i) kernel FP8 **persistente** para arañar el ~17 % que separa de cuBLASLt; (ii) revisar 2:4 cuando el stack incorpore kernels sparse para sm_121; (iii) FP8+2:4 vía cuSPARSELt directo.

## Reproducibilidad

Cada experimento genera automáticamente tabla + gráfica (`docs/TFM/resultados/`) y guarda contexto (GPU, versiones, job de Slurm) en `meta.json`.

```bash
sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp run_matmul
sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp run_matmul_tma
sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp run_matmul_fp8
sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp run_matmul_sparsity
```
