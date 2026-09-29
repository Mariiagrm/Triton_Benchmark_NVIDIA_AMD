# tfm_entorno

Entorno de experimentos del TFM (kernels Triton / Gluon / TLX / Helion) en el nodo
**hennessy** (aarch64, NVIDIA GB10). Todo se lanza con un único ejecutor: `ejecutar.sh`.

## Resultados principales

Optimización de matmul en tensor cores de **GB10 (sm_121)**; objetivo: superar la barrera
de ~100 TFLOP/s del FP16 denso. Resumen completo en [docs/resumen.md](docs/resumen.md)
(detalle por técnica en [docs/tma.md](docs/tma.md), [docs/fp8.md](docs/fp8.md),
[docs/sparsity.md](docs/sparsity.md)).

| técnica | TFLOP/s (8192³) | vs FP16 | ¿sube el techo? | por qué |
|:---|---:|---:|:---:|:---|
| FP16 baseline (Triton) | ~92 | — | referencia | límite = MMA, no memoria |
| TMA en FP16 | ~93 | +0 % | ❌ | activa `cp.async.bulk.tensor` pero no es memory-bound |
| **FP8 denso (Triton)** | **150** | **+62 %** | ✅ | MMA `m16n8k32.e4m3`; error 3.75 % |
| **FP8 denso (cuBLASLt)** | **192** | **+108 %** | ✅ | `torch._scaled_mm` |
| FP8 + TMA (Triton) | 150 | +62 % | ❌ (vs FP8) | hueco vs cuBLAS = scheduling, no datos |
| Sparsity 2:4 (cuSPARSELt) | 32 | −65 % | ❌ | cae a kernel `sm80_xmma_sparse` (Ampère) |

**Conclusión:** en este stack (torch 2.9 / triton 3.8 / cuSPARSELt), la única palanca que
sube el techo es **FP8 denso**. TMA y sparsity 2:4 —las técnicas "de catálogo"— no rinden
en GB10 (TMA porque el matmul no es memory-bound; 2:4 por falta de kernel sparse nativo
sm_121; y FP8+2:4 no está soportado, así que los ~838 TFLOPS del catálogo no son
reproducibles aquí).

## Uso

Desde el login (ibsen) o desde hennessy:

```bash
sbatch ~/hennessy/tfm_entorno/ejecutar.sh imagen                        # construir + validar la imagen
sbatch ~/hennessy/tfm_entorno/ejecutar.sh validar                       # solo validar
sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp experimento_cero          # lanzar un experimento
sbatch -J cero-bf16 ~/hennessy/tfm_entorno/ejecutar.sh exp experimento_cero --dtype bf16
VALIDAR=0 sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp experimento_cero  # sin validacion previa
```

Dentro de una sesión interactiva (`./srun_hennessy.sh`), sin cola:

```bash
~/tfm_entorno/ejecutar.sh exp experimento_cero --sizes 1048576
~/tfm_entorno/ejecutar.sh shell      # bash dentro del contenedor
```

- Log: `logs/<nombre-job>-<JOBID>.out`
- Resultados: `results/<experimento>/<fecha>_job<JOBID>/{resultados.csv,meta.json}`,
  y `results/<experimento>/ultimo` apunta a la ejecución más reciente.

## Cómo funciona

- Imagen `tfm:ngc-arm64` construida desde `Dockerfile` (NGC PyTorch 25.10 + TLX + Helion).
  `exp` la construye si no existe (`REBUILD=1` para forzarlo).
- Cada ejecución usa un contenedor **efímero** (`docker run --rm`) con este directorio
  montado en `/workspace/tfm`: no hace falta reconstruir al cambiar código.
- `exp` avisa si la GPU está ocupada y, por defecto, ejecuta antes `src/validar_gpu.py`.
- Caché de Triton persistente en `.cache/triton`.

## Añadir un experimento

1. Copiar `src/experimento_cero.py` como `src/<nombre>.py`.
2. Kernels a nivel de módulo; validar contra la referencia con `torch.testing.assert_close`.
   **Todo cálculo matricial va en tensor cores** (`tl.dot`): llamar a
   `comun.exigir_tensor_cores(kernel_compilado)`, que aborta si el PTX no tiene
   `mma.sync`/`wgmma`/`tcgen05.mma`. Importar `comun` activa TF32 para fp32 en cuBLAS.
3. Medir con `comun.medir(fn, flops=..., bytes_movidos=...)` y terminar con
   `comun.guardar(filas, parametros=vars(args), resumen=...)`.
4. `sbatch ~/hennessy/tfm_entorno/ejecutar.sh exp <nombre> [args]`.

## Informes para la memoria

**Todo experimento** genera al terminar `tabla.md`, `tabla.tex`, `grafica.png` y `grafica.pdf`
en su carpeta de resultados, y copia la última a `docs/resultados/<experimento>.{md,tex}` (+
`figuras/`): lo hace `comun.guardar()` automáticamente vía `src/informe.py` (informe genérico).
`baseline_matmul` usa su propio informe (`informe_baseline_matmul.py`, gráfica Triton vs cuBLAS).

Regenerar cualquier ejecución (en el login, sin GPU):

```bash
python3.11 src/informe.py results/<experimento>/ultimo
```

Las secciones redactadas para la memoria del TFM están en `docs/*.md` y `docs/*.tex`
(includables con `\input`): `resumen`, `tma`, `fp8`, `sparsity`.

## Estructura

```
ejecutar.sh           ejecutor estándar (imagen / validar / exp / shell)
Dockerfile            imagen del entorno
src/comun.py          contexto, medida, guardado y detección PTX (MMA/TMA) comunes
src/informe.py        informe genérico (tabla .md/.tex + gráfica .png/.pdf) de cualquier experimento
src/validar_gpu.py    validación PyTorch/Triton/TLX/Helion en GPU
src/experimento_cero.py       plantilla: GEMM en tensor cores (Triton fijo) vs cuBLAS
src/baseline_matmul.py        baseline: matmul Triton autotune vs cuBLAS
src/informe_baseline_matmul.py  informe propio del baseline (Triton vs cuBLAS)
src/experimento_tma.py        baseline vs block-pointers vs descriptores TMA
src/experimento_fp8.py        FP16 vs FP8 (Triton) vs FP8+TMA vs FP8 cuBLASLt
src/experimento_sparsity.py   denso vs 2:4 sparse (cuSPARSELt)
src/triton_kernels/   matmul.py (baseline), matmul_blockptr.py, matmul_tma.py,
                      matmul_fp8.py, matmul_fp8_tma.py
src/gluon_kernels/    kernels Gluon
docs/                 secciones de la memoria: resumen/tma/fp8/sparsity (.md y .tex) + resultados/
results/  logs/  analysis/
```
