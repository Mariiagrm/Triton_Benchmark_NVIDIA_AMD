# tfm_entorno

Entorno de experimentos del TFM (kernels Triton / Gluon / TLX / Helion) en el nodo
**hennessy** (aarch64, NVIDIA GB10). Todo se lanza con un único ejecutor: `ejecutar.sh`.

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

`baseline_matmul` genera al terminar `tabla.md`, `tabla.tex`, `grafica.png` y `grafica.pdf`
en su carpeta de resultados, y copia la última a `docs/resultados/`.
Para regenerarlo (en el login, sin GPU): `python3.11 src/informe_baseline_matmul.py [carpeta]`.

## Estructura

```
ejecutar.sh           ejecutor estándar (imagen / validar / exp / shell)
Dockerfile            imagen del entorno
src/comun.py          contexto, medida y guardado comunes
src/validar_gpu.py    validación PyTorch/Triton/TLX/Helion en GPU
src/experimento_cero.py  plantilla: GEMM en tensor cores (Triton fijo) vs cuBLAS
src/baseline_matmul.py   baseline: matmul Triton autotune vs cuBLAS
src/informe_baseline_matmul.py  tabla (.md/.tex) y grafica (.png/.pdf) del baseline
src/triton_kernels/      kernels Triton (matmul.py)
src/gluon_kernels/       kernels Gluon
results/  logs/  analysis/  docs/
```
