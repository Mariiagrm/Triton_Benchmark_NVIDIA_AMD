# tfm_entorno

Benchmarks de kernels del TFM (Triton / TLX / Cutlass-CuTe / Gluon / Helion) en el nodo
**hennessy** (aarch64, NVIDIA GB10). Los kernels se dividen en **compute-bound** (`src/CBKernels/`,
p. ej. matmul) y **memory-bound** (`src/MBKernels/`, p. ej. RMSNorm). Todo se lanza con un
único ejecutor: `ejecutar.sh`.

## Resultados principales

Optimización de matmul en tensor cores de **GB10 (sm_121)**; objetivo: superar la barrera
de ~100 TFLOP/s del FP16 denso. Resumen completo en [docs/TFM/resumen.md](docs/TFM/resumen.md)
(detalle por técnica en [docs/TFM/tma.md](docs/TFM/tma.md), [docs/TFM/fp8.md](docs/TFM/fp8.md),
[docs/TFM/sparsity.md](docs/TFM/sparsity.md)).

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

## Estructura

```
tfm_entorno/
├── README.md             este fichero
├── requirements.txt      dependencias Python (en hennessy las instala el Dockerfile)
├── CMakeLists.txt        compilacion de kernels C++/CUDA (src/*Kernels/cutlass/*.cu)
├── Dockerfile            una imagen por DSL (targets triton-tlx / gluon / helion / cutlass)
├── ejecutar.sh           ejecutor estandar (imagen / validar / exp / shell)
│
├── src/                  codigo fuente de los kernels, un subpaquete por DSL
│   ├── CBKernels/        compute-bound (matmul)             imagen:
│   │   ├── triton/       Triton estandar: matmul.py, ...     tfm-triton-tlx
│   │   ├── triton_tlx/   Triton + extensiones TLX            tfm-triton-tlx
│   │   ├── cutlass/      Cutlass/CuTe (.cu, CMake)           tfm-cutlass
│   │   ├── gluon/        Gluon                               tfm-gluon
│   │   └── helion/       Helion                              tfm-helion
│   └── MBKernels/        memory-bound (RMSNorm, Softmax)
│       ├── triton/       rmsnorm_baseline.py
│       ├── gluon/        rmsnorm.py
│       ├── cutlass/      rmsnorm_ext.cu + rmsnorm.py (compila la extension con nvcc)
│       ├── helion/       rmsnorm.py
│       └── triton_tlx/   rmsnorm.py
│
├── benchmarks/           scripts que ejecutan los kernels y miden (misma division que src/)
│   ├── CBKernels/        compute-bound (TFLOP/s)
│   │   ├── triton/       Triton + TLX (imagen triton-tlx)
│   │   │   ├── run_matmul.py           matmul Triton (autotune tamanos/bloques/warps) vs cuBLAS
│   │   │   ├── run_matmul_tma.py       baseline vs block-pointers vs descriptores TMA
│   │   │   ├── run_matmul_fp8.py       FP16 vs FP8 (Triton) vs FP8+TMA vs FP8 cuBLASLt
│   │   │   ├── run_matmul_sparsity.py  denso vs 2:4 sparse (cuSPARSELt)
│   │   │   └── informe_matmul.py       informe propio de run_matmul (Triton vs cuBLAS)
│   │   ├── gluon/        (imagen gluon)
│   │   ├── cutlass/      (imagen cutlass)
│   │   └── helion/       (imagen helion)
│   ├── MBKernels/        memory-bound (GB/s): RMSNorm en cada DSL vs PyTorch
│   │   ├── triton/       run_rmsnorm_triton.py    kernel propio (punteros, una fila por programa)
│   │   │                 run_rmsnorm_tlx.py       TLX: persistente + prefetch cp.async a memoria compartida
│   │   ├── gluon/        run_rmsnorm_gluon.py     mismo algoritmo con layout explicito
│   │   ├── cutlass/      run_rmsnorm_cutlass.py   cutlass::rmsnorm oficial (extension de PyTorch)
│   │   └── helion/       run_rmsnorm_helion.py    ejemplo oficial de Helion (autotune)
│   ├── banco_rmsnorm.py        banco comun de RMSNorm (formas, validacion, medida, guardado)
│   ├── validation.py           verificacion de la salida frente a PyTorch
│   ├── plantilla.py            plantilla para un benchmark nuevo
│   ├── comun.py                contexto, medida, guardado y deteccion PTX (MMA/TMA)
│   ├── informe.py              tabla .md/.tex + grafica .png/.pdf + metricas consolidadas
│   └── validar_gpu.py          validacion de una imagen (base comun + su DSL) en GPU
│
├── results/              datos crudos y metricas
│   ├── matmul_metrics.csv      autogenerado: ultima ejecucion de cada run_matmul*
│   ├── rmsnorm_metrics.csv     autogenerado: ultima ejecucion de cada run_rmsnorm_* (todos los DSLs)
│   └── <benchmark>/<fecha>_job<JOBID>/   resultados.csv, meta.json, tabla.*, grafica.*
│                                         (<benchmark>/ultimo -> la mas reciente)
│
├── docs/TFM/             memoria: resumen/tma/fp8/sparsity (.md y .tex) + resultados/ (tablas y figuras)
└── logs/                 logs de Slurm (no versionados)
```

## Acceso al servidor (hennessy)

- Se entra al login **ibsen**; el home de hennessy está montado en
  `/machines/hennessy/home/mariag` (enlace `~/hennessy`), así que este repo se edita desde el
  login en `~/hennessy/tfm_entorno` y en el nodo es `~/tfm_entorno`.
- Docker y la GPU solo están disponibles **dentro** del nodo: se accede por Slurm
  (particiones `hennessy-benchmark` y `hennessy-test`).
  - En cola: `bash ~/hennessy/tfm_entorno/ejecutar.sh encolar ...`. `encolar` deduce el nodo de
    la ruta del repo (`/machines/<nodo>/...` en el login, o el nodo actual) y llama a `sbatch`
    con `-p <nodo>-benchmark` (o `PARTICION=...`), `--chdir` y el log en rutas del nodo.
    Desde el login hay que usar `bash ...`: `/machines/*/home` está montado con `noexec`.
  - En otro nodo (p. ej. patterson), con el repo clonado allí, es lo mismo:
    `bash /machines/patterson/home/mariag/<repo>/ejecutar.sh encolar ...` desde el login, o
    `./ejecutar.sh ...` directamente dentro del nodo.
  - Sesión interactiva: `~/srun_hennessy.sh` (en el login) abre una shell en hennessy;
    `~/srun_hennessy.sh cola` muestra la cola.

## Imágenes (una por DSL)

Cada DSL tiene su propia imagen Docker, todas definidas como *targets* del mismo `Dockerfile`
sobre una base común (NGC PyTorch 25.10: torch, Triton, CUDA, y las herramientas de análisis).
La base se construye una vez y la reutilizan las cuatro.

| DSL | imagen | añade sobre la base | kernels |
|:---|:---|:---|:---|
| Triton + TLX | `tfm-triton-tlx:ngc-arm64` | `triton-utlx` + `TRITON_PLUGIN_PATHS` | `src/*Kernels/triton/`, `triton_tlx/` |
| Gluon | `tfm-gluon:ngc-arm64` | nada (Gluon viene con Triton; sin plugin TLX) | `src/*Kernels/gluon/` |
| Helion | `tfm-helion:ngc-arm64` | Helion (`pip install -e`) | `src/*Kernels/helion/` |
| Cutlass/CuTe | `tfm-cutlass:ngc-arm64` | CUTLASS v4.8.0 en `/opt/cutlass` + CMake | `src/*Kernels/cutlass/` |

`ejecutar.sh validar <dsl>` comprueba en GPU la base (PyTorch, Triton, tensor cores) y el DSL de
esa imagen. `exp` elige la imagen por la carpeta del benchmark
(`benchmarks/<CB|MB>Kernels/<dsl>/`) o por la variable `DSL=`.

## Ejecución

Desde el login (ibsen) o desde hennessy:

```bash
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar imagen                   # construir + validar las 4 imagenes
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar imagen helion            # solo una
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar validar gluon            # solo validar
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul           # benchmark (imagen triton-tlx)
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp triton-tlx           # TODOS los benchmarks de un DSL
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_rmsnorm_gluon --dtypes bf16   # imagen gluon
bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul_fp8 --sizes 4096 8192
DSL=helion bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp benchmarks/otro.py   # imagen explicita
VALIDAR=0 bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_matmul  # sin validacion previa
```

Dentro de una sesión interactiva (`~/srun_hennessy.sh`), sin cola:

```bash
~/tfm_entorno/ejecutar.sh exp plantilla --sizes 1024
~/tfm_entorno/ejecutar.sh shell cutlass   # bash dentro del contenedor de un DSL
```

- Log: `logs/<nombre-job>-<JOBID>.out`
- Resultados: `results/<benchmark>/<fecha>_job<JOBID>/{resultados.csv,meta.json}` (el
  `meta.json` guarda el DSL, el id de la imagen, el driver NVIDIA y el kernel del nodo), `results/<benchmark>/ultimo` apunta a
  la ejecución más reciente, y `results/{matmul,rmsnorm}_metrics.csv` se regeneran con la
  última de cada benchmark.

## Compilación

- **Python (Triton, TLX, Gluon, Helion):** no hay paso de compilación; los kernels se compilan
  JIT al ejecutarse. `exp` construye la imagen si no existe (`REBUILD=1` para forzarlo).
  Caché de Triton persistente en `.cache/triton`.
- **C++/CUDA (Cutlass/CuTe):** dos formas, ambas dentro de la imagen cutlass (define `CUTLASS_DIR`):
  - `*_ext.cu`: extensiones de PyTorch que se compilan solas desde Python la primera vez que
    se usan (`torch.utils.cpp_extension`, caché en `.cache/torch_extensions/`); así se miden con
    el mismo banco que los kernels Python. Ej.: `src/MBKernels/cutlass/rmsnorm_ext.cu`.
  - El resto de `.cu` de `src/*Kernels/cutlass/` son programas independientes y se compilan
    con CMake (sm_121a por defecto):

  ```bash
  ~/tfm_entorno/ejecutar.sh shell cutlass
  cmake -S . -B build && cmake --build build -j
  ```

## Cómo funciona

- Cada ejecución usa un contenedor **efímero** (`docker run --rm`) con este directorio
  montado en `/workspace/tfm` (en la imagen de su DSL) y `PYTHONPATH=src:benchmarks`: no hace falta reconstruir al
  cambiar código. Los kernels se importan como paquetes:
  `from CBKernels.triton.matmul import matmul`, `from MBKernels.triton.rmsnorm_baseline import rmsnorm`.
- `exp` avisa si la GPU está ocupada y, por defecto, ejecuta antes `benchmarks/validar_gpu.py`.

## Añadir un kernel / benchmark

1. Kernel en `src/<CB|MB>Kernels/<dsl>/<nombre>.py` (a nivel de módulo).
2. Benchmark: copiar `benchmarks/plantilla.py` como
   `benchmarks/<CB|MB>Kernels/<dsl>/run_<familia>[_<variante>].py`. Se lanza por su nombre y
   la **carpeta decide la imagen** (triton → triton-tlx, gluon, cutlass, helion). El nombre
   debe ser único en todo `benchmarks/` (identifica sus resultados): p. ej. `run_matmul_helion.py`.
   El prefijo decide la tabla consolidada: `run_matmul_*` → `results/matmul_metrics.csv`,
   `run_rmsnorm*` → `results/rmsnorm_metrics.csv`, `run_softmax*` → `results/softmax_metrics.csv`...
3. Validar la salida frente a PyTorch con `benchmarks/validation.py`
   (`comprobar_matmul`, `comprobar_rmsnorm`, `comprobar`, `error_rel`).
   **Todo cálculo matricial va en tensor cores** (`tl.dot`): llamar a
   `comun.exigir_tensor_cores(kernel_compilado)`, que aborta si el PTX no tiene
   `mma.sync`/`wgmma`/`tcgen05.mma`. Importar `comun` activa TF32 para fp32 en cuBLAS.
   Excepción: operadores sin producto matricial (p. ej. RMSNorm) no llevan MMA y se miden en GB/s.
4. Medir con `comun.medir(fn, flops=..., bytes_movidos=...)` y terminar con
   `comun.guardar(filas, parametros=vars(args), resumen=...)`.
5. `bash ~/hennessy/tfm_entorno/ejecutar.sh encolar exp run_<...> [args]`.

## Informes para la memoria

**Todo benchmark** genera al terminar `tabla.md`, `tabla.tex`, `grafica.png` y `grafica.pdf`
en su carpeta de resultados, copia la última a `docs/TFM/resultados/<benchmark>.{md,tex}` (+
`figuras/`) y regenera `results/<familia>_metrics.csv`: lo hace `comun.guardar()` vía
`benchmarks/informe.py`. `run_matmul` usa su propio informe (`informe_matmul.py`, Triton vs cuBLAS).

Regenerar cualquier ejecución (en el login, sin GPU):

```bash
python3.11 benchmarks/informe.py results/<benchmark>/ultimo
```

Las secciones redactadas para la memoria del TFM están en `docs/TFM/*.md` y `docs/TFM/*.tex`
(includables con `\input`): `resumen`, `plataforma` (hardware y software exactos de las pruebas), `tma`, `fp8`, `sparsity` y
`procedencia_kernels` (origen de cada kernel).
