# Plataforma experimental

*Descripción exacta del hardware y el software de todas las pruebas. Los datos se han leído de los propios nodos (`lscpu`, `/proc/meminfo`, `nvidia-smi`, `/proc/driver/nvidia/version` y el historial de `apt` y `dpkg`), de las propiedades de la GPU que da el runtime de CUDA dentro del contenedor y del `meta.json` que guarda cada ejecución. Fecha de recogida: 2 de octubre de 2026.*

## Nodos

Las pruebas de esta memoria se han hecho en dos nodos con **el mismo hardware**: un sistema NVIDIA GB10 (Grace Blackwell) con memoria unificada. Ambos tienen instalados los paquetes de sistema `nvidia-spark-*` de NVIDIA DGX Spark. Se usan mediante Slurm (particiones `<nodo>-benchmark` y `<nodo>-test`) y Docker.

| nodo | pruebas |
|:---|:---|
| hennessy | todos los matmul (`run_matmul`, `_tma`, `_fp8`, `_sparsity`) y RMSNorm en Triton (`run_rmsnorm_triton`) |
| patterson | RMSNorm en Gluon, Helion y CUTLASS (`run_rmsnorm_gluon`, `_helion`, `_cutlass`) |

## Entornos de ejecución

Los resultados se guardan por **entorno** en `results/<entorno>/` y `docs/TFM/resultados/<entorno>/`. Las tablas consolidadas `results/*_metrics.csv` llevan la columna `maquina`. Cada entorno es una combinación de nodo y software de sistema:

| entorno | nodo | driver / kernel | periodo |
|:---|:---|:---|:---|
| `hennessy-580` | hennessy | 580.159.03 / Linux 6.17.0-1021-nvidia | 29–30 sep 2026 (resultados de esta memoria) |
| `hennessy` | hennessy | 610.57.04 / Linux 7.0.0-1019-nvidia | desde el 2 oct 2026 |
| `patterson` | patterson | 580.126.09 / Linux 6.17.0-1008-nvidia | desde el 30 sep 2026 |
| `pascal` | pascal | **NVIDIA GeForce RTX 5090** (sm_120), no un GB10; driver y kernel no registrados (ejecuciones del job 20499, anteriores al registro) | 2 oct 2026 |

Por defecto el entorno es el nombre del nodo. Con `TFM_MAQUINA=<nombre>` se separa otro entorno en el mismo nodo; así se hizo `hennessy-580`, para las ejecuciones anteriores al cambio de driver. Desde el 2 de octubre cada `meta.json` registra también `driver_nvidia` y `kernel_linux`.

## GPU

Valores del runtime de CUDA (`torch.cuda.get_device_properties` y Triton) y de `nvidia-smi`, idénticos en los dos nodos:

| propiedad | valor |
|:---|:---|
| GPU | NVIDIA GB10, arquitectura Blackwell |
| *compute capability* | 12.1 (`sm_121`) |
| SMs | 48 |
| hilos residentes por SM | 1536 |
| registros por SM | 65 536 (32 bits) |
| memoria compartida por SM | 100 KiB (102 400 B) |
| memoria compartida por bloque | 48 KiB por defecto; 99 KiB (101 376 B) con *opt-in* |
| caché L2 | 24 MiB (25 165 824 B) |
| tamaño de warp | 32 |
| reloj SM | máx. 3003 MHz (`nvidia-smi`); 2418 MHz según CUDA (reloj de aplicación por defecto) |
| tipo | integrada (`is_integrated = 1`): comparte la memoria del sistema con la CPU |

## Memoria

| propiedad | valor |
|:---|:---|
| tipo | LPDDR5X unificada CPU/GPU |
| bus | 256 bits, 8533 MT/s |
| ancho de banda teórico | 256/8 × 8533·10⁶ = **273 GB/s** (referencia del "% del pico" en RMSNorm) |
| visible para CUDA | 128 520 806 400 B (119,7 GiB) |
| `MemTotal` del sistema | 127 598 608 kB (hennessy) · 125 508 600 kB (patterson) |

## CPU

| propiedad | valor |
|:---|:---|
| arquitectura | Arm, aarch64 |
| núcleos | 20 (1 hilo por núcleo): 10 Cortex-X925 (máx. 3,9 GHz) + 10 Cortex-A725 (máx. 2,808 GHz) |
| caché | L1d 1,3 MiB (20 inst.), L2 25 MiB (20 inst.), L3 24 MiB (2 inst.) |
| NUMA | 1 nodo |

## Software del sistema durante las pruebas

Los dos nodos **no** tenían el mismo driver. Los datos de hennessy son los de antes de su actualización del 30 de septiembre, porque todas sus pruebas son anteriores; se comprueba con el historial de `apt` y los arranques (`last -x reboot`).

| | hennessy | patterson |
|:---|:---|:---|
| periodo de las pruebas | 29 sep 2026 09:35 – 30 sep 14:23 | 30 sep 2026 15:55 – 16:41 |
| sistema operativo | Ubuntu 24.04 LTS | Ubuntu 24.04.4 LTS |
| kernel Linux | 6.17.0-1021-nvidia (arrancado el 21 sep) | 6.17.0-1008-nvidia (arrancado el 9 sep) |
| driver NVIDIA | **580.159.03** (módulo de kernel abierto) | **580.126.09** (módulo de kernel abierto) |
| modo de direccionamiento GPU | no registrado | ATS |
| Docker | no registrado | 29.1.3 (nvidia-container-toolkit 1.18.2) |

> **Aviso: hennessy cambió después de las pruebas.** El 30 de septiembre, entre las 16:39 y las 18:17, se actualizó a **Linux 7.0.0-1019-nvidia** y al **driver 610.57.04**, con Ubuntu 24.04.5 y Docker 29.6.2 (toolkit 1.20.1). Desde entonces `nvidia-smi` muestra el modo de direccionamiento `None` en lugar de `ATS`, y CUDA 13.3 en el driver. Las ejecuciones en hennessy posteriores al 30 de septiembre no son directamente comparables con las anteriores.

## Software en los contenedores

Todas las imágenes salen del mismo `Dockerfile` (una por DSL) sobre una base común, y todas fijan el mismo Triton.

| componente | versión |
|:---|:---|
| imagen base | NGC PyTorch 25.10, `nvcr.io/nvidia/pytorch:25.10-py3` (`sha256:42263b2424fc…`) |
| PyTorch | 2.9.0a0+145a3a7bda.nv25.10 |
| CUDA (toolkit) | 13.0 (`nvcc` V13.0.88) |
| cuDNN | 9.14.0 |
| Python | 3.12.3 |
| Triton | 3.8.0 (PyPI, igual en todas las imágenes) |
| triton-utlx (TLX) | 3.8.0.post1 (solo en la imagen triton-tlx) |
| Helion | 1.4.1.dev349+g256f2c8c5 (hennessy) · 1.4.1.dev368+g0f3b2423e (patterson) |
| CUTLASS | v4.8.0 (en `/opt/cutlass`), CMake 3.31.6 |

Imágenes concretas (el `meta.json` de cada ejecución guarda el id completo):

| imagen | id | nodo | ejecuciones |
|:---|:---|:---|:---|
| `tfm:ngc-arm64` (imagen única anterior) | `sha256:202e6ba1ee16…` | hennessy | matmul, jobs 20388–20402 |
| `tfm-triton-tlx:ngc-arm64` | `sha256:29f39b65a482…` | hennessy | matmul y RMSNorm Triton, job 20423 |
| `tfm-gluon:ngc-arm64` | `sha256:6fbc8b6f4ec1…` | patterson | RMSNorm Gluon, job 20435 |
| `tfm-helion:ngc-arm64` | `sha256:365cc5ee2a5f…` | patterson | RMSNorm Helion, job 20435 |
| `tfm-cutlass:ngc-arm64` | `sha256:147c547a4cb1…` | patterson | RMSNorm CUTLASS, job 20435 |

## Metodología de medida

- **Medida:** `triton.testing.do_bench`, con 100 ms de calentamiento y 500 ms de repetición. Se informa la mediana, con los percentiles 20 y 80. `do_bench` vacía la L2 entre repeticiones, así que no se mide la caché.
- **Primera llamada:** la compilación JIT y el autotuning se hacen en la primera llamada, que queda fuera de la medida y se registra aparte.
- **Corrección:** cada kernel se valida frente a PyTorch antes de medirlo (`benchmarks/validation.py`). En matmul se comprueba además en el PTX que se usan tensor cores.
- **Relojes:** **no se han fijado**. Los relojes de aplicación están inactivos (`Applications Clocks Setting: Not Active`), el límite de potencia no es configurable (`N/A`) y el modo de persistencia está activado. La GPU funciona con su gestión dinámica de frecuencia por defecto.
- **Aislamiento:** cada prueba corre en un contenedor efímero con la GPU completa. `ejecutar.sh` avisa si hay otros procesos usando la GPU.
